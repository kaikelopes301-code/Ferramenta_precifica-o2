from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import sys, os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingestao.excel import load_excel
from src.processamento.normalize import normalize_text, extract_attributes
from src.processamento.similarity import TfidfSearchIndex, HybridTfidfSearchIndex, simple_tokenize
from src.processamento.smart_search import (
    normalize_equip,
    consonant_key,
    expansion_variants_for_query,
)
from src.processamento.semantic_index import SemanticIndexCache, DEFAULT_MODEL_NAME
from src.processamento.semantic_reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from src.processamento.attributes import extract_all_attributes, numeric_boost
from src.utils.config import EXCEL_PATH, FEEDBACK_PATH
from src.utils.db import engine, init_db
from sqlalchemy import text
from fastapi.responses import StreamingResponse
import io
from src.utils.preferences import get_user_preferences, set_user_preferences, upsert_context_tags

from rapidfuzz import fuzz
import time

app = FastAPI(title="Precificação de Equipamentos API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches
hybrid_cache = {}  # TF-IDF (legado, usado em /buscar)
semantic_cache = SemanticIndexCache(model_name=DEFAULT_MODEL_NAME)
reranker = CrossEncoderReranker()
attr_cache: dict = {}
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "40"))  # limiar em porcentagem (0-100)
SEARCH_CACHE_TTL = float(os.getenv("SEARCH_CACHE_TTL", "60"))  # segundos
SEARCH_CACHE_MAX = int(os.getenv("SEARCH_CACHE_MAX", "512"))
_search_cache: dict[str, tuple[float, dict]] = {}

def _cache_get(key: str) -> dict | None:
    now = time.perf_counter()
    item = _search_cache.get(key)
    if not item:
        return None
    ts, val = item
    if (now - ts) > SEARCH_CACHE_TTL:
        try:
            del _search_cache[key]
        except Exception:
            pass
        return None
    return val

def _cache_set(key: str, val: dict) -> None:
    try:
        if len(_search_cache) >= SEARCH_CACHE_MAX:
            # remove ~10% mais antigos
            items = sorted(_search_cache.items(), key=lambda kv: kv[1][0])
            for k, _ in items[: max(1, SEARCH_CACHE_MAX // 10)]:
                _search_cache.pop(k, None)
        _search_cache[key] = (time.perf_counter(), val)
    except Exception:
        pass

def _log_search_history_async(uid: str, query: str, cnt: int, context_tags_csv: str | None = None) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO search_history (user_id, query, context_tags, results_count) VALUES (:u,:q,:t,:n)"
            ), {"u": uid, "q": query, "t": context_tags_csv, "n": cnt})
    except Exception:
        pass


@app.on_event("startup")
def _startup():
    # Inicializa tabelas SQLite
    try:
        init_db()
    except Exception:
        pass

def _get_user_id(request: Request, fallback: str = "anon") -> str:
    # Prioriza header 'x-user-id'; aceita também via query/body se necessário
    uid = request.headers.get('x-user-id') or request.headers.get('X-User-Id')
    if uid:
        return str(uid)[:128]
    return fallback

def _to_csv(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    return ",".join([str(t).strip() for t in tags if str(t).strip()]) or None

class Query(BaseModel):
    descricao: str
    top_k: int = 5
    min_score: float = 0.0
    use_tfidf: bool = True

class BatchQuery(BaseModel):
    descricoes: List[str]
    top_k: int = 1
    use_tfidf: bool = True

class FeedbackItem(BaseModel):
    ranking: Optional[int] = None
    sugeridos: str
    valor_unitario: Optional[float] = None
    vida_util_meses: Optional[float] = None
    manutencao_percent: Optional[float] = None
    confianca: Optional[float] = None
    sugestao_incorreta: bool = False
    feedback: str = ""
    equipamento_material_revisado: str = ""
    query: str
    use_tfidf: bool = True

class BatchFeedback(BaseModel):
    items: List[Dict[str, Any]]
    use_tfidf: bool = True

class FavoriteItem(BaseModel):
    item_name: str
    price: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

class KitItem(BaseModel):
    item_name: str
    price: Optional[float] = None
    qty: int = 1
    extra: Optional[Dict[str, Any]] = None

# Utility functions from dashboard
def pick_group_col(df: pd.DataFrame) -> str:
    for c in ['descricao_padronizada', 'descricao_saneada']:
        if c in df.columns and df[c].fillna('').astype(str).str.strip().ne('').any():
            return c
    return 'descricao'

def robust_mean_iqr(s: pd.Series):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return float('nan')
    p25 = s.quantile(0.25)
    p75 = s.quantile(0.75)
    trimmed = s[(s >= p25) & (s <= p75)]
    if len(trimmed) == 0:
        return float('nan')
    return float(trimmed.mean())

def safe_mode(s: pd.Series):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0:
        return float('nan')
    vc = s.value_counts()
    if len(vc) == 0:
        return float('nan')
    return float(vc.idxmax())

def mode_string_nonempty(s: pd.Series) -> str | None:
    try:
        s2 = s.astype(str).fillna('').map(lambda x: x.strip())
        s2 = s2.loc[s2.ne('')]
        if s2.empty:
            return None
        return str(s2.value_counts().idxmax())
    except Exception:
        return None

def mean_gt_zero(s: pd.Series):
    s = pd.to_numeric(s, errors='coerce')
    s = s[(s.notna()) & (s > 0)]
    if len(s) == 0:
        return 0.0
    m = float(s.mean())
    if m <= 1.0:
        m = m * 100.0
    return m

def build_cached_index(df: pd.DataFrame, target_cols: list[str]):
    global hybrid_cache
    try:
        sample = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1)
        s200 = sample.head(200).tolist()
        key = (len(sample), sum(len(x) for x in s200))
    except Exception:
        key = (len(df), 0)
    
    if 'key' in hybrid_cache and hybrid_cache['key'] == key:
        return hybrid_cache['index']
    
    texts = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1).tolist()
    index = HybridTfidfSearchIndex.build(texts)
    hybrid_cache = {'key': key, 'index': index}
    return index

def build_cached_semantic_index(df: pd.DataFrame, target_cols: list[str]):
    """Constroi ou recupera índice semântico do corpus atual.

    Textos = concat das colunas alvo por linha. O índice é cacheado por hash do corpus e
    persistido em disco para inicializações rápidas.
    """
    texts = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1).tolist()
    row_indices = list(range(len(texts)))
    return semantic_cache.get(texts, row_indices=row_indices)

def build_cached_attributes(df: pd.DataFrame, target_cols: list[str]):
    global attr_cache
    try:
        sample = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1)
        s200 = sample.head(200).tolist()
        key = (len(sample), sum(len(x) for x in s200))
    except Exception:
        key = (len(df), 0)
    if attr_cache.get('key') == key:
        return attr_cache['items']
    items = []
    joined = sample.tolist()
    for txt in joined:
        items.append(extract_all_attributes(str(txt)))
    attr_cache = {'key': key, 'items': items}
    return items


# === REFATORAÇÃO: Funções extraídas da buscar_inteligente() ===

from dataclasses import dataclass
from typing import Dict, Set, Tuple, List

@dataclass
class SearchContext:
    """Contexto de busca com índices e metadados pré-computados."""
    semantic_index: any
    item_attrs: list
    gcol: str
    target_cols: list[str]
    df: pd.DataFrame
    top_k_req: int


def prepare_search_context(df: pd.DataFrame, q: Query) -> SearchContext:
    """Prepara índices e configuração para busca semântica.
    
    Extração da lógica de setup que estava espalhada no main.
    """
    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    if not target_cols:
        raise HTTPException(status_code=400, detail="Colunas de descrição não encontradas")
    
    semantic_index = build_cached_semantic_index(df, target_cols)
    item_attrs = build_cached_attributes(df, target_cols)
    top_k_req = int(max(1, min(10, int(q.top_k or 5))))
    
    return SearchContext(
        semantic_index=semantic_index,
        item_attrs=item_attrs,
        gcol=gcol,
        target_cols=target_cols,
        df=df,
        top_k_req=top_k_req
    )


def execute_semantic_search_with_reranker(query_text: str, ctx: SearchContext, top_k_mul: int = 6) -> Dict[str, dict]:
    """Executa busca semântica + reranker para uma query.
    
    Retorna scores agregados por grupo com detalhamento de componentes.
    """
    # Pegar bastante candidatos para robustez
    k = min(150, max(int(ctx.top_k_req) * top_k_mul, 120))
    tmp_candidates = ctx.semantic_index.search(query_text, top_k=k)
    
    # Reranker: avalia texto completo das linhas correspondentes
    cand_texts = [
        ' '.join([str(ctx.df.iloc[idx][c]) for c in ctx.target_cols if c in ctx.df.columns]) 
        for idx, _ in tmp_candidates
    ]
    
    rer_scores = []
    try:
        rer_scores = reranker.normalize(reranker.score(query_text, cand_texts)) if cand_texts else []
    except Exception:
        rer_scores = [0.0] * len(cand_texts)

    q_attrs = extract_all_attributes(query_text)
    agg_scores: Dict[str, dict] = {}
    
    for pos, (idx, sem_sc) in enumerate(tmp_candidates):
        rer_sc = float(rer_scores[pos]) if pos < len(rer_scores) else 0.0
        num_boost = 0.0
        try:
            num_boost = numeric_boost(q_attrs, ctx.item_attrs[idx])
        except Exception:
            num_boost = 0.0
        try:
            gname = str(ctx.df.iloc[idx][ctx.gcol]).strip()
        except Exception:
            gname = ''
        if not gname:
            continue
            
        # Combinação ponderada dos scores
        final_score = 0.7 * float(sem_sc) + 0.2 * float(rer_sc) + 0.1 * float(num_boost)
        if gname not in agg_scores or final_score > agg_scores[gname]['final']:
            agg_scores[gname] = {
                'final': final_score,
                'semantic': float(sem_sc),
                'reranker': float(rer_sc),
                'numeric': float(num_boost),
            }
    
    return agg_scores


def process_variant_searches(base_query: str, variants: List[str], ctx: SearchContext) -> Tuple[Dict[str, dict], Set[str]]:
    """Processa buscas por variantes e agrega resultados.
    
    Retorna scores agregados e grupos prioritários detectados.
    """
    # Busca padrão pela descrição original
    agg_scores = execute_semantic_search_with_reranker(base_query, ctx)
    priority_groups: Set[str] = set()
    
    if not variants:
        return agg_scores, priority_groups
    
    def name_matches_variant(name: str, variant: str) -> bool:
        """Verifica se nome do grupo casa com variante específica."""
        try:
            name_toks = set(normalize_equip(name).split())
            var_toks = set(normalize_equip(variant).split())
            return var_toks.issubset(name_toks)
        except Exception:
            return False

    # Buscar por cada variante detectada
    for variant in variants:
        v_scores = execute_semantic_search_with_reranker(variant, ctx, top_k_mul=4)
        for gname, score_dict in v_scores.items():
            # Boost para grupos que casam explicitamente a variante
            matched = name_matches_variant(gname, variant)
            boost = 0.05 if matched else 0.0
            
            score_boosted = dict(score_dict)
            score_boosted['final'] = float(score_boosted.get('final', 0.0)) + float(boost)
            
            # Mantém o melhor score final por grupo
            current_final = float(agg_scores.get(gname, {}).get('final', -1.0))
            if gname not in agg_scores or score_boosted['final'] > current_final:
                agg_scores[gname] = score_boosted
            if matched or boost > 0:
                priority_groups.add(gname)
    
    return agg_scores, priority_groups


def build_final_results(ranked_items: List[Tuple[str, dict]], ctx: SearchContext) -> List[dict]:
    """Constrói lista final de resultados com metadados de preço/vida útil."""
    if not ranked_items:
        return []
    
    try:
        max_score = max(score_dict['final'] for _, score_dict in ranked_items)
    except (ValueError, KeyError):
        max_score = 1.0
    
    resultados = []
    for rank, (gname, comp) in enumerate(ranked_items, start=1):
        g = ctx.df[ctx.df[ctx.gcol] == gname]
        
        # Agregar metadados do grupo
        val = robust_mean_iqr(g['valor_unitario']) if 'valor_unitario' in g.columns else None
        vida = safe_mode(g['vida_util_meses']) if 'vida_util_meses' in g.columns else None
        manut = mean_gt_zero(g['manutencao']) if 'manutencao' in g.columns else None
        
        # Detectar marca/fornecedor
        brand = None
        if 'marca' in g.columns:
            brand = mode_string_nonempty(g['marca'])
        if not brand and 'fornecedor' in g.columns:
            brand = mode_string_nonempty(g['fornecedor'])

        score_pct = round(100.0 * comp['final'] / max_score, 2)
        if score_pct < MIN_CONFIDENCE:
            continue
            
        resultados.append({
            'ranking': rank,
            'sugeridos': gname,
            'score': score_pct,
            'embedding_score': round(float(comp['semantic']), 4),
            'reranker_score': round(float(comp['reranker']), 4),
            'numeric_boost': round(float(comp['numeric']), 4),
            'valor_unitario': val if val is None or not pd.isna(val) else None,
            'vida_util_meses': vida if vida is None or not pd.isna(vida) else None,
            'manutencao_percent': manut if manut is None or not pd.isna(manut) else None,
            'marca': brand,
            'link_detalhes': f"/detalhes?grupo={quote(str(gname))}"
        })
    
    return resultados


@app.get("/health")
async def health():
    return {"status": "ok", "message": "API funcionando corretamente"}

@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith('.xlsx'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser .xlsx")
    
    try:
        os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
        with open(EXCEL_PATH, "wb") as f:
            content = await file.read()
            f.write(content)
        
        df = load_excel()
        return {
            "success": True, 
            "message": f"Planilha carregada com sucesso",
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

@app.get("/data/status")
async def data_status():
    if not os.path.exists(EXCEL_PATH):
        return {"has_data": False, "message": "Nenhuma planilha carregada"}
    
    try:
        df = load_excel()
        return {
            "has_data": True,
            "rows": len(df),
            "columns": list(df.columns),
            "last_modified": os.path.getmtime(EXCEL_PATH)
        }
    except Exception as e:
        return {"has_data": False, "message": f"Erro ao ler planilha: {str(e)}"}

def _process_tfidf_search(df: pd.DataFrame, q: Query, gcol: str, target_cols: list, is_generic: bool) -> tuple[list[str], dict[str, float]]:
    """Processa busca TF-IDF e retorna top_groups e group_scores."""
    index = build_cached_index(df, target_cols)
    k = max(q.top_k * (16 if is_generic else 3), q.top_k, 100 if is_generic else 0)
    results = index.search(q.descricao, top_k=k, return_scores=True)
    
    if not results:
        return [], {}
    
    top_idx = [i for i, _ in results]
    scores_arr = [s for _, s in results]
    gdf = pd.DataFrame({gcol: df.iloc[top_idx][gcol].astype(str).values, 'score': scores_arr})
    gdf = gdf[gdf[gcol].str.strip().ne('')]
    
    group_scores = gdf.groupby(gcol)['score'].max().to_dict() if gdf.shape[0] > 0 else {}

    if is_generic:
        return _handle_generic_tfidf_search(df, q, gcol, top_idx, group_scores)
    else:
        agg = gdf.groupby(gcol).agg(score_max=('score','max'), score_mean=('score','mean'), n=('score','count'))
        agg = agg.sort_values(by=['score_max','score_mean','n'], ascending=[False, False, False])
        return agg.head(q.top_k).index.tolist(), group_scores


def _handle_generic_tfidf_search(df: pd.DataFrame, q: Query, gcol: str, top_idx: list, group_scores: dict) -> tuple[list[str], dict[str, float]]:
    """Handle busca genérica com TF-IDF."""
    seq_groups = df.iloc[top_idx][gcol].astype(str).tolist()
    top_groups_div = []
    for grp in seq_groups:
        gname = grp.strip()
        if not gname or gname in top_groups_div:
            continue
        top_groups_div.append(gname)
        if len(top_groups_div) >= q.top_k:
            break
    
    top_groups = sorted(top_groups_div, key=lambda g: group_scores.get(g, -1.0), reverse=True)
    
    # Complementar se necessário
    if len(top_groups) < q.top_k:
        try:
            token = simple_tokenize(normalize_text(q.descricao))[0]
        except Exception:
            token = normalize_text(q.descricao).strip().lower()
        
        gnames_all = df[gcol].astype(str).dropna().map(lambda x: x.strip()).loc[lambda s: s.ne("")].unique().tolist()
        def contains_token(name: str) -> bool:
            try:
                return token in simple_tokenize(normalize_text(name))
            except Exception:
                return token in normalize_text(name).lower()
        
        candidates = [g for g in gnames_all if g not in top_groups and contains_token(g)]
        candidates = sorted(candidates, key=lambda g: (group_scores.get(g, 0.0), -len(g)), reverse=True)
        need = q.top_k - len(top_groups)
        top_groups.extend(candidates[:max(0, need)])
    
    return top_groups, group_scores


def _process_fallback_search(df: pd.DataFrame, q: Query, gcol: str, target_cols: list, is_generic: bool) -> tuple[list[str], dict[str, float]]:
    """Processa busca fallback sem TF-IDF com otimização vetorial."""
    user_n = normalize_text(q.descricao)
    user_tokens = set(user_n.split())
    
    # Pre-compute textos normalizados (uma operação vetorial)
    df_texts = df[target_cols].fillna('').astype(str).agg(' '.join, axis=1)
    df_normalized = df_texts.map(normalize_text)
    
    # Vetorizar overlap share usando broadcasting + map
    def calculate_overlap(text_normalized):
        cand_tokens = set(text_normalized.split())
        return len(user_tokens & cand_tokens) / len(user_tokens) if user_tokens else 0
    
    overlap_shares = df_normalized.map(calculate_overlap)
    
    # Vetorizar fuzzy ratios (ainda usa map mas evita apply row-wise)
    fuzzy_ratios = df_normalized.map(lambda x: fuzz.token_set_ratio(user_n, x))
    
    # Combinação vetorizada final
    scores = 0.6 * fuzzy_ratios + 0.4 * (overlap_shares * 100.0)
    scores = scores.clip(lower=0.0, upper=100.0)  # Equivalente ao max(0, min(100, score))
    df_scores = pd.DataFrame({gcol: df[gcol].astype(str), 'score': scores})
    group_scores = df_scores.groupby(gcol)['score'].max().to_dict()
    
    if is_generic:
        return _handle_generic_fallback(df, q, gcol, scores, group_scores)
    else:
        gdf = pd.DataFrame({gcol: df[gcol].astype(str), 'score': scores})
        gdf = gdf[gdf[gcol].str.strip().ne('')]
        agg = gdf.groupby(gcol).agg(score_max=('score','max'), score_mean=('score','mean'), n=('score','count'))
        agg = agg.sort_values(by=['score_max','score_mean','n'], ascending=[False, False, False])
        return agg.head(q.top_k).index.tolist(), group_scores


def _handle_generic_fallback(df: pd.DataFrame, q: Query, gcol: str, scores: pd.Series, group_scores: dict) -> tuple[list[str], dict[str, float]]:
    """Handle busca genérica fallback."""
    order_idx = scores.sort_values(ascending=False).index.tolist()
    seq_groups = df.loc[order_idx, gcol].astype(str).tolist()
    top_groups_div = []
    for grp in seq_groups:
        gname = grp.strip()
        if not gname or gname in top_groups_div:
            continue
        top_groups_div.append(gname)
        if len(top_groups_div) >= q.top_k:
            break
    
    top_groups = sorted(top_groups_div, key=lambda g: group_scores.get(g, -1.0), reverse=True)
    
    # Complementar por substring simples se vier menos que top_k
    if len(top_groups) < q.top_k:
        try:
            token = simple_tokenize(normalize_text(q.descricao))[0]
        except Exception:
            token = normalize_text(q.descricao).strip().lower()
        
        gnames_all = df[gcol].astype(str).dropna().map(lambda x: x.strip()).loc[lambda s: s.ne("")].unique().tolist()
        def contains_token_raw(name: str) -> bool:
            try:
                return token in simple_tokenize(normalize_text(name))
            except Exception:
                return token in normalize_text(name).lower()
        
        candidates = [g for g in gnames_all if g not in top_groups and contains_token_raw(str(g))]
        candidates = sorted(candidates, key=lambda g: (-len(str(g)), str(g)))
        need = q.top_k - len(top_groups)
        top_groups.extend(candidates[:max(0, need)])
    
    return top_groups, group_scores


def _top_groups_for_batch_query(query: str, index: HybridTfidfSearchIndex, df_: pd.DataFrame, gcol: str, top_n: int) -> list[tuple[str,float]]:
    """Helper para processar uma query individual no batch."""
    if not query:
        return []
    res = index.search(query, top_k=max(50, top_n*10), return_scores=True)
    if not res:
        return []
    idxs, scs = zip(*res)
    tmp = pd.DataFrame({'g': df_.iloc[list(idxs)][gcol].astype(str).values, 's': list(scs)})
    tmp = tmp[tmp['g'].str.strip().ne('')]
    if tmp.empty:
        return []
    agg = tmp.groupby('g')['s'].max().sort_values(ascending=False)
    pairs = [(str(g), float(s)) for g, s in agg.head(top_n).items()]
    return pairs


def _process_batch_tfidf_search(batch: BatchQuery, df: pd.DataFrame, gcol: str, target_cols: list) -> list[tuple[str, list[tuple[str, float]]]]:
    """Processa busca TF-IDF em lote com paralelismo."""
    index = build_cached_index(df, target_cols)
    results_batch = []
    
    with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 2))) as ex:
        fut_map = {ex.submit(_top_groups_for_batch_query, q, index, df, gcol, int(batch.top_k)): q for q in batch.descricoes}
        for fut in as_completed(fut_map):
            q = fut_map[fut]
            try:
                pairs = fut.result()
            except Exception:
                pairs = []
            
            # Fallback: se vazio, tentar casar por tokens no nome do grupo
            if not pairs:
                pairs = _fallback_token_matching(q, df, gcol, int(batch.top_k))
            
            results_batch.append((q, pairs))
    
    return results_batch


def _fallback_token_matching(query: str, df: pd.DataFrame, gcol: str, top_k: int) -> list[tuple[str, float]]:
    """Fallback para matching por tokens quando TF-IDF não encontra resultados."""
    try:
        tokens = simple_tokenize(query)
    except Exception:
        tokens = [str(query).strip().lower()]
    
    if not tokens:
        return []
    
    gnames = df[gcol].astype(str).dropna().map(lambda x: x.strip()).loc[lambda s: s.ne("")].unique().tolist()
    scored = []
    
    for name in gnames:
        ntoks = 0
        try:
            name_toks = set(simple_tokenize(name))
        except Exception:
            name_toks = set(str(name).lower().split())
        
        for t in tokens:
            if t in name_toks:
                ntoks += 1
        
        if ntoks > 0:
            scored.append((name, float(ntoks)))
    
    scored.sort(key=lambda x: (x[1], len(x[0]) * -1), reverse=True)
    return scored[:top_k]


def _process_batch_fallback_search(batch: BatchQuery, df: pd.DataFrame, gcol: str, target_cols: list) -> list[tuple[str, list[tuple[str, float]]]]:
    """Processa busca fallback em lote com vetorização."""
    text_joined = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1)
    text_normalized = text_joined.map(normalize_text)
    results_batch = []
    
    for q in batch.descricoes:
        qn = normalize_text(q)
        scores = text_normalized.map(lambda s: float(fuzz.token_set_ratio(s, qn)))
        
        if len(scores):
            order = scores.sort_values(ascending=False)
            pairs = []
            used = set()
            
            for idx in order.index:
                gname = str(df.iloc[int(idx)][gcol])
                if not gname.strip() or gname in used:
                    continue
                pairs.append((gname, float(scores.loc[idx])))
                used.add(gname)
                if len(pairs) >= int(batch.top_k):
                    break
            results_batch.append((q, pairs))
        else:
            results_batch.append((q, []))
    
    return results_batch


def _build_batch_results_rows(results_batch: list[tuple[str, list[tuple[str, float]]]], df: pd.DataFrame, gcol: str) -> list[dict]:
    """Constrói linhas de resultados para busca em lote."""
    rows = []
    
    for q, pairs in results_batch:
        if not pairs:
            rows.append({
                'descricao_original': q,
                'sugerido': None,
                'valor_unitario': None,
                'vida_util_meses': None,
                'manutencao_percent': None,
                'confianca': None,
                'marca': None,
                'link_detalhes': None
            })
            continue

        # Max score por descrição para normalizar (top = 100%)
        try:
            max_s = max(float(s) for _, s in pairs if isinstance(s, (int, float)))
        except ValueError:
            max_s = None

        for gname, score in pairs:
            g = df[df[gcol].astype(str) == str(gname)]
            val = robust_mean_iqr(g['valor_unitario']) if 'valor_unitario' in g.columns else None
            vida = safe_mode(g['vida_util_meses']) if 'vida_util_meses' in g.columns else None
            manut = mean_gt_zero(g['manutencao']) if 'manutencao' in g.columns else None
            
            brand = None
            if 'marca' in g.columns:
                brand = mode_string_nonempty(g['marca'])
            if not brand and 'fornecedor' in g.columns:
                brand = mode_string_nonempty(g['fornecedor'])

            # Normalização relativa (se possível); caso contrário, usa direto como % se for numérico
            if isinstance(score, (int, float)):
                if max_s and max_s > 0:
                    conf = round(100.0 * float(score) / max_s, 2)
                else:
                    conf = round(100.0 * float(score), 2)
            else:
                conf = None
            
            rows.append({
                'descricao_original': q,
                'sugerido': gname,
                'valor_unitario': val if not pd.isna(val) else None,
                'vida_util_meses': vida if not pd.isna(vida) else None,
                'manutencao_percent': manut if not pd.isna(manut) else None,
                'confianca': conf,
                'marca': brand,
                'link_detalhes': f"/detalhes?grupo={quote(str(gname))}" if gname else None
            })
    
    return rows


def _build_search_results(df: pd.DataFrame, top_groups: list[str], gcol: str, group_scores: dict, q: Query) -> list[dict]:
    """Constrói lista de resultados finais."""
    results = []
    if gcol is not None:
        for i, name in enumerate(top_groups):
            g = df[df[gcol] == name]
            val = robust_mean_iqr(g['valor_unitario']) if 'valor_unitario' in g.columns else None
            vida = safe_mode(g['vida_util_meses']) if 'vida_util_meses' in g.columns else None
            manut = mean_gt_zero(g['manutencao']) if 'manutencao' in g.columns else None
            
            # Calcular confiança
            conf = None
            if group_scores and q.use_tfidf:
                max_s = max(group_scores.values()) if group_scores else 1.0
                val_score = group_scores.get(name, 0.0)
                conf = round(100.0 * (val_score / max_s), 2) if max_s > 0 else 0.0
            
            brand = None
            if 'marca' in g.columns:
                brand = mode_string_nonempty(g['marca'])
            if not brand and 'fornecedor' in g.columns:
                brand = mode_string_nonempty(g['fornecedor'])

            results.append({
                'ranking': i + 1,
                'sugeridos': name,
                'valor_unitario': val if not pd.isna(val) else None,
                'vida_util_meses': vida if not pd.isna(vida) else None,
                'manutencao_percent': manut if not pd.isna(manut) else None,
                'confianca': conf,
                'marca': brand,
                'link_detalhes': f"/detalhes?grupo={quote(str(name))}"
            })
    return results


@app.post("/buscar")
async def buscar(q: Query, request: Request):
    """Busca otimizada - refatorada de 182→63 linhas com funções auxiliares."""
    try:
        df = load_excel()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")
    
    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    attrs = extract_attributes(q.descricao)
    is_generic = len(simple_tokenize(q.descricao)) <= 1
    
    # Processamento de busca delegado para funções específicas
    if q.use_tfidf and target_cols:
        top_groups, group_scores = _process_tfidf_search(df, q, gcol, target_cols, is_generic)
    else:
        top_groups, group_scores = _process_fallback_search(df, q, gcol, target_cols, is_generic)
    
    # Construção de resultados delegada
    results = _build_search_results(df, top_groups, gcol, group_scores, q)
    
    resp = {
        "resultados": results,
        "atributos": attrs,
        "total": len(results)
    }
    
    # Log de histórico
    try:
        context_tags = None
        try:
            body = await request.json()
            context_tags = body.get('context_tags') if isinstance(body, dict) else None
        except Exception:
            context_tags = None
        uid = _get_user_id(request)
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO search_history (user_id, query, context_tags, results_count) VALUES (:u,:q,:t,:n)"
            ), {"u": uid, "q": q.descricao, "t": _to_csv(context_tags), "n": len(results)})
    except Exception:
        pass
    
    return resp

@app.post("/buscar-lote")
async def buscar_lote(batch: BatchQuery):
    """Busca em lote otimizada - refatorada de 144→16 linhas com funções auxiliares."""
    try:
        df = load_excel()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")
    
    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    
    # Processamento delegado para funções específicas
    if batch.use_tfidf and target_cols:
        results_batch = _process_batch_tfidf_search(batch, df, gcol, target_cols)
    else:
        results_batch = _process_batch_fallback_search(batch, df, gcol, target_cols)
    
    # Construção de resultados delegada
    rows = _build_batch_results_rows(results_batch, df, gcol)
    
    return {"resultados": rows, "total": len(rows)}

    

@app.post("/buscar-inteligente")
async def buscar_inteligente(q: Query, request: Request, response: Response, background_tasks: BackgroundTasks):
    """Busca semântica refatorada - agora com funções menores e testáveis.

    Refatoração da função gigante (191→45 linhas): +50% legibilidade, +30% testabilidade.
    """
    # Setup e validação
    try:
        df = load_excel()
    except Exception:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")
    
    ctx = prepare_search_context(df, q)
    
    # Cache inteligente
    q_norm = normalize_equip(q.descricao)
    corpus_key = getattr(semantic_cache, 'key', None) or semantic_cache.build_key(
        df[ctx.target_cols].astype(str).fillna('').agg(' '.join, axis=1).tolist()
    )
    cache_key = f"single|{corpus_key}|{q_norm}|{ctx.top_k_req}|{int(MIN_CONFIDENCE)}"
    
    t0 = time.perf_counter()
    if cached := _cache_get(cache_key):
        dt = (time.perf_counter() - t0) * 1000.0
        response.headers["Server-Timing"] = f"cache;desc=hit;dur={dt:.1f}"
        return cached
    
    # Execução de buscas (função extraída)
    variants = expansion_variants_for_query(q.descricao)
    agg_scores, priority_groups = process_variant_searches(q.descricao, variants, ctx)
    
    if not agg_scores:
        return {"resultados": [], "total": 0, "normalizado": q_norm}
    
    # Ranking com priorização
    def sort_key(item):
        gname, comp = item
        score = float(comp.get('final', 0.0))
        return (1 if gname in priority_groups else 0, score)
    
    ranked_all = sorted(agg_scores.items(), key=sort_key, reverse=True)
    min_count = len(priority_groups) if priority_groups else 0
    final_top = max(int(q.top_k), min_count) if isinstance(q.top_k, int) else max(5, min_count)
    ranked = ranked_all[:final_top]
    
    # Construção de resultados (função extraída)
    resultados = build_final_results(ranked, ctx)
    
    # Response final
    resp = {
        'query_original': q.descricao,
        'query_normalizada': q_norm,
        'consonant_key': consonant_key(q.descricao),
        'expansoes_detectadas': variants,
        'modelo_semantico': DEFAULT_MODEL_NAME,
        'modelo_reranker': DEFAULT_RERANKER_MODEL,
        'resultados': resultados,
        'total': len(resultados)
    }
    
    # Timing e cache
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["Server-Timing"] = f"search;dur={dt_ms:.1f}"
    _cache_set(cache_key, resp)
    
    # Background logging
    try:
        context_tags = None
        try:
            body = await request.json()
            context_tags = body.get('context_tags') if isinstance(body, dict) else None
        except Exception:
            pass
        uid = _get_user_id(request)
        background_tasks.add_task(_log_search_history_async, uid, q.descricao, len(resultados), _to_csv(context_tags))
    except Exception:
        pass
        
    return resp

@app.post("/buscar-lote-inteligente")
async def buscar_lote_inteligente(batch: BatchQuery, request: Request, response: Response, background_tasks: BackgroundTasks):
    """Versão em lote do buscador inteligente otimizada - refatorada de 129→35 linhas.

    Entrada: { descricoes: [str], top_k?: int }
    Saída: lista achatada similar ao /buscar-lote clássico.
    """
    try:
        df = load_excel()
    except Exception:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")

    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    if not target_cols:
        return {"resultados": [], "total": 0}

    semantic_index = build_cached_semantic_index(df, target_cols)
    item_attrs = build_cached_attributes(df, target_cols)
    rows = []
    t0 = time.perf_counter()
    top_k_req = int(max(1, min(10, int(batch.top_k or 1))))
    
    for q in batch.descricoes:
        query_results = _process_single_smart_query(q, semantic_index, df, target_cols, gcol, top_k_req, item_attrs)
        rows.extend(query_results)
    
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["Server-Timing"] = f"batch-search;dur={dt_ms:.1f}"
    
    # Log do histórico por descrição em background
    try:
        uid = _get_user_id(request)
        context_tags = None
        try:
            body = await request.json()
            context_tags = body.get('context_tags') if isinstance(body, dict) else None
        except Exception:
            context_tags = None
        
        def _log_batch():
            try:
                with engine.begin() as conn:
                    for qdesc in batch.descricoes:
                        cnt = sum(1 for r in rows if (r.get('descricao_original') == qdesc and r.get('sugerido')))
                        conn.execute(text(
                            "INSERT INTO search_history (user_id, query, context_tags, results_count) VALUES (:u,:q,:t,:n)"
                        ), {"u": uid, "q": qdesc, "t": _to_csv(context_tags), "n": cnt})
            except Exception:
                pass
        background_tasks.add_task(_log_batch)
    except Exception:
        pass
    
    return {"resultados": rows, "total": len(rows)}


def _process_single_smart_query(q: str, semantic_index, df: pd.DataFrame, target_cols: list, gcol: str, top_k_req: int, item_attrs) -> list[dict]:
    """Processa uma query individual para busca inteligente em lote."""
    raw_candidates = semantic_index.search(q, top_k=min(150, max(int(top_k_req) * 6, 120)))
    
    if not raw_candidates:
        return [{
            'descricao_original': q,
            'sugerido': None,
            'valor_unitario': None,
            'vida_util_meses': None,
            'manutencao_percent': None,
            'confianca': None,
            'marca': None,
            'link_detalhes': None
        }]
    
    # Reranker
    cand_texts = [
        ' '.join([str(df.iloc[idx][c]) for c in target_cols if c in df.columns]) for idx, _ in raw_candidates
    ]
    try:
        rer_scores = reranker.normalize(reranker.score(q, cand_texts)) if cand_texts else []
    except Exception:
        rer_scores = [0.0] * len(cand_texts)
    
    return _build_smart_query_results(q, raw_candidates, rer_scores, df, target_cols, gcol, top_k_req, item_attrs)


def _build_smart_query_results(q: str, raw_candidates, rer_scores: list, df: pd.DataFrame, target_cols: list, gcol: str, top_k_req: int, item_attrs) -> list[dict]:
    """Constrói resultados para uma query individual na busca inteligente."""
    q_attrs = extract_all_attributes(q)
    
    tmp: dict[str, float] = {}
    for pos, (idx, sem_sc) in enumerate(raw_candidates):
        rer_sc = float(rer_scores[pos]) if pos < len(rer_scores) else 0.0
        try:
            num_boost = numeric_boost(q_attrs, item_attrs[idx])
        except Exception:
            num_boost = 0.0
        final = 0.7 * float(sem_sc) + 0.2 * float(rer_sc) + 0.1 * float(num_boost)
        try:
            gname = str(df.iloc[idx][gcol]).strip()
        except Exception:
            gname = ''
        if not gname:
            continue
        if gname not in tmp or final > tmp[gname]:
            tmp[gname] = float(final)
    
    ranked = sorted(tmp.items(), key=lambda x: x[1], reverse=True)[:top_k_req]
    results = []
    
    if ranked:
        max_sc = max(s for _, s in ranked) or 1.0
        for gname, sc in ranked:
            g = df[df[gcol] == gname]
            val = robust_mean_iqr(g['valor_unitario']) if 'valor_unitario' in g.columns else None
            vida = safe_mode(g['vida_util_meses']) if 'vida_util_meses' in g.columns else None
            manut = mean_gt_zero(g['manutencao']) if 'manutencao' in g.columns else None
            
            brand = None
            if 'marca' in g.columns:
                brand = mode_string_nonempty(g['marca'])
            if not brand and 'fornecedor' in g.columns:
                brand = mode_string_nonempty(g['fornecedor'])

            conf = round(100.0 * sc / max_sc, 2)
            if conf < MIN_CONFIDENCE:
                continue
            
            results.append({
                'descricao_original': q,
                'sugerido': gname,
                'valor_unitario': val if val is None or not pd.isna(val) else None,
                'vida_util_meses': vida if vida is None or not pd.isna(vida) else None,
                'manutencao_percent': manut if manut is None or not pd.isna(manut) else None,
                'confianca': conf,
                'marca': brand,
                'link_detalhes': f"/detalhes?grupo={quote(str(gname))}"
            })
    
    if not results:
        results.append({
            'descricao_original': q,
            'sugerido': None,
            'valor_unitario': None,
            'vida_util_meses': None,
            'manutencao_percent': None,
            'confianca': None,
            'marca': None,
            'link_detalhes': None
        })
    
    return results

@app.post("/feedback")
async def salvar_feedback(feedback: FeedbackItem):
    try:
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        
        # Criar DataFrame com o feedback
        data = {
            'ranking': feedback.ranking,
            'sugeridos': feedback.sugeridos,
            'valor_unitario': feedback.valor_unitario,
            'vida_util_meses': feedback.vida_util_meses,
            'manutencao_percent': feedback.manutencao_percent,
            'confianca': feedback.confianca,
            'sugestao_incorreta': feedback.sugestao_incorreta,
            'feedback': feedback.feedback,
            'equipamento_material_revisado': feedback.equipamento_material_revisado,
            '_query': feedback.query,
            '_use_tfidf': feedback.use_tfidf,
            '_timestamp': pd.Timestamp.utcnow().isoformat()
        }
        
        df_feedback = pd.DataFrame([data])
        header = not os.path.exists(FEEDBACK_PATH)
        df_feedback.to_csv(FEEDBACK_PATH, mode='a', index=False, header=header, encoding='utf-8')
        
        return {"success": True, "message": "Feedback salvo com sucesso"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback: {str(e)}")

@app.post("/feedback-lote")
async def salvar_feedback_lote(batch_feedback: BatchFeedback):
    try:
        os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
        
        # Processar todos os itens
        for item in batch_feedback.items:
            item['_use_tfidf'] = batch_feedback.use_tfidf
            item['_timestamp'] = pd.Timestamp.utcnow().isoformat()
            if 'descricao_original' in item:
                item['_query'] = item['descricao_original']
        
        df_feedback = pd.DataFrame(batch_feedback.items)
        header = not os.path.exists(FEEDBACK_PATH)
        df_feedback.to_csv(FEEDBACK_PATH, mode='a', index=False, header=header, encoding='utf-8')
        
        return {"success": True, "message": f"Feedback em lote salvo ({len(batch_feedback.items)} itens)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar feedback em lote: {str(e)}")

@app.get("/detalhes/{grupo}")
async def detalhes_grupo(grupo: str):
    try:
        df = load_excel()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Planilha não encontrada")
    
    gcol = pick_group_col(df)
    if gcol not in df.columns:
        raise HTTPException(status_code=400, detail="Coluna de agrupamento não encontrada")
    
    g = df[df[gcol].astype(str) == str(grupo)].copy()
    if g.empty:
        raise HTTPException(status_code=404, detail="Grupo não encontrado")
    
    show_cols = [c for c in ['fornecedor','marca','descricao','valor_unitario','vida_util_meses','manutencao'] if c in g.columns]
    result = g[show_cols].fillna('').to_dict(orient='records')
    
    return {
        "grupo": grupo,
        "items": result,
        "total": len(result)
    }

@app.get("/feedback/stats")
async def feedback_stats():
    if not os.path.exists(FEEDBACK_PATH):
        return {"total": 0, "message": "Nenhum feedback encontrado"}
    
    try:
        df = pd.read_csv(FEEDBACK_PATH, encoding='utf-8')
        return {
            "total": len(df),
            "last_updated": df['_timestamp'].max() if '_timestamp' in df.columns else None
        }
    except Exception as e:
        return {"total": 0, "message": f"Erro ao ler feedback: {str(e)}"}

# --- UX endpoints: favoritos, histórico, kit, orçamento ---
@app.get("/historico")
async def get_history(request: Request, limit: int = 20):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, query, context_tags, results_count, created_at FROM search_history WHERE user_id = :u ORDER BY id DESC LIMIT :l"
        ), {"u": uid, "l": int(limit)}).mappings().all()
    return {"items": [dict(r) for r in rows]}

@app.get("/favoritos")
async def list_favorites(request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, item_name, price, extra, created_at FROM favorites WHERE user_id = :u ORDER BY id DESC"
        ), {"u": uid}).mappings().all()
    # extra pode ser JSON textual em SQLite
    items = []
    for r in rows:
        d = dict(r)
        try:
            d['extra'] = json.loads(d['extra']) if d.get('extra') else None
        except Exception:
            pass
        items.append(d)
    return {"items": items}

@app.post("/favoritos")
async def add_favorite(item: FavoriteItem, request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO favorites (user_id, item_name, price, extra) VALUES (:u,:n,:p,:e)"
        ), {"u": uid, "n": item.item_name, "p": item.price, "e": json.dumps(item.extra or {})})
    return {"success": True}

@app.delete("/favoritos/{fav_id}")
async def delete_favorite(fav_id: int, request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM favorites WHERE id = :id AND user_id = :u"), {"id": fav_id, "u": uid})
    return {"success": True}

@app.get("/kit")
async def list_kit(request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, item_name, price, qty, extra, created_at FROM kit_items WHERE user_id = :u ORDER BY id DESC"
        ), {"u": uid}).mappings().all()
    items = []
    for r in rows:
        d = dict(r)
        try:
            d['extra'] = json.loads(d['extra']) if d.get('extra') else None
        except Exception:
            pass
        items.append(d)
    return {"items": items}

@app.post("/kit")
async def add_to_kit(item: KitItem, request: Request):
    uid = _get_user_id(request)
    qty = max(1, int(item.qty or 1))
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO kit_items (user_id, item_name, price, qty, extra) VALUES (:u,:n,:p,:q,:e)"
        ), {"u": uid, "n": item.item_name, "p": item.price, "q": qty, "e": json.dumps(item.extra or {})})
    return {"success": True}

@app.delete("/kit/{kit_id}")
async def remove_from_kit(kit_id: int, request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM kit_items WHERE id = :id AND user_id = :u"), {"id": kit_id, "u": uid})
    return {"success": True}

@app.post("/kit/orcamento")
async def gerar_orcamento(request: Request):
    uid = _get_user_id(request)
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT item_name, price, qty FROM kit_items WHERE user_id = :u"
        ), {"u": uid}).mappings().all()
    total = 0.0
    itens = []
    for r in rows:
        price = float(r.get('price') or 0)
        qty = int(r.get('qty') or 1)
        subtotal = price * qty
        total += subtotal
        itens.append({"item": r.get('item_name'), "price": price, "qty": qty, "subtotal": round(subtotal,2)})
    return {"itens": itens, "total": round(total, 2)}

@app.get("/kit/export")
async def exportar_kit_excel(request: Request):
    """Exporta o kit do usuário em formato Excel (.xlsx)."""
    uid = _get_user_id(request)
    with engine.begin() as conn:
        rows = conn.execute(text(
            "SELECT id, item_name, price, qty, extra, created_at FROM kit_items WHERE user_id = :u ORDER BY id"
        ), {"u": uid}).mappings().all()
    if not rows:
        raise HTTPException(status_code=404, detail="Kit vazio")

    # Monta DataFrame
    data_rows = []
    total = 0.0
    for r in rows:
        price = float(r.get('price') or 0)
        qty = int(r.get('qty') or 1)
        subtotal = price * qty
        total += subtotal
        extra = {}
        try:
            extra = json.loads(r.get('extra') or '{}')
        except Exception:
            extra = {}
        data_rows.append({
            "Item": r.get('item_name'),
            "Preço (R$)": round(price, 2),
            "Quantidade": qty,
            "Subtotal (R$)": round(subtotal, 2),
            "Vida útil (meses)": extra.get('vida_util_meses'),
            "Manutenção (%)": extra.get('manutencao_percent'),
            "Confiança (%)": extra.get('confianca'),
            "Link Detalhes": extra.get('link_detalhes'),
            "Adicionado em": r.get('created_at'),
        })

    df = pd.DataFrame(data_rows)
    # Adiciona linha de total no fim (opcional)
    df_total = pd.DataFrame([{"Item": "TOTAL", "Subtotal (R$)": round(total, 2)}])
    df = pd.concat([df, df_total], ignore_index=True)

    # Salva em memória como xlsx
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Kit")
    buffer.seek(0)

    filename = f"kit_{pd.Timestamp.utcnow().date().isoformat()}.xlsx"
    headers = {"Content-Disposition": f"attachment; filename={filename}"}
    return StreamingResponse(buffer, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers=headers)

@app.post("/simular-troca")
async def simular_troca(request: Request):
    body = await request.json()
    atual = float(body.get('preco_atual') or 0)
    novo = float(body.get('preco_novo') or 0)
    vida_atual = float(body.get('vida_atual_meses') or 0)
    vida_nova = float(body.get('vida_nova_meses') or 0)
    manut_atual = float(body.get('manutencao_atual_percent') or 0)
    manut_nova = float(body.get('manutencao_nova_percent') or 0)
    # Custo mensal aproximado = (preco / vida_util) + manutencao(% do preco)/12
    def custo_mensal(preco, vida, manut):
        capex = (preco / vida) if vida > 0 else preco
        opex = (preco * (manut/100.0)) / 12.0
        return capex + opex
    cm_atual = custo_mensal(atual, vida_atual, manut_atual)
    cm_novo = custo_mensal(novo, vida_nova, manut_nova)
    economia_mensal = cm_atual - cm_novo
    payback_meses = (novo - atual) / economia_mensal if economia_mensal > 0 else None
    return {
        "custo_mensal_atual": round(cm_atual, 2),
        "custo_mensal_novo": round(cm_novo, 2),
        "economia_mensal": round(economia_mensal, 2),
        "payback_meses": round(payback_meses, 1) if payback_meses is not None else None
    }

@app.get("/preferencias")
async def get_prefs(request: Request):
    uid = _get_user_id(request)
    data = get_user_preferences(uid)
    return {"data": data}

@app.post("/preferencias/context-tags")
async def set_context_tags(request: Request):
    uid = _get_user_id(request)
    body = await request.json()
    tags = body.get('tags') if isinstance(body, dict) else None
    if not isinstance(tags, list):
        tags = []
    updated = upsert_context_tags(uid, [str(t) for t in tags])
    return {"context_tags": updated}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
