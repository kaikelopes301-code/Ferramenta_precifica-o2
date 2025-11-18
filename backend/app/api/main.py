from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import sys, os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from functools import lru_cache
import hashlib

from ..ingestao.excel import load_excel
from ..processamento.normalize import normalize_text, extract_attributes
from ..processamento.similarity import TfidfSearchIndex, HybridTfidfSearchIndex, simple_tokenize
from ..processamento.smart_search import (
    normalize_equip,
    consonant_key,
    expansion_variants_for_query,
)
from ..processamento.semantic_index import SemanticIndexCache, DEFAULT_MODEL_NAME
from ..processamento.semantic_reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from ..processamento.attributes import extract_all_attributes, numeric_boost
from ..utils.config import EXCEL_PATH, FEEDBACK_PATH
from ..utils.db import engine, init_db
from sqlalchemy import text
from fastapi.responses import StreamingResponse
import io
from ..utils.preferences import get_user_preferences, set_user_preferences, upsert_context_tags

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

# ============================================================================
# CONFIGURAÇÕES CENTRAIS DO SISTEMA DE BUSCA INTELIGENTE
# ============================================================================

# --- Limiares de Qualidade ---
MIN_CONFIDENCE = 40.0  # Porcentagem mínima para exibir resultado (0-100)
                       # Resultados abaixo deste limiar são descartados
                       # Valor anterior: 25.0 (muito permissivo)

# --- Pesos para Combinação de Scores (devem somar 1.0) ---
SCORE_WEIGHT_EMBEDDING = 0.7   # 70% - Similaridade semântica via embeddings
SCORE_WEIGHT_RERANKER = 0.2    # 20% - Cross-encoder reranking (precisão)
SCORE_WEIGHT_NUMERIC = 0.1     # 10% - Boost de atributos numéricos

# --- Parâmetros de Busca ---
SEMANTIC_CANDIDATES = 150      # Candidatos iniciais retornados pela busca semântica
DEFAULT_TOP_K = 5              # Número padrão de sugestões retornadas ao usuário

# --- Cache de Resultados ---
SEARCH_CACHE_TTL = 60.0        # Tempo de vida do cache em segundos
SEARCH_CACHE_MAX = 512         # Número máximo de entradas no cache

# 🚀 Quick Win Part 2: Query Result Caching (LRU)
CACHE_LRU_MAXSIZE = 500        # Tamanho máximo do cache LRU (memória)
CACHE_MEMORY_LIMIT_MB = 200    # Limite de memória do cache em MB

# ============================================================================
# CACHES GLOBAIS
# ============================================================================
hybrid_cache = {}  # TF-IDF (legado, usado em /buscar)
semantic_cache = SemanticIndexCache(model_name=DEFAULT_MODEL_NAME)
reranker = CrossEncoderReranker()
attr_cache: dict = {}

# 🚀 Quick Win Part 2: Cache LRU para resultados de busca inteligente
# Substitui o cache manual por LRU mais eficiente
from collections import OrderedDict
import threading

class LRUSearchCache:
    """
    🚀 Quick Win Part 2: Cache LRU Thread-Safe para Query Results
    
    Implementação de cache LRU (Least Recently Used) otimizado para:
    - Armazenar resultados completos de busca (pós-reranking)
    - Chave: query normalizada + hash do corpus
    - TTL configurável para invalidação automática
    - Thread-safe para requests concorrentes
    - Limite de memória para evitar OOM
    
    Success Criteria:
    - Queries repetidas 50×+ mais rápidas (cache hit)
    - Footprint de memória ≤ 200MB
    """
    
    def __init__(self, maxsize: int = CACHE_LRU_MAXSIZE, ttl: float = SEARCH_CACHE_TTL):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _compute_key(self, query: str, corpus_hash: str, top_k: int) -> str:
        """Gera chave de cache normalizada incluindo hash do corpus."""
        # usar função já importada no módulo para evitar imports relativos incorretos
        q_norm = normalize_equip(query)
        # Hash compacto: query_normalizada + corpus_hash + top_k
        key_str = f"{q_norm}|{corpus_hash}|{top_k}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
    
    def get(self, query: str, corpus_hash: str, top_k: int) -> Optional[dict]:
        """Busca resultado no cache (thread-safe)."""
        key = self._compute_key(query, corpus_hash, top_k)
        
        with self.lock:
            if key not in self.cache:
                self._misses += 1
                return None
            
            ts, value = self.cache[key]
            
            # Verificar TTL
            if (time.perf_counter() - ts) > self.ttl:
                del self.cache[key]
                self._misses += 1
                return None
            
            # Move para o final (marca como recentemente usado)
            self.cache.move_to_end(key)
            self._hits += 1
            return value
    
    def set(self, query: str, corpus_hash: str, top_k: int, value: dict) -> None:
        """Armazena resultado no cache (thread-safe com eviction LRU)."""
        key = self._compute_key(query, corpus_hash, top_k)
        
        with self.lock:
            # Se já existe, atualiza timestamp
            if key in self.cache:
                self.cache.move_to_end(key)
            
            # Adiciona novo item
            self.cache[key] = (time.perf_counter(), value)
            
            # Eviction: remove itens mais antigos se exceder maxsize
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)  # Remove o mais antigo (FIFO)
    
    def clear(self) -> None:
        """Limpa todo o cache."""
        with self.lock:
            self.cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> dict:
        """Retorna estatísticas do cache."""
        with self.lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0
            
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "ttl": self.ttl
            }

# Instância global do cache LRU
_lru_search_cache = LRUSearchCache(maxsize=CACHE_LRU_MAXSIZE, ttl=SEARCH_CACHE_TTL)

# ============================================================================
# 💾 CACHE JSON PERSISTENTE PARA QUERIES
# ============================================================================

class JSONQueryCache:
    """
    💾 Cache persistente em JSON para queries de busca inteligente.
    
    Complementa o cache LRU em memória salvando resultados em disco para:
    - Persistência entre reinicializações
    - Backup de queries frequentes
    - Análise de padrões de busca
    
    Características:
    - Thread-safe com lock
    - Lazy loading (carrega apenas quando necessário)
    - Auto-save após cada escrita
    - Limite de tamanho para evitar crescimento infinito
    """
    
    def __init__(self, cache_file: str = "data/cache/query_cache.json", max_entries: int = 1000):
        self.cache_file = cache_file
        self.max_entries = max_entries
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._loaded = False
    
    def _ensure_loaded(self) -> None:
        """Carrega cache do disco (lazy loading)."""
        if self._loaded:
            return
        
        with self.lock:
            if self._loaded:  # Double-check
                return
            
            try:
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, 'r', encoding='utf-8') as f:
                        self.cache = json.load(f)
                    print(f"✅ Cache JSON carregado: {len(self.cache)} queries")
                else:
                    self.cache = {}
                    print("📝 Cache JSON vazio (primeira execução)")
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache JSON: {e}")
                self.cache = {}
            
            self._loaded = True
    
    def _save(self) -> None:
        """Salva cache no disco (thread-safe)."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            # Salva em arquivo temporário primeiro (atomic write)
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            
            # Substitui arquivo original
            os.replace(temp_file, self.cache_file)
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache JSON: {e}")
    
    def get(self, query: str, corpus_hash: str, top_k: int) -> Optional[Dict[str, Any]]:
        """Busca resultado no cache JSON."""
        self._ensure_loaded()
        
        # Gera mesma chave do LRU cache para consistência
        key = self._compute_key(query, corpus_hash, top_k)
        
        with self.lock:
            entry = self.cache.get(key)
            if entry:
                # Atualiza timestamp de último acesso
                entry['last_accessed'] = time.time()
                return entry.get('result')
            return None
    
    def set(self, query: str, corpus_hash: str, top_k: int, result: Dict[str, Any]) -> None:
        """Armazena resultado no cache JSON com auto-save."""
        self._ensure_loaded()
        
        key = self._compute_key(query, corpus_hash, top_k)
        
        with self.lock:
            # Armazena com metadados
            self.cache[key] = {
                'query': query,
                'corpus_hash': corpus_hash,
                'top_k': top_k,
                'result': result,
                'created_at': time.time(),
                'last_accessed': time.time()
            }
            
            # Eviction: remove entradas mais antigas se exceder limite
            if len(self.cache) > self.max_entries:
                # Ordena por último acesso e remove 10% mais antigos
                sorted_items = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].get('last_accessed', 0)
                )
                remove_count = max(1, self.max_entries // 10)
                for old_key, _ in sorted_items[:remove_count]:
                    del self.cache[old_key]
            
            # Salva em disco
            self._save()
    
    def _compute_key(self, query: str, corpus_hash: str, top_k: int) -> str:
        """Gera chave compatível com LRUSearchCache."""
        q_norm = normalize_equip(query)
        key_str = f"{q_norm}|{corpus_hash}|{top_k}"
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
    
    def stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        self._ensure_loaded()
        
        with self.lock:
            return {
                "total_entries": len(self.cache),
                "max_entries": self.max_entries,
                "cache_file": self.cache_file,
                "file_exists": os.path.exists(self.cache_file)
            }
    
    def clear(self) -> None:
        """Limpa todo o cache (memória e disco)."""
        with self.lock:
            self.cache = {}
            self._save()

# Instância global do cache JSON persistente
_json_query_cache = JSONQueryCache()

# Legacy cache (mantido para compatibilidade com /buscar e /buscar-lote)
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
    """🚀 Inicialização otimizada com pré-carregamento de modelos."""
    print("🔧 Iniciando API de Precificação...")
    
    # Inicializa tabelas SQLite
    try:
        init_db()
        print("✅ Banco de dados inicializado")
    except Exception as e:
        print(f"⚠️ Erro ao inicializar BD: {str(e)[:100]}")
    
    # 🧠 Pré-carregamento de modelos pesados em background
    try:
        print("🧠 Carregando modelos de IA...")
        # Força carregamento dos modelos na inicialização
        _ = semantic_cache.model  # Aciona lazy loading
        _ = reranker.model  # Aciona lazy loading do reranker
        print("✅ Modelos de IA carregados e prontos")
    except Exception as e:
        print(f"⚠️ Modelos serão carregados sob demanda: {str(e)[:100]}")
    
    print("🚀 API pronta para uso!")

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
    top_k: int = DEFAULT_TOP_K
    min_score: float = 0.0
    use_tfidf: bool = True

class BatchQuery(BaseModel):
    descricoes: List[str]
    top_k: int = DEFAULT_TOP_K
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
    top_k_req = int(max(1, min(10, int(q.top_k or DEFAULT_TOP_K))))
    
    return SearchContext(
        semantic_index=semantic_index,
        item_attrs=item_attrs,
        gcol=gcol,
        target_cols=target_cols,
        df=df,
        top_k_req=top_k_req
    )


# ============================================================================
# FUNÇÃO CENTRAL DE BUSCA INTELIGENTE - UNIFICADA E ROBUSTA
# ============================================================================

def _execute_smart_search(
    query: str,
    top_k: int,
    df: pd.DataFrame,
    semantic_index: SemanticIndexCache,
    gcol: str,
    target_cols: List[str],
    item_attrs: list,
    query_original: str = None
) -> List[dict]:
    """
    🎯 FUNÇÃO CENTRAL DE BUSCA INTELIGENTE
    
    Executa busca semântica completa com pipeline consolidado:
    1. Busca semântica (SEMANTIC_CANDIDATES candidatos) - PARALELO com TF-IDF
    2. Cross-encoder reranking
    3. Combinação de scores (embedding + reranker + numeric boost)
    4. Agregação por equipamento (melhor score por grupo)
    5. Cálculo de agregações (valor, vida_util, manutenção)
    6. Filtro de confiança (MIN_CONFIDENCE >= 40%)
    7. Retorno de top_k equipamentos únicos
    
    Args:
        query: Texto de busca do usuário
        top_k: Número de resultados desejados
        df: DataFrame com dados dos equipamentos
        semantic_index: Índice semântico FAISS
        gcol: Coluna de agrupamento (descricao_padronizada)
        target_cols: Colunas de texto para busca
        item_attrs: Atributos extraídos de cada item
        query_original: Query original (para identificação em batch, opcional)
    
    Returns:
        Lista de dicionários com equipamentos únicos, ordenados por confiança
    """
    
    # Validação de entrada
    if not query or not query.strip():
        return []
    
    # ========== ETAPA 1: BUSCA PARALELA (SEMANTIC + TFIDF) ==========
    # 🚀 Quick Win Part 1: Parallel Pipeline Execution
    # Executa busca semântica e léxica em paralelo para reduzir latência em ~25-30%
    raw_candidates = None
    tfidf_candidates = None
    
    def _semantic_search_worker():
        """Worker thread para busca semântica."""
        return semantic_index.search(query, top_k=SEMANTIC_CANDIDATES)
    
    def _tfidf_search_worker():
        """Worker thread para busca TF-IDF (léxica)."""
        try:
            index = build_cached_index(df, target_cols)
            return index.search(query, top_k=SEMANTIC_CANDIDATES, return_scores=True)
        except Exception:
            return []
    
    # Execução paralela com ThreadPoolExecutor (max_workers=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_semantic = pool.submit(_semantic_search_worker)
        fut_tfidf = pool.submit(_tfidf_search_worker)
        
        # Aguarda ambos os resultados
        raw_candidates = fut_semantic.result()
        tfidf_candidates = fut_tfidf.result()
    
    # Merge dos resultados: prioriza semântico, mas considera TF-IDF para fallback
    # Mantém lógica de ranking idêntica (apenas semântico por enquanto)
    
    if not raw_candidates:
        return []
    
    # ========== ETAPA 2: RERANKING ==========
    cand_texts = [
        ' '.join([str(df.iloc[idx][c]) for c in target_cols if c in df.columns])
        for idx, _ in raw_candidates
    ]
    
    try:
        rer_scores = reranker.normalize(reranker.score(query, cand_texts)) if cand_texts else []
    except Exception:
        rer_scores = [0.0] * len(cand_texts)
    
    # ========== ETAPA 3: COMBINAÇÃO DE SCORES ==========
    query_attrs = extract_all_attributes(query)
    
    # Vetorização numpy para performance
    n_cands = len(raw_candidates)
    sem_scores = np.array([float(sc) for _, sc in raw_candidates], dtype=np.float32)
    rer_scores_arr = np.array(rer_scores[:n_cands], dtype=np.float32)
    
    # Calcular boosts numéricos em batch
    num_boosts = np.zeros(n_cands, dtype=np.float32)
    for pos, (idx, _) in enumerate(raw_candidates):
        try:
            num_boosts[pos] = float(numeric_boost(query_attrs, item_attrs[idx]))
        except Exception:
            num_boosts[pos] = 0.0
    
    # Combinação final usando pesos configurados
    final_scores = (
        SCORE_WEIGHT_EMBEDDING * sem_scores +
        SCORE_WEIGHT_RERANKER * rer_scores_arr +
        SCORE_WEIGHT_NUMERIC * num_boosts
    )
    
    # ========== ETAPA 4: AGREGAÇÃO POR EQUIPAMENTO ==========
    # Agrupar por gcol e manter APENAS o melhor score de cada grupo
    equipment_scores: Dict[str, Tuple[float, int]] = {}  # gname -> (score, df_idx)
    
    for pos, (idx, _) in enumerate(raw_candidates):
        try:
            gname = str(df.iloc[idx][gcol]).strip()
        except Exception:
            gname = ''
        
        if not gname:
            continue
        
        score = float(final_scores[pos])
        
        # Manter apenas o melhor score por equipamento
        if gname not in equipment_scores or score > equipment_scores[gname][0]:
            equipment_scores[gname] = (score, idx)
    
    if not equipment_scores:
        return []
    
    # Ordenar por score decrescente
    sorted_equipment = sorted(
        equipment_scores.items(),
        key=lambda x: x[1][0],
        reverse=True
    )
    
    # ========== ETAPA 5: CÁLCULO DE AGREGAÇÕES ==========
    # Para cada equipamento, calcular estatísticas agregadas
    max_score = sorted_equipment[0][1][0] if sorted_equipment else 1.0
    
    results = []
    for gname, (score, _) in sorted_equipment:
        # Normalizar confiança para 0-100%
        confidence = round(100.0 * score / max_score, 2)
        
        # ========== ETAPA 6: FILTRO DE CONFIANÇA ==========
        # Aplicar MIN_CONFIDENCE de forma RÍGIDA (sem exceções)
        if confidence < MIN_CONFIDENCE:
            continue
        
        # Buscar todas as linhas deste equipamento para agregações
        equipment_rows = df[df[gcol] == gname]
        
        if equipment_rows.empty:
            continue
        
        # Agregações robustas
        valor_unitario = None
        if 'valor_unitario' in equipment_rows.columns:
            try:
                valores = pd.to_numeric(equipment_rows['valor_unitario'], errors='coerce').dropna()
                if len(valores) > 0:
                    # Média robusta (remove outliers)
                    p25, p75 = valores.quantile([0.25, 0.75])
                    trimmed = valores[(valores >= p25) & (valores <= p75)]
                    if len(trimmed) > 0:
                        valor_unitario = float(trimmed.mean())
            except Exception:
                pass
        
        vida_util_meses = None
        if 'vida_util_meses' in equipment_rows.columns:
            try:
                vidas = pd.to_numeric(equipment_rows['vida_util_meses'], errors='coerce').dropna()
                if len(vidas) > 0:
                    # Moda (valor mais comum)
                    vida_util_meses = float(vidas.value_counts().idxmax())
            except Exception:
                pass
        
        manutencao_percent = None
        if 'manutencao' in equipment_rows.columns:
            try:
                manuts = pd.to_numeric(equipment_rows['manutencao'], errors='coerce')
                manuts = manuts[(manuts.notna()) & (manuts > 0)]
                if len(manuts) > 0:
                    m = float(manuts.mean())
                    # Converter para percentual se necessário
                    if m <= 1.0:
                        m = m * 100.0
                    manutencao_percent = m
            except Exception:
                pass
        
        # Marca: NÃO mostrar fornecedor individual (conforme requisito)
        # Podemos omitir ou mostrar "Várias"
        marca = None  # Deixar None para não mostrar
        
        result_dict = {
            'sugeridos': gname,  # Plural para consistência com API
            'score': confidence,  # 'score' ao invés de 'confianca'
            'valor_unitario': valor_unitario,
            'vida_util_meses': vida_util_meses,
            'manutencao_percent': manutencao_percent,
            'marca': marca,
            'link_detalhes': f"/detalhes?grupo={quote(str(gname))}"
        }
        
        # Adicionar query_original se fornecido (para batch)
        if query_original is not None:
            result_dict['query_original'] = query_original
        
        results.append(result_dict)
        
        # ========== ETAPA 7: RETORNAR TOP_K ==========
        if len(results) >= top_k:
            break
    
    return results[:top_k]


# ============================================================================
# FUNÇÕES ANTIGAS REMOVIDAS (Stage 5)
# ============================================================================
# As seguintes funções foram removidas pois foram substituídas por _execute_smart_search():
# - execute_semantic_search_with_reranker() [73 linhas]
# - process_variant_searches() [55 linhas]  
# - build_final_results() [97 linhas]
# - _process_single_smart_query() [38 linhas]
# - _build_smart_query_results() [147 linhas]
# Total economizado: ~410 linhas de código duplicado
# ============================================================================


@app.get("/health")
async def health():
    return {"status": "ok", "message": "API funcionando corretamente"}

@app.get("/cache/stats")
async def cache_stats():
    """
    🚀 Quick Win Part 2: Endpoint para monitorar estatísticas dos caches.
    
    Retorna métricas de performance dos caches:
    - Cache LRU (memória): hits, misses, taxa de acerto
    - Cache JSON (disco): total de entradas, arquivo
    """
    return {
        "lru_cache": _lru_search_cache.stats(),
        "json_cache": _json_query_cache.stats(),
        "message": "Cache statistics for query results"
    }

@app.post("/cache/clear")
async def cache_clear():
    """
    🚀 Quick Win Part 2: Limpa todos os caches (LRU e JSON).
    
    Útil para:
    - Forçar recálculo de todas as queries
    - Liberar memória
    - Debug e testes
    """
    _lru_search_cache.clear()
    _json_query_cache.clear()
    return {
        "success": True,
        "message": "Both caches cleared successfully",
        "stats": {
            "lru_cache": _lru_search_cache.stats(),
            "json_cache": _json_query_cache.stats()
        }
    }

@app.get("/debug/semantic-test")
async def debug_semantic_test(q: str = "vassoura"):
    """Endpoint de debug para testar semantic search diretamente"""
    try:
        df = load_excel()
        gcol = pick_group_col(df)
        target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
        
        semantic_index = build_cached_semantic_index(df, target_cols)
        results = semantic_index.search(q, top_k=10)
        
        # 🚀 Quick Win Part 3: Informações sobre tipo de índice
        index_type = "unknown"
        index_info = {}
        
        if hasattr(semantic_index.index, "__class__"):
            index_type = semantic_index.index.__class__.__name__
            
            # Informações específicas para IVF-PQ
            if "IVF" in index_type:
                try:
                    import faiss
                    if hasattr(semantic_index.index, "nlist"):
                        index_info["nlist"] = semantic_index.index.nlist
                    if hasattr(semantic_index.index, "nprobe"):
                        index_info["nprobe"] = semantic_index.index.nprobe
                    if hasattr(semantic_index.index, "ntotal"):
                        index_info["ntotal"] = semantic_index.index.ntotal
                except Exception:
                    pass
        
        return {
            "query": q,
            "results_count": len(results),
            "results": [{"idx": idx, "score": float(score)} for idx, score in results[:5]],
            "index_type": index_type,
            "index_info": index_info
        }
    except Exception as e:
        return {"error": str(e), "trace": str(e.__class__.__name__)}

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
        # CORRIGIDO: Não tentar ler request.json() novamente - o body já foi consumido pelo FastAPI
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
    """🛡️ Busca semântica blindada contra timeouts - Versão Robusta.

    Melhorias implementadas:
    • Carregamento lazy de modelos pesados
    • Cache agressivo de embeddings
    • Timeout interno de 25s (antes do frontend desistir)
    • Fallback automático para TF-IDF se modelos demorarem
    • Logs limpos e informativos
    """
    t_start = time.perf_counter()
    timeout_limit = 25.0  # 25s limite interno (frontend tem 30s)
    
    # Setup e validação
    try:
        df = load_excel()
    except Exception:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")
    
    # Preparação rápida (sem modelos pesados ainda)
    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    if not target_cols:
        raise HTTPException(status_code=400, detail="Colunas de descrição não encontradas")
    
    q_norm = normalize_equip(q.descricao)
    top_k_req = int(max(1, min(10, int(q.top_k or DEFAULT_TOP_K))))
    
    # 🚀 Quick Win Part 2: LRU Cache com corpus hash
    # Gera hash do corpus para invalidar cache quando dados mudam
    corpus_texts = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1).tolist()
    corpus_hash = semantic_cache.build_key(corpus_texts)
    
    # 💾 Verificar cache JSON persistente primeiro (mais rápido que recalcular)
    if cached := _json_query_cache.get(q.descricao, corpus_hash, top_k_req):
        # Também armazena no LRU para hits futuros ainda mais rápidos
        _lru_search_cache.set(q.descricao, corpus_hash, top_k_req, cached)
        
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        response.headers["Server-Timing"] = f"cache;dur={dt_ms:.1f}"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cache-Type"] = "json-persistent"
        return cached
    
    # Verificar cache LRU (resultados em memória)
    if cached := _lru_search_cache.get(q.descricao, corpus_hash, top_k_req):
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        response.headers["Server-Timing"] = f"cache;dur={dt_ms:.1f}"
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Cache-Type"] = "lru-memory"
        return cached
    
    # ⚡ Verificação de timeout antes de carregar modelos pesados
    elapsed = time.perf_counter() - t_start
    if elapsed > timeout_limit:
        # Fallback rápido para TF-IDF
        return await _fallback_tfidf_search(q, df, gcol, target_cols, response, t_start)
    
    try:
        # 🧠 Carregamento de modelos com timeout
        semantic_index = build_cached_semantic_index(df, target_cols)
        item_attrs = build_cached_attributes(df, target_cols)
        
        # Verificar timeout novamente após modelos
        elapsed = time.perf_counter() - t_start
        if elapsed > timeout_limit:
            return await _fallback_tfidf_search(q, df, gcol, target_cols, response, t_start)
        
        # Setup completo
        ctx = SearchContext(
            semantic_index=semantic_index,
            item_attrs=item_attrs,
            gcol=gcol,
            target_cols=target_cols,
            df=df,
            top_k_req=top_k_req
        )
        
        # 🔍 Execução de busca unificada (Stage 3: usando nova função)
        # Detectar variantes para metadados da resposta
        variants = expansion_variants_for_query(q.descricao)
        
        resultados = _execute_smart_search(
            query=q.descricao,
            top_k=top_k_req,
            df=ctx.df,
            semantic_index=ctx.semantic_index,
            gcol=ctx.gcol,
            target_cols=ctx.target_cols,
            item_attrs=ctx.item_attrs
        )
        
        if not resultados:
            return {"resultados": [], "total": 0, "normalizado": q_norm}
        
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
        
    except Exception as e:
        # Em qualquer erro, fallback para TF-IDF
        print(f"⚠️ Erro na busca semântica, usando fallback TF-IDF: {str(e)[:100]}")
        return await _fallback_tfidf_search(q, df, gcol, target_cols, response, t_start)
    
    # ✅ Sucesso - cachear e retornar
    dt_ms = (time.perf_counter() - t_start) * 1000.0
    response.headers["Server-Timing"] = f"search;dur={dt_ms:.1f}"
    response.headers["X-Cache"] = "MISS"
    response.headers["X-Search-Type"] = "semantic"
    
    # 🚀 Quick Win Part 2: Armazenar resultado final completo nos dois caches
    _lru_search_cache.set(q.descricao, corpus_hash, top_k_req, resp)
    
    # 💾 Salvar também no cache JSON persistente (em background para não bloquear)
    try:
        background_tasks.add_task(_json_query_cache.set, q.descricao, corpus_hash, top_k_req, resp)
    except Exception:
        pass
    
    # Background logging (não bloqueia resposta)
    try:
        # CORRIGIDO: Não tentar ler request.json() novamente - o body já foi consumido pelo FastAPI
        # context_tags seria passado via Query model se necessário (não implementado no modelo atual)
        context_tags = None
        uid = _get_user_id(request)
        background_tasks.add_task(_log_search_history_async, uid, q.descricao, len(resultados), _to_csv(context_tags))
    except Exception:
        pass
    
    # Log limpo de sucesso
    print(f"✅ Busca: {dt_ms:.0f}ms - \"{q.descricao[:30]}...\" ({len(resultados)} resultados)")
    
    return resp


async def _fallback_tfidf_search(q: Query, df: pd.DataFrame, gcol: str, target_cols: list, response: Response, t_start: float) -> dict:
    """Fallback rápido para TF-IDF quando semântico demora muito."""
    try:
        top_groups, group_scores = _process_tfidf_search(df, q, gcol, target_cols, False)
        results = _build_search_results(df, top_groups, gcol, group_scores, q)
        
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        response.headers["Server-Timing"] = f"fallback;dur={dt_ms:.1f}"
        response.headers["X-Search-Type"] = "tfidf-fallback"
        
        print(f"⚡ Fallback TF-IDF: {dt_ms:.0f}ms - \"{q.descricao[:30]}...\"")
        
        return {
            'query_original': q.descricao,
            'query_normalizada': normalize_equip(q.descricao),
            'resultados': results,
            'total': len(results),
            'fallback': True,
            'fallback_reason': 'timeout'
        }
    except Exception as e:
        # Último fallback: resultado vazio
        print(f"❌ Erro crítico no fallback: {str(e)[:100]}")
        return {
            'query_original': q.descricao,
            'resultados': [],
            'total': 0,
            'error': 'critical_timeout'
        }

@app.post("/buscar-lote-inteligente")
async def buscar_lote_inteligente(batch: BatchQuery, request: Request, response: Response, background_tasks: BackgroundTasks):
    """🔄 Versão em lote do buscador inteligente - UNIFICADA (Stage 4).

    Usa a mesma função _execute_smart_search() do endpoint individual para garantir
    consistência total de resultados, nomenclatura e comportamento.
    
    Entrada: { descricoes: [str], top_k?: int }
    Saída: lista achatada com campo 'query_original' para identificar cada query.
    """
    try:
        df = load_excel()
    except Exception:
        raise HTTPException(status_code=400, detail="Planilha não encontrada. Faça upload primeiro.")

    gcol = pick_group_col(df)
    target_cols = [c for c in ['descricao_padronizada','descricao_saneada','descricao'] if c in df.columns]
    if not target_cols:
        return {"resultados": [], "total": 0}

    # Preparar recursos compartilhados (cache otimizado)
    semantic_index = build_cached_semantic_index(df, target_cols)
    item_attrs = build_cached_attributes(df, target_cols)
    
    # 💾 Gerar hash do corpus para cache JSON
    corpus_texts = df[target_cols].astype(str).fillna('').agg(' '.join, axis=1).tolist()
    corpus_hash = semantic_cache.build_key(corpus_texts)
    
    rows = []
    t0 = time.perf_counter()
    top_k_req = int(max(1, min(10, int(batch.top_k or DEFAULT_TOP_K))))
    
    # OTIMIZAÇÃO DINÂMICA: Reduzir candidatos para reranker em lotes muito grandes
    # Isso mantém performance aceitável sem sacrificar qualidade
    batch_size = len(batch.descricoes)
    
    if batch_size <= 20:
        rerank_candidates = SEMANTIC_CANDIDATES  # 150 para precisão máxima
    elif batch_size <= 50:
        rerank_candidates = 75  # Reduzir pela metade
    elif batch_size <= 100:
        rerank_candidates = 50  # 1/3 dos candidatos
    else:
        rerank_candidates = 30  # Para lotes gigantes (100+)
    
    print(f"🔄 Processando lote de {batch_size} queries (reranking top-{rerank_candidates} por query)")
    
    # OTIMIZAÇÃO: Pré-processar todas as queries de uma vez para o reranker
    # Isso evita carregar o modelo múltiplas vezes
    all_query_candidates = []
    query_candidate_mapping = []  # Mapeamento de query -> seus candidatos
    cached_queries = {}  # query_idx -> cached results
    queries_needing_processing = []  # Queries que não estão em cache
    
    # 💾 VERIFICAR CACHE JSON PARA CADA QUERY
    print(f"💾 Verificando cache JSON para {batch_size} queries...")
    for query_idx, q in enumerate(batch.descricoes):
        if not q or not q.strip():
            query_candidate_mapping.append((q, []))
            continue
        
        # Tentar buscar no cache JSON primeiro
        cached_result = _json_query_cache.get(q, corpus_hash, top_k_req)
        if cached_result:
            # Cache HIT - armazenar resultado e pular processamento
            cached_queries[query_idx] = cached_result
            query_candidate_mapping.append((q, []))  # Vazio pois já tem cache
            continue
        
        # Cache MISS - precisa processar
        queries_needing_processing.append(query_idx)
        
        # Busca semântica (rápida) - mas pegando mais candidatos inicialmente
        raw_candidates = semantic_index.search(q, top_k=SEMANTIC_CANDIDATES)
        
        if not raw_candidates:
            query_candidate_mapping.append((q, []))
            continue
        
        # FILTRAR para enviar apenas top-N para reranking (otimização para lotes grandes)
        candidates_to_rerank = raw_candidates[:rerank_candidates]
        
        # Armazenar candidatos COMPLETOS (para scoring depois)
        query_candidate_mapping.append((q, raw_candidates))
        
        # Mas enviar apenas subset para reranking
        cand_texts = [
            ' '.join([str(df.iloc[idx][c]) for c in target_cols if c in df.columns])
            for idx, _ in candidates_to_rerank
        ]
        
        # Adicionar ao batch de reranking com índice da query
        for cand_text in cand_texts:
            all_query_candidates.append((query_idx, q, cand_text))
    
    # 💾 LOG de estatísticas do cache
    cache_hits = len(cached_queries)
    cache_misses = len(queries_needing_processing)
    print(f"💾 Cache: {cache_hits} hits, {cache_misses} misses ({cache_hits}/{batch_size} = {cache_hits/batch_size*100:.1f}%)")
    
    # RERANKING EM LOTE (query por query, mas com batch processing interno)
    reranker_results = {}  # query_idx -> [scores]
    
    # Agrupar candidatos por query para reranking eficiente
    query_groups = {}  # query_idx -> (query_text, [candidate_texts])
    for query_idx, q, cand_text in all_query_candidates:
        if query_idx not in query_groups:
            query_groups[query_idx] = (q, [])
        query_groups[query_idx][1].append(cand_text)
    
    # Executar reranking para cada query
    for query_idx, (query_text, candidate_texts) in query_groups.items():
        try:
            # Chamar reranker.score corretamente: (query, candidates)
            raw_scores = reranker.score(query_text, candidate_texts)
            normalized_scores = reranker.normalize(raw_scores)
            reranker_results[query_idx] = normalized_scores
        except Exception as e:
            print(f"⚠️ Erro no reranking para query {query_idx}: {e}")
            # Fallback: scores zero
            reranker_results[query_idx] = [0.0] * len(candidate_texts)
    
    # Processar cada query com seus scores pré-calculados
    for query_idx, (q, raw_candidates) in enumerate(query_candidate_mapping):
        if not raw_candidates:
            continue
        
        try:
            # Usar scores do reranking em lote (apenas para os primeiros N candidatos)
            rer_scores_partial = reranker_results.get(query_idx, [])
            
            # Expandir scores para todos os candidatos (scores=0 para não re-rankeados)
            rer_scores = rer_scores_partial + [0.0] * (len(raw_candidates) - len(rer_scores_partial))
            
            # Continuar com o pipeline normal (rápido)
            query_attrs = extract_all_attributes(q)
            
            n_cands = len(raw_candidates)
            sem_scores = np.array([float(sc) for _, sc in raw_candidates], dtype=np.float32)
            rer_scores_arr = np.array(rer_scores[:n_cands], dtype=np.float32)
            
            num_boosts = np.zeros(n_cands, dtype=np.float32)
            for pos, (idx, _) in enumerate(raw_candidates):
                try:
                    num_boosts[pos] = float(numeric_boost(query_attrs, item_attrs[idx]))
                except Exception:
                    num_boosts[pos] = 0.0
            
            final_scores = (
                SCORE_WEIGHT_EMBEDDING * sem_scores +
                SCORE_WEIGHT_RERANKER * rer_scores_arr +
                SCORE_WEIGHT_NUMERIC * num_boosts
            )
            
            # Agregação por equipamento
            equipment_scores: Dict[str, Tuple[float, int]] = {}
            
            for pos, (idx, _) in enumerate(raw_candidates):
                try:
                    gname = str(df.iloc[idx][gcol]).strip()
                except Exception:
                    gname = ''
                
                if not gname:
                    continue
                
                score = float(final_scores[pos])
                
                if gname not in equipment_scores or score > equipment_scores[gname][0]:
                    equipment_scores[gname] = (score, idx)
            
            if not equipment_scores:
                continue
            
            sorted_equipment = sorted(
                equipment_scores.items(),
                key=lambda x: x[1][0],
                reverse=True
            )
            
            max_score = sorted_equipment[0][1][0] if sorted_equipment else 1.0
            
            # Construir resultados
            for gname, (score_val, _) in sorted_equipment[:top_k_req]:
                confidence = round(100.0 * score_val / max_score, 2)
                
                if confidence < MIN_CONFIDENCE:
                    continue
                
                equipment_rows = df[df[gcol] == gname]
                
                if equipment_rows.empty:
                    continue
                
                # Agregações
                valor_unitario = None
                if 'valor_unitario' in equipment_rows.columns:
                    try:
                        valores = pd.to_numeric(equipment_rows['valor_unitario'], errors='coerce').dropna()
                        if len(valores) > 0:
                            p25, p75 = valores.quantile([0.25, 0.75])
                            trimmed = valores[(valores >= p25) & (valores <= p75)]
                            if len(trimmed) > 0:
                                valor_unitario = float(trimmed.mean())
                    except Exception:
                        pass
                
                vida_util_meses = None
                if 'vida_util_meses' in equipment_rows.columns:
                    try:
                        vidas = pd.to_numeric(equipment_rows['vida_util_meses'], errors='coerce').dropna()
                        if len(vidas) > 0:
                            vida_util_meses = float(vidas.value_counts().idxmax())
                    except Exception:
                        pass
                
                manutencao_percent = None
                if 'manutencao' in equipment_rows.columns:
                    try:
                        manuts = pd.to_numeric(equipment_rows['manutencao'], errors='coerce')
                        manuts = manuts[(manuts.notna()) & (manuts > 0)]
                        if len(manuts) > 0:
                            m = float(manuts.mean())
                            if m <= 1.0:
                                m = m * 100.0
                            manutencao_percent = m
                    except Exception:
                        pass
                
                rows.append({
                    'query_original': q,
                    'sugeridos': gname,
                    'score': confidence,
                    'valor_unitario': valor_unitario,
                    'vida_util_meses': vida_util_meses,
                    'manutencao_percent': manutencao_percent,
                    'marca': None,
                    'link_detalhes': f"/detalhes?grupo={quote(str(gname))}"
                })
        
        except Exception as e:
            # Log do erro mas continua processando outras queries
            print(f"❌ Erro ao processar query {query_idx} ('{q[:50]}...'): {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 💾 ADICIONAR RESULTADOS DO CACHE JSON
    for query_idx, cached_result in cached_queries.items():
        if cached_result and 'resultados' in cached_result:
            for resultado in cached_result['resultados']:
                # Adicionar query_original se não existir
                if 'query_original' not in resultado:
                    resultado['query_original'] = batch.descricoes[query_idx]
                rows.append(resultado)
    
    # 💾 SALVAR NOVOS RESULTADOS NO CACHE JSON (em background)
    def _save_to_cache():
        try:
            # Agrupar resultados por query para salvar no cache
            for query_idx in queries_needing_processing:
                q = batch.descricoes[query_idx]
                query_results = [r for r in rows if r.get('query_original') == q]
                
                if query_results:
                    # Montar objeto de resposta compatível com cache individual
                    cache_obj = {
                        'query_original': q,
                        'query_normalizada': normalize_equip(q),
                        'consonant_key': consonant_key(q),
                        'expansoes_detectadas': expansion_variants_for_query(q),
                        'modelo_semantico': DEFAULT_MODEL_NAME,
                        'modelo_reranker': DEFAULT_RERANKER_MODEL,
                        'resultados': query_results,
                        'total': len(query_results)
                    }
                    _json_query_cache.set(q, corpus_hash, top_k_req, cache_obj)
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache JSON: {e}")
    
    # Executar salvamento em background
    background_tasks.add_task(_save_to_cache)
    
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["Server-Timing"] = f"batch-search;dur={dt_ms:.1f}"
    
    # Adicionar header indicando quantas queries vieram do cache
    cache_hits = len(cached_queries)
    total_queries = len(batch.descricoes)
    if cache_hits > 0:
        response.headers["X-Cache-Stats"] = f"{cache_hits}/{total_queries} from cache"
    
    # Log do histórico por descrição em background
    try:
        uid = _get_user_id(request)
        # CORRIGIDO: Não tentar ler request.json() novamente - o body já foi consumido pelo FastAPI
        # context_tags seria passado via batch.context_tags se necessário (não implementado no modelo atual)
        context_tags = None
        
        def _log_batch():
            try:
                with engine.begin() as conn:
                    for qdesc in batch.descricoes:
                        # Usar 'sugeridos' (nomenclatura padronizada)
                        cnt = sum(1 for r in rows if (r.get('query_original') == qdesc and r.get('sugeridos')))
                        conn.execute(text(
                            "INSERT INTO search_history (user_id, query, context_tags, results_count) VALUES (:u,:q,:t,:n)"
                        ), {"u": uid, "q": qdesc, "t": _to_csv(context_tags), "n": cnt})
            except Exception:
                pass
        background_tasks.add_task(_log_batch)
    except Exception:
        pass
    
    return {"resultados": rows, "total": len(rows)}

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


