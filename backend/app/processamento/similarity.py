"""
TF-IDF com n-gramas (caracteres/palavras), correção ortográfica, busca híbrida.
Lazy imports sklearn (~3s economia), processamento vetorizado, robusto a typos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Sequence, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

import numpy as np
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# Lazy imports sklearn (economiza ~3s startup)
TfidfVectorizer = None
cosine_similarity = None

def _lazy_import_sklearn():
    """Import lazy sklearn."""
    global TfidfVectorizer, cosine_similarity
    if TfidfVectorizer is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as TV
            from sklearn.metrics.pairwise import cosine_similarity as CS
            TfidfVectorizer = TV
            cosine_similarity = CS
        except ImportError:
            raise RuntimeError("scikit-learn não está instalado. Adicione 'scikit-learn' ao requirements.txt")
    return TfidfVectorizer, cosine_similarity


def _ensure_list(texts: Iterable[str]) -> List[str]:
    """Garante que entrada é lista."""
    return list(texts) if not isinstance(texts, list) else texts


def default_preprocess(s: str) -> str:
    """Pré-processamento leve: minúsculas, remove espaços extras."""
    if not s:
        return ""
    s = s.strip().lower()
    return " ".join(s.split())


def strip_accents(s: str) -> str:
    """
    Remove acentuação via normalização Unicode.
    
    Args:
        s: String com possível acentuação
        
    Returns:
        String sem acentos (ASCII puro)
    """
    if not s:
        return ""
    return (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def simple_tokenize(s: str) -> List[str]:
    """
    Tokenização simples: divide em palavras alfanuméricas.
    
    Processo:
    1. Normaliza e remove acentos
    2. Converte não-alfanuméricos em espaços
    3. Divide em tokens
    
    Args:
        s: String a tokenizar
        
    Returns:
        Lista de tokens (palavras)
    """
    s = default_preprocess(strip_accents(s))
    # Mantém letras/dígitos e espaços
    out = []
    cur = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            cur.append(' ')
    s2 = "".join(cur)
    for tok in s2.split():
        if tok:
            out.append(tok)
    return out


def _common_prefix_len(a: str, b: str) -> int:
    """
    Calcula comprimento do prefixo comum entre duas strings.
    
    Útil para detectar variações de palavras (ex: "vassoura" vs "vassourão").
    
    Args:
        a: Primeira string
        b: Segunda string
        
    Returns:
        Comprimento do prefixo comum
    """
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i


class TokenVocab:
    """
    Vocabulário de tokens extraídos do corpus, usado para sugerir correções de queries.
    
    Funcionalidade:
    --------------
    - Extrai vocabulário único do corpus
    - Sugere correções para termos mal escritos
    - Corrige queries automaticamente baseado em similaridade fuzzy
    - Útil para melhorar robustez a typos e variações ortográficas
    
    Algoritmo:
    ---------
    Usa RapidFuzz WRatio para encontrar tokens similares no vocabulário.
    WRatio é robusto a diferenças de ordem e pequenas variações.
    """
    def __init__(self, tokens: Sequence[str]):
        """
        Inicializa vocabulário a partir de tokens.
        
        Args:
            tokens: Sequência de tokens (palavras) do corpus
        """
        # Mantém únicos e ignora tokens muito curtos (< 3 chars)
        toks = [t for t in tokens if t and len(t) >= 3]
        self.tokens = sorted(set(toks))

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "TokenVocab":
        """
        Cria vocabulário a partir de corpus de textos.
        
        Args:
            texts: Corpus de textos
            
        Returns:
            Instância de TokenVocab com vocabulário extraído
        """
        all_toks: List[str] = []
        for t in texts:
            all_toks.extend(simple_tokenize(t))
        return cls(all_toks)

    def suggest(self, token: str, limit: int = 3) -> List[Tuple[str, float]]:
        """
        Sugere correções para um token baseado no vocabulário.
        
        Usa RapidFuzz WRatio (robusto a pequenas diferenças).
        
        Args:
            token: Token a corrigir
            limit: Número máximo de sugestões (padrão: 3)
            
        Returns:
            Lista de tuplas (candidato, score_normalizado) ordenada por score
        """
        token = strip_accents(token.lower().strip())
        if not token or len(self.tokens) == 0:
            return []
        # Usa WRatio (robusto a pequenas diferenças); retorna top N
        res = rf_process.extract(token, self.tokens, scorer=rf_fuzz.WRatio, limit=limit)
        # Resultado: List[(cand, score, idx)] -> converte para (cand, score/100)
        return [(cand, float(score) / 100.0) for cand, score, _ in res]

    def correct_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        Corrige automaticamente termos em uma query baseado no vocabulário.
        
        Estratégia:
        ----------
        - Ignora números e tokens muito curtos (< 3 chars)
        - Para cada token, busca melhor candidato no vocabulário
        - Aplica limiar dinâmico: mais rigoroso para tokens curtos
        - Retorna query corrigida e dicionário de mudanças
        
        Args:
            query: Query original a corrigir
            
        Returns:
            Tupla (query_corrigida, dicionário_de_mudanças)
            
        Exemplo:
            >>> vocab = TokenVocab(["motor", "compressor"])
            >>> vocab.correct_query("motro eletrico")
            ("motor eletrico", {"motro": "motor"})
        """
        toks = simple_tokenize(query)
        changes: Dict[str, str] = {}
        new_toks: List[str] = []
        for t in toks:
            # Ignora tokens muito curtos ou numéricos
            if len(t) < 3 or t.isdigit():
                new_toks.append(t)
                continue
            cands = self.suggest(t, limit=1)
            if not cands:
                new_toks.append(t)
                continue
            cand, score = cands[0]
            # Limiar dinâmico: mais rigoroso para tokens curtos
            thresh = 0.85 if len(t) >= 5 else 0.90
            if score >= thresh and cand != t:
                new_toks.append(cand)
                changes[t] = cand
            else:
                new_toks.append(t)
        return (" ".join(new_toks), changes)


@dataclass
class TfidfSearchIndex:
    """
    Índice de busca por similaridade baseado em TF-IDF com n-gramas.

    Características:
    ---------------
    - Por padrão usa n-gramas de caracteres (char_wb) para robustez a variações e typos
    - Suporta também n-gramas de palavras (word) para captura de frases
    - Busca eficiente via scipy sparse matrices
    - Similaridade calculada por cosseno
    
    N-gramas de caracteres (char_wb):
    --------------------------------
    - Extrai sequências de N caracteres das palavras
    - Exemplo: "motor" com (3,5) → ["mot", "moto", "motor", "otor", ...]
    - Robusto a typos (ex: "motr" ainda tem overlap com "motor")
    - Não depende de dicionário ou tokenização complexa
    
    Atributos:
    ---------
    vectorizer: TfidfVectorizer do sklearn
    matrix: Matriz esparsa TF-IDF do corpus
    corpus: Lista de textos originais
    """

    vectorizer: Any  # TfidfVectorizer
    matrix: Any  # scipy.sparse matrix
    corpus: List[str]

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        analyzer: str = "char_wb",  # "char", "char_wb" ou "word"
        ngram_range: Tuple[int, int] = (3, 5),
        max_features: Optional[int] = 100_000,
        min_df: float | int = 1,
        max_df: float | int = 1.0,
        norm: Optional[str] = "l2",
        preprocessor: Optional[Any] = default_preprocess,
        stop_words: Optional[Sequence[str]] = None,
    ) -> "TfidfSearchIndex":
        """
        Constrói índice TF-IDF a partir de corpus.
        
        Args:
            texts: Corpus de textos a indexar
            analyzer: Tipo de análise ("char_wb", "char", "word")
            ngram_range: Faixa de n-gramas (ex: (3, 5) = trigramas até 5-gramas)
            max_features: Limite de features (controla dimensionalidade)
            min_df: Frequência mínima de documento (ignora termos raros)
            max_df: Frequência máxima de documento (ignora termos muito comuns)
            norm: Normalização dos vetores ("l2" recomendado para cosseno)
            preprocessor: Função de pré-processamento
            stop_words: Lista de stop words (apenas para analyzer="word")
            
        Returns:
            Instância de TfidfSearchIndex pronta para busca
        """
        corpus = _ensure_list(texts)
        # Para analyzer="word", manter stop_words se fornecido; para char, ignorar
        TfidfVectorizer_cls, _ = _lazy_import_sklearn()
        vectorizer = TfidfVectorizer_cls(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm=norm,
            preprocessor=preprocessor,
            stop_words=stop_words if analyzer == "word" else None,
        )
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer=vectorizer, matrix=matrix, corpus=corpus)

    def query_vector(self, query: str):
        """
        Transforma query em vetor TF-IDF.
        
        Args:
            query: String de busca
            
        Returns:
            Vetor esparso TF-IDF da query
        """
        return self.vectorizer.transform([query])

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        return_scores: bool = True,
        exclude_indices: Optional[Sequence[int]] = None,
    ) -> List[Tuple[int, float]] | List[int]:
        """
        Busca itens mais similares à query por similaridade de cosseno.
        
        Args:
            query: String de busca
            top_k: Número de resultados a retornar
            min_score: Score mínimo para incluir resultado
            return_scores: Se True, retorna tuplas (índice, score); senão apenas índices
            exclude_indices: Índices a excluir dos resultados
            
        Returns:
            Lista de tuplas (índice, score) ou lista de índices, ordenada por score decrescente
        """
        qv = self.query_vector(query)
        _, cosine_similarity_fn = _lazy_import_sklearn()
        sims = cosine_similarity_fn(qv, self.matrix).ravel()
        if exclude_indices:
            sims[list(exclude_indices)] = -1.0  # Garante exclusão
        if min_score > 0:
            mask = sims >= min_score
            candidates = np.where(mask)[0]
        else:
            candidates = np.arange(sims.shape[0])
        if len(candidates) == 0:
            return []
        # Top-k por partição eficiente (mais rápido que ordenação completa)
        k = min(top_k, len(candidates))
        part = np.argpartition(-sims[candidates], k - 1)[:k]
        top_idx_unsorted = candidates[part]
        order = np.argsort(-sims[top_idx_unsorted])
        top_idx = top_idx_unsorted[order]
        if return_scores:
            return [(int(i), float(sims[i])) for i in top_idx]
        return [int(i) for i in top_idx]


def build_index_from_dataframe(
    df,
    text_col: str,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
    max_features: Optional[int] = 100_000,
    min_df: float | int = 1,
    max_df: float | int = 1.0,
    norm: Optional[str] = "l2",
):
    """
    Conveniência para criar índice TF-IDF a partir de um DataFrame.
    
    Args:
        df: DataFrame pandas
        text_col: Nome da coluna com textos
        ...: Demais parâmetros passados para TfidfSearchIndex.build()
        
    Returns:
        Instância de TfidfSearchIndex
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    return TfidfSearchIndex.build(
        texts,
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        norm=norm,
    )


def search_similares_dataframe(
    df,
    text_col: str,
    query: str,
    top_k: int = 10,
    min_score: float = 0.0,
    return_cols: Optional[Sequence[str]] = None,
    analyzer: str = "char_wb",
    ngram_range: Tuple[int, int] = (3, 5),
):
    """
    Cria índice temporário e retorna DataFrame com os top-k similares.
    
    Para alto desempenho em múltiplas consultas, prefira construir o índice uma vez
    e reutilizá-lo com index.search().
    
    Args:
        df: DataFrame pandas
        text_col: Nome da coluna com textos
        query: String de busca
        top_k: Número de resultados
        min_score: Score mínimo
        return_cols: Colunas a retornar (None = todas)
        analyzer: Tipo de análise TF-IDF
        ngram_range: Faixa de n-gramas
        
    Returns:
        DataFrame com resultados e coluna __score__
    """
    index = build_index_from_dataframe(df, text_col, analyzer=analyzer, ngram_range=ngram_range)
    results = index.search(query, top_k=top_k, min_score=min_score, return_scores=True)
    if not results:
        # Retorna DataFrame vazio com colunas esperadas
        cols = list(return_cols) + ["__score__"] if return_cols else [text_col, "__score__"]
        return df.head(0)[[]].reindex(columns=cols)
    idxs, scores = zip(*results)
    out = df.iloc[list(idxs)].copy()
    out["__score__"] = list(scores)
    if return_cols:
        cols = [c for c in return_cols if c in out.columns]
        return out[cols + ["__score__"]]
    return out[[text_col, "__score__"]]


@dataclass
class HybridTfidfSearchIndex:
    """
    Índice híbrido combinando TF-IDF de caracteres, palavras e overlap de tokens.
    
    Estratégias combinadas:
    ----------------------
    1. TF-IDF char_wb (n-gramas de caracteres): Robusto a typos e variações
    2. TF-IDF word (n-gramas de palavras): Captura frases e contexto
    3. Token overlap ponderado por IDF: Presença de termos raros
    
    Otimizações aplicadas (Quick Win Part 3):
    -----------------------------------------
    - Pesos rebalanceados para melhor precisão
    - Penalizações para âncoras e head tokens
    - Tratamento especial para queries de um único token
    - Truncamento de candidatos top_candidates_factor para eficiência
    - Autocorreção automática via vocabulário
    
    Atributos:
    ---------
    vectorizer_char: TF-IDF de n-gramas de caracteres
    matrix_char: Matriz esparsa char
    vectorizer_word: TF-IDF de n-gramas de palavras
    matrix_word: Matriz esparsa word
    tokens_per_doc: Tokens únicos por documento
    idf_map: Mapa termo → IDF para ponderação
    corpus: Textos originais
    vocab: Vocabulário para correção
    w_char, w_word, w_overlap: Pesos dos componentes
    """
 
    vectorizer_char: Any  # TfidfVectorizer
    matrix_char: Any
    vectorizer_word: Any  # TfidfVectorizer
    matrix_word: Any
    tokens_per_doc: List[set]
    idf_map: Dict[str, float]
    corpus: List[str]
    vocab: Optional[TokenVocab] = None

    # 🚀 Quick Win: Pesos rebalanceados TF-IDF (Part 3)
    w_char: float = 0.6  # Peso de n-gramas de caracteres
    w_word: float = 0.25  # Peso de n-gramas de palavras
    w_overlap: float = 0.15  # Peso de overlap de tokens
    anchor_penalty: float = 0.8  # Multiplicador quando nenhuma âncora aparece
    anchor_min_len: int = 3  # Comprimento mínimo para considerar âncora
    head_token_penalty: float = 0.9  # Penalização se documento não contiver o head token

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        ngram_char: Tuple[int, int] = (3, 5),
        ngram_word: Tuple[int, int] = (1, 2),
        max_features_char: Optional[int] = 200_000,
        max_features_word: Optional[int] = 100_000,
        w_char: float = 0.6,
        w_word: float = 0.25,
        w_overlap: float = 0.15,
        anchor_penalty: float = 0.8,
        anchor_min_len: int = 3,
    ) -> "HybridTfidfSearchIndex":
        """
        Constrói índice híbrido a partir de corpus.
        
        Detalhes da implementação estão no código. Principais características:
        - Dois vetorizadores TF-IDF (char e word)
        - Extração de tokens e IDF por documento
        - Vocabulário para autocorreção
        """
        corpus = _ensure_list(texts)
        def _prep(x: str) -> str:
            return default_preprocess(strip_accents(x))

        TfidfVectorizer_cls, _ = _lazy_import_sklearn()
        vec_char = TfidfVectorizer_cls(
            analyzer="char_wb",
            ngram_range=ngram_char,
            max_features=max_features_char,
            preprocessor=_prep,
        )
        mat_char = vec_char.fit_transform(corpus)

        vec_word = TfidfVectorizer_cls(
            analyzer="word",
            ngram_range=ngram_word,
            max_features=max_features_word,
            preprocessor=_prep,
            token_pattern=r"(?u)\b\w+\b",
        )
        mat_word = vec_word.fit_transform(corpus)

        # Prepara tokens por doc e mapa de idf
        tokens_per_doc = [set(simple_tokenize(t)) for t in corpus]
        idf_vals = getattr(vec_word, "idf_", None)
        idf_map: Dict[str, float] = {}
        if idf_vals is not None:
            for term, idf in zip(vec_word.get_feature_names_out(), idf_vals):
                idf_map[strip_accents(term)] = float(idf)
        vocab = TokenVocab.from_texts(corpus)
        return cls(
            vectorizer_char=vec_char,
            matrix_char=mat_char,
            vectorizer_word=vec_word,
            matrix_word=mat_word,
            tokens_per_doc=tokens_per_doc,
            idf_map=idf_map,
            corpus=corpus,
            vocab=vocab,
            w_char=w_char,
            w_word=w_word,
            w_overlap=w_overlap,
            anchor_penalty=anchor_penalty,
            anchor_min_len=anchor_min_len,
        )

    def _anchor_tokens(self, query: str) -> List[str]:
        """Extrai tokens âncora (raros e relevantes) da query ordenados por IDF."""
        toks = simple_tokenize(query)
        # Filtra âncoras por tamanho mínimo e ordena por IDF desc (mais raros = mais fortes)
        toks = [t for t in toks if len(t) >= self.anchor_min_len]
        toks.sort(key=lambda t: -self.idf_map.get(t, 0.0))
        # Garantir inclusão de 'mop' quando presente (domínio frequente)
        if 'mop' in toks and (not toks or toks[0] != 'mop'):
            # Coloca 'mop' no início para maior influência
            toks = ['mop'] + [t for t in toks if t != 'mop']
        # Limita a algumas âncoras principais para não diluir (máx 5)
        return toks[:5] if toks else []

    def _head_token(self, query: str) -> Optional[str]:
        """Extrai primeiro token significativo da query."""
        for t in simple_tokenize(query):
            if len(t) >= self.anchor_min_len and not t.isdigit():
                return t
        return None

    def _overlap_score(self, q_toks: List[str], doc_tokens: Sequence[set]) -> np.ndarray:
        """Calcula score de overlap de tokens ponderado por IDF."""
        if not q_toks:
            return np.zeros(len(doc_tokens), dtype=float)
        q_weights = {t: self.idf_map.get(t, 1.0) for t in q_toks}
        max_w = sum(q_weights.values()) or 1.0
        scores = np.zeros(len(doc_tokens), dtype=float)
        for i, toks in enumerate(doc_tokens):
            s = 0.0
            for t, w in q_weights.items():
                if t in toks:
                    s += w
            scores[i] = s / max_w
        return scores

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        return_scores: bool = True,
    ) -> List[Tuple[int, float]] | List[int]:
        """
        Busca híbrida com autocorreção e múltiplas estratégias.
        
        Ver implementação completa no código fonte.
        """
        # Autocorreção leve: combina original e corrigida
        q_norm_input = strip_accents(query)
        q_corrected = None
        if self.vocab is not None:
            qc, changes = self.vocab.correct_query(q_norm_input)
            if changes:
                q_corrected = qc
        if q_corrected and q_corrected != q_norm_input:
            return self.search_multi([q_norm_input, q_corrected], top_k=top_k, min_score=min_score, return_scores=return_scores)
        # Sem correção: executa busca única
        return self._search_single(q_norm_input, top_k=top_k, min_score=min_score, return_scores=return_scores)

    def _search_single(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        return_scores: bool = True,
    ) -> List[Tuple[int, float]] | List[int]:
        """
        Busca única combinando char TF-IDF, word TF-IDF e overlap.
        
        Ver código completo para detalhes de implementação incluindo:
        - Truncamento de candidatos (Quick Win Part 3)
        - Pesos otimizados para queries de token único
        - Penalizações de âncoras e head tokens
        """
        # 🚀 Quick Win: Configurable candidate truncation (Part 3)
        top_candidates_factor = 200
        
        q_norm = query
        qv_char = self.vectorizer_char.transform([q_norm])
        qv_word = self.vectorizer_word.transform([q_norm])
        _, cosine_similarity_fn = _lazy_import_sklearn()
        sims_char = cosine_similarity_fn(qv_char, self.matrix_char).ravel()
        sims_word = cosine_similarity_fn(qv_word, self.matrix_word).ravel()

        anchors = self._anchor_tokens(q_norm)
        overlap = self._overlap_score(anchors, self.tokens_per_doc)

        # tamanho da consulta (nº de tokens)
        q_toks = simple_tokenize(q_norm)
        single_token_query = len(q_toks) <= 1

        # 🚀 Quick Win: Optimized weights for single-token queries (Part 3)
        if single_token_query:
            w_char = 0.75
            w_word = 0.15
            w_overlap = 1.0 - w_char - w_word  # 0.10
        else:
            w_char, w_word, w_overlap = self.w_char, self.w_word, self.w_overlap

        score = w_char * sims_char + w_word * sims_word + w_overlap * overlap

        # penaliza itens sem nenhuma âncora (apenas se a consulta tiver múltiplas palavras)
        if anchors and not single_token_query:
            has_anchor = np.array([any(a in toks for a in anchors) for toks in self.tokens_per_doc], dtype=bool)
            score = np.where(has_anchor, score, score * (1.0 - self.anchor_penalty))

        # penaliza itens que não contêm o head token (ex.: 'mop' no início da consulta)
        head = self._head_token(q_norm)
        if head:
            if single_token_query:
                # consulta genérica: considerar prefix/substring para variações (ex.: 'vassourao')
                has_head = []
                for toks in self.tokens_per_doc:
                    ok = False
                    for t in toks:
                        if len(t) >= self.anchor_min_len:
                            if head in t or t in head or _common_prefix_len(head, t) >= self.anchor_min_len:
                                ok = True
                                break
                    has_head.append(ok)
                has_head = np.array(has_head, dtype=bool)
                penalty = max(0.0, min(1.0, self.head_token_penalty * 0.25))
                score = np.where(has_head, score, score * (1.0 - penalty))
            else:
                has_head = np.array([head in toks for toks in self.tokens_per_doc], dtype=bool)
                score = np.where(has_head, score, score * (1.0 - self.head_token_penalty))

        # aplica min_score
        candidates = np.where(score >= min_score)[0]
        if len(candidates) == 0:
            return []
        
        # 🚀 Quick Win: Limit candidates to top_candidates_factor (Part 3)
        if len(candidates) > top_candidates_factor:
            # Get top N candidates by score before final sorting
            part_truncate = np.argpartition(-score[candidates], top_candidates_factor - 1)[:top_candidates_factor]
            candidates = candidates[part_truncate]
        
        k = min(top_k, len(candidates))
        part = np.argpartition(-score[candidates], k - 1)[:k]
        top_idx_unsorted = candidates[part]
        order = np.argsort(-score[top_idx_unsorted])
        top_idx = top_idx_unsorted[order]
        if return_scores:
            return [(int(i), float(score[i])) for i in top_idx]
        return [int(i) for i in top_idx]

    def search_multi(
        self,
        queries: Sequence[str],
        top_k: int = 10,
        min_score: float = 0.0,
        return_scores: bool = True,
    ) -> List[Tuple[int, float]] | List[int]:
        if not queries:
            return []
        # combina por score máximo entre as consultas
        all_scores: Optional[np.ndarray] = None
        for q in queries:
            res = self._search_single(q, top_k=len(self.corpus), min_score=0.0, return_scores=True)
            if not res:
                continue
            idxs, scores = zip(*res)
            tmp = np.zeros(len(self.corpus), dtype=float)
            tmp[list(idxs)] = np.array(scores, dtype=float)
            if all_scores is None:
                all_scores = tmp
            else:
                all_scores = np.maximum(all_scores, tmp)
        if all_scores is None:
            return []
        candidates = np.where(all_scores >= min_score)[0]
        if len(candidates) == 0:
            return []
        k = min(top_k, len(candidates))
        part = np.argpartition(-all_scores[candidates], k - 1)[:k]
        top_idx_unsorted = candidates[part]
        order = np.argsort(-all_scores[top_idx_unsorted])
        top_idx = top_idx_unsorted[order]
        if return_scores:
            return [(int(i), float(all_scores[i])) for i in top_idx]
        return [int(i) for i in top_idx]
