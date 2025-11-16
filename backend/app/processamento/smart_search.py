"""
Adaptadores para busca inteligente com scoring heurístico.
Quick Win Part 4: Importa normalização centralizada de normalize.py.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Quick Win Part 4: Importa normalização centralizada
from .normalize import (
    normalize_equip,
    strip_accents,
    expansion_variants_for_query,
    expand_variants,
    ABBREV_MAP,
    DOMAIN_SYNONYMS,
    UNIT_EQUIV,
)

ABBREV_PATH = Path(__file__).parent / 'abbrev_map.json'
DOMAIN_SYNONYMS_PATH = Path(__file__).parent / 'domain_synonyms.json'

# Constantes legadas
VOGAIS = set('aeiou')
NUM_RE = re.compile(r'\d+[\.,]?\d*')


def _split_clean(s: str) -> List[str]:
    """Helper local - retrocompatibilidade."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return [t for t in s.split() if t]


def consonant_key(text: str) -> str:
    """Chave de consoantes comprimida (remove vogais, comprime repetições)."""
    norm = normalize_equip(text)
    parts = []
    for w in norm.split():
        cseq = ''.join(ch for ch in w if ch.isalpha() and ch not in VOGAIS)
        # Remove repetições consecutivas
        comp = []
        last = ''
        for ch in cseq:
            if ch != last:
                comp.append(ch)
                last = ch
        if comp:
            parts.append(''.join(comp))
    return ''.join(parts)


def extract_numbers(norm_text: str) -> List[str]:
    """Extrai todos os números de um texto normalizado."""
    return [n.replace(',', '.') for n in NUM_RE.findall(norm_text)]


def score_numeric_overlap(q_nums: List[str], doc_nums: List[str]) -> float:
    """
    Calcula overlap numérico entre query e documento.
    
    Retorna razão de números da query presentes no documento.
    """
    if not q_nums or not doc_nums:
        return 0.0
    q_set = set(q_nums)
    d_set = set(doc_nums)
    inter = len(q_set & d_set)
    return inter / max(len(q_set), 1)


def score_consonant(q_key: str, d_key: str) -> float:
    """
    Calcula similaridade baseada em chaves de consoantes.
    
    Retorna overlap de conjuntos de consoantes únicas.
    """
    if not q_key or not d_key:
        return 0.0
    # Simple overlap
    q_set = set(q_key)
    d_set = set(d_key)
    inter = len(q_set & d_set)
    return inter / max(len(q_set), 1)


def smart_score(query: str, candidate: str) -> float:
    """
    Calcula score heurístico combinando múltiplas características.
    
    Componentes:
    -----------
    - Token overlap (peso 0.5): Palavras em comum
    - Consonant similarity (peso 0.3): Estrutura consonantal
    - Numeric overlap (peso 0.2): Números em comum
    
    Args:
        query: Query de busca
        candidate: Texto candidato
        
    Returns:
        Score combinado [0, 1]
    """
    q_norm = normalize_equip(query)
    d_norm = normalize_equip(candidate)
    q_key = consonant_key(query)
    d_key = consonant_key(candidate)
    q_nums = extract_numbers(q_norm)
    d_nums = extract_numbers(d_norm)
    
    # Base token overlap
    q_tokens = set(q_norm.split())
    d_tokens = set(d_norm.split())
    token_overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
    numeric = score_numeric_overlap(q_nums, d_nums)
    cons = score_consonant(q_key, d_key)
    
    # Pesos heurísticos
    return 0.5 * token_overlap + 0.3 * cons + 0.2 * numeric


class SmartSearchAdapter:
    """
    Adapta um índice existente adicionando reranking por heurísticas smart_score.
    
    Uso típico:
    ----------
    Envolve um HybridTfidfSearchIndex ou similar para melhorar precisão do ranking
    final através de heurísticas específicas do domínio.
    
    Expectativa: índice base tem atributos corpus (List[str]) e método search(query, ...).
    """
    def __init__(self, base_index):
        """
        Inicializa adapter com índice base.
        
        Args:
            base_index: Índice de busca existente (ex: HybridTfidfSearchIndex)
        """
        self.base = base_index
        self.corpus = getattr(base_index, 'corpus', [])

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Busca com reranking heurístico.
        
        Processo:
        --------
        1. Busca mais candidatos que necessário (8x) no índice base
        2. Aplica smart_score a cada candidato
        3. Combina score base com score heurístico (60% base + 40% heurístico)
        4. Reordena e retorna top-k
        
        Args:
            query: Query de busca
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de tuplas (índice, score_combinado) ordenada por score
        """
        # Primeiro pega mais candidatos do que precisa, usando query normalizada/expandida
        q_norm = normalize_equip(query)
        base_res = self.base.search(q_norm, top_k=max(top_k * 8, 80), return_scores=True)
        
        # Fallback: se vazio, tenta com a original
        if not base_res:
            base_res = self.base.search(query, top_k=max(top_k * 8, 80), return_scores=True)
        if not base_res:
            return []
        
        # Reescore com heurística
        rescored: List[Tuple[int, float]] = []
        for idx, base_score in base_res:
            try:
                cand_text = self.corpus[idx]
            except Exception:
                continue
            hscore = smart_score(query, cand_text)
            # Combina base score + heurística (assume base entre 0 e 1)
            combined = 0.6 * base_score + 0.4 * hscore
            rescored.append((idx, combined))
        
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]


__all__ = [
    'normalize_equip', 
    'consonant_key', 
    'smart_score', 
    'SmartSearchAdapter',
    'expansion_variants_for_query',
    'expand_variants',
]
