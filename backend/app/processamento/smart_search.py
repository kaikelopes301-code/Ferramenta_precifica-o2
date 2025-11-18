import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 🚀 Quick Win Part 4: Import normalization from centralized module
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

# Re-load for backward compatibility (if needed by other modules)
# Main usage should be via normalize module
# with open(ABBREV_PATH, 'r', encoding='utf-8') as f:
#     ABBREV_MAP: Dict[str,str] = json.load(f)

# Legacy constants for backward compatibility
VOGAIS = set('aeiou')
NUM_RE = re.compile(r'\d+[\.,]?\d*')


def _split_clean(s: str) -> List[str]:
    """Local helper for smart_search - kept for backward compatibility."""
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return [t for t in s.split() if t]


def consonant_key(text: str) -> str:
    norm = normalize_equip(text)
    parts = []
    for w in norm.split():
        cseq = ''.join(ch for ch in w if ch.isalpha() and ch not in VOGAIS)
        # remove repetições
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
    return [n.replace(',', '.') for n in NUM_RE.findall(norm_text)]


def score_numeric_overlap(q_nums: List[str], doc_nums: List[str]) -> float:
    if not q_nums or not doc_nums:
        return 0.0
    q_set = set(q_nums)
    d_set = set(doc_nums)
    inter = len(q_set & d_set)
    return inter / max(len(q_set), 1)


def score_consonant(q_key: str, d_key: str) -> float:
    if not q_key or not d_key:
        return 0.0
    # simple overlap
    q_set = set(q_key)
    d_set = set(d_key)
    inter = len(q_set & d_set)
    return inter / max(len(q_set), 1)


def smart_score(query: str, candidate: str) -> float:
    q_norm = normalize_equip(query)
    d_norm = normalize_equip(candidate)
    q_key = consonant_key(query)
    d_key = consonant_key(candidate)
    q_nums = extract_numbers(q_norm)
    d_nums = extract_numbers(d_norm)
    # base token overlap
    q_tokens = set(q_norm.split())
    d_tokens = set(d_norm.split())
    token_overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
    numeric = score_numeric_overlap(q_nums, d_nums)
    cons = score_consonant(q_key, d_key)
    # pesos heurísticos
    return 0.5 * token_overlap + 0.3 * cons + 0.2 * numeric


class SmartSearchAdapter:
    """Adapta um índice existente (como HybridTfidfSearchIndex) adicionando rerank por heurísticas.
    Expectativa: índice tem atributos corpus (List[str]) e método search(query,...).
    """
    def __init__(self, base_index):
        self.base = base_index
        self.corpus = getattr(base_index, 'corpus', [])

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        # primeiro pega mais candidatos do que precisa, usando query normalizada/expandida
        q_norm = normalize_equip(query)
        base_res = self.base.search(q_norm, top_k=max(top_k * 8, 80), return_scores=True)
        # fallback: se vazio, tenta com a original
        if not base_res:
            base_res = self.base.search(query, top_k=max(top_k * 8, 80), return_scores=True)
        if not base_res:
            return []
        rescored: List[Tuple[int, float]] = []
        for idx, _ in base_res:
            try:
                cand_text = self.corpus[idx]
            except Exception:
                continue
            hscore = smart_score(query, cand_text)
            # combina base score + heurística (normaliza) - assume base entre 0 e 1 (tfidf cosine approx)
            combined = 0.6 * _ + 0.4 * hscore
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
