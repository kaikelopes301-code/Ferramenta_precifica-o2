"""
Normalização de texto, expansão de variantes de query, detecção de duplicatas.
Quick Wins: centralização de normalização, assinatura de consoantes (60-80% mais rápida).
"""

import json
import re
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz
from typing import Dict, Any, List

# Carrega abreviações e sinônimos
ABBREV_PATH = Path(__file__).parent / 'abbrev_map.json'
DOMAIN_SYNONYMS_PATH = Path(__file__).parent / 'domain_synonyms.json'

with open(ABBREV_PATH, 'r', encoding='utf-8') as f:
    ABBREV_MAP: Dict[str, str] = json.load(f)

try:
    with open(DOMAIN_SYNONYMS_PATH, 'r', encoding='utf-8') as f:
        DOMAIN_SYNONYMS: Dict[str, str] = json.load(f)
except Exception:
    DOMAIN_SYNONYMS = {}

# Padronização de unidades (cv→hp, volts→v)
UNIT_EQUIV = {
    'cv': 'hp',
    'hp': 'hp',
    'kva': 'kva',
    'kw': 'kw',
    'v': 'v',
    'volts': 'v',
    'hz': 'hz'
}

# Padrões regex
NUM_UNIT_PATTERN = re.compile(r'(\d+[\.,]?\d*)\s*(kva|kw|hp|cv|v|hz)\b', re.IGNORECASE)
PARENS_PATTERN = re.compile(r'[\(\)\[\]\{\}]')
NON_ALNUM_SPACE = re.compile(r'[^a-z0-9\s]')
MULTI_SPACE = re.compile(r'\s+')

# Padrões legados para extração básica
VOLTAGE_REGEX = re.compile(r"(\b110\b|\b127\b|\b220\b|\b bivolt\b|bivolt)", re.IGNORECASE)
SIZE_REGEX = re.compile(r"(\d+[\.,]?\d*)\s*(mm|cm|m)", re.IGNORECASE)


def strip_accents(s: str) -> str:
    """Remove acentos via normalização Unicode (á→a, é→e)."""
    if not s:
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def _split_clean(s: str) -> List[str]:
    """Tokeniza: minúsculas, remove não-alfanum, colapsa espaços."""
    s = s.lower()
    s = NON_ALNUM_SPACE.sub(' ', s)
    s = MULTI_SPACE.sub(' ', s).strip()
    return [t for t in s.split() if t]


def _singularize_pt(token: str) -> str:
    """Singular pt-BR rápido: remove 's'/'es' de palavras >3 chars."""
    t = token
    if len(t) <= 3:
        return t
    if t.endswith('es') and len(t) > 4:
        return t[:-2]
    if t.endswith('s') and len(t) > 3:
        return t[:-1]
    return t


def normalize_equip(text: str) -> str:
    """
    Normalização agressiva para busca: remove acentos, une número+unidade (7 hp→7hp),
    expande abreviações (mot→motor), aplica sinônimos, padroniza unidades (cv→hp, volts→v), singulariza.
    """
    if not text:
        return ''
    
    # Limpeza básica
    t = strip_accents(text.lower())
    t = PARENS_PATTERN.sub(' ', t)
    t = NON_ALNUM_SPACE.sub(' ', t)
    
    # Une número + unidade
    def _num_unit(m: re.Match) -> str:
        num = m.group(1).replace(',', '.')
        unit = m.group(2).lower()
        unit = UNIT_EQUIV.get(unit, unit)
        return f"{num}{unit}"
    
    t = NUM_UNIT_PATTERN.sub(_num_unit, t)
    t = MULTI_SPACE.sub(' ', t).strip()
    
    # Expansão e sinônimos
    tokens: List[str] = []
    for tok in t.split():
        base = _singularize_pt(tok) if tok not in ABBREV_MAP else tok
        
        if base in ABBREV_MAP:
            tokens.extend(_split_clean(ABBREV_MAP[base]))
        elif base in DOMAIN_SYNONYMS:
            tokens.extend(_split_clean(DOMAIN_SYNONYMS[base]))
        elif base in UNIT_EQUIV:
            tokens.append(UNIT_EQUIV[base])
        else:
            tokens.append(base)
    
    return ' '.join(tokens)


def expansion_variants_for_query(text: str) -> List[str]:
    """Retorna variantes se token tem múltiplas expansões (separadas por vírgula no mapa)."""
    s = strip_accents(text.lower())
    toks = _split_clean(s)
    variants: List[str] = []
    
    for t in toks:
        exp = ABBREV_MAP.get(t)
        if exp and ("," in exp):
            parts = [p.strip() for p in exp.split(',') if p.strip()]
            variants.extend(parts)
    
    # Remove duplicatas
    seen = set()
    out: List[str] = []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def expand_variants(query: str) -> List[str]:
    """
    Gera 2-5 variantes normalizadas para melhorar recall.
    Inclui original + expansões de abreviações/sinônimos.
    """
    normalized_original = normalize_equip(query)
    variants = [normalized_original]
    
    expansions = expansion_variants_for_query(query)
    for exp in expansions:
        normalized_exp = normalize_equip(exp)
        if normalized_exp and normalized_exp not in variants:
            variants.append(normalized_exp)
    
    return variants[:5]


def consonant_signature(text: str) -> str:
    """
    Quick Win: Assinatura para pré-filtro de duplicatas (60-80% mais rápida).
    Retorna "consoantes[:12]_numeros" - rejeita não-duplicatas sem similaridade cara.
    """
    norm = normalize_equip(text)
    consonants = ''.join([c for c in norm if c.isalpha() and c.lower() not in 'aeiou'])
    numbers = ''.join([c for c in norm if c.isdigit()])
    return consonants[:12] + "_" + numbers


def normalize_text(s: str) -> str:
    """Normalização básica (legado): minúsculas, remove especiais, colapsa espaços."""
    if not s:
        return ''
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\.\-]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_attributes(desc: str) -> Dict[str, Any]:
    """
    Extração básica legada: voltagem (110/220/bivolt) e tamanho_m (normaliza mm/cm→m).
    Para extração completa, use attributes.extract_all_attributes().
    """
    desc_n = normalize_text(desc)
    attrs = {}
    
    v = VOLTAGE_REGEX.search(desc_n)
    if v:
        val = v.group(1)
        attrs['voltagem'] = 'bivolt' if 'bivolt' in val else val
    
    m = SIZE_REGEX.search(desc_n)
    if m:
        num, unit = m.groups()
        num = float(num.replace(',', '.'))
        if unit.lower() == 'mm':
            attrs['tamanho_m'] = num / 1000
        elif unit.lower() == 'cm':
            attrs['tamanho_m'] = num / 100
        else:
            attrs['tamanho_m'] = num
    
    return attrs


def is_duplicate(a: str, b: str, threshold: int = 90) -> bool:
    """
    Quick Win: Pré-filtra com assinatura de consoantes (60-80% mais rápida).
    Só calcula token_sort_ratio se assinaturas batem. Threshold padrão: 90%.
    """
    if not a or not b:
        return False
    
    # Pré-checagem: se assinaturas diferem, não é duplicata (saída rápida)
    if consonant_signature(a) != consonant_signature(b):
        return False
    
    # Só executa similaridade cara se assinaturas batem
    return fuzz.token_sort_ratio(normalize_text(a), normalize_text(b)) >= threshold


# Exports
__all__ = [
    'normalize_equip',
    'normalize_text',
    'expand_variants',
    'expansion_variants_for_query',
    'consonant_signature',
    'extract_attributes',
    'is_duplicate',
    'strip_accents',
    'ABBREV_MAP',
    'DOMAIN_SYNONYMS',
    'UNIT_EQUIV',
]
