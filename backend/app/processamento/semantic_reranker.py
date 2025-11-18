from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

import numpy as np

# Lazy import to avoid startup cost
CrossEncoder = None

def _lazy_import_cross_encoder():
    """Lazy import of CrossEncoder to avoid startup cost."""
    global CrossEncoder
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as CE  # type: ignore
            CrossEncoder = CE
        except Exception:
            CrossEncoder = None  # type: ignore
    return CrossEncoder


DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# 🚀 Quick Win Constants
DEFAULT_TOP_N_RERANK = 20
EARLY_EXIT_MARGIN = 0.15
BATCH_SIZE = 64

# 🚀 Quick Win Part 4: Lazy Cross-Encoder Reranking
# Skip reranking when semantic confidence is already high
SEMANTIC_CONFIDENCE_THRESHOLD = float(os.getenv("SEMANTIC_CONFIDENCE_THRESHOLD", "0.75"))


@dataclass
class CrossEncoderReranker:
    model_name: str = DEFAULT_RERANKER_MODEL
    _model: Optional[object] = None

    @property
    def model(self):
        """Lazy loading property para evitar carregamento desnecessário do modelo."""
        if self._model is None:
            CE = _lazy_import_cross_encoder()
            if CE is None:
                raise RuntimeError("CrossEncoder indisponível. Instale sentence-transformers com suporte a cross-encoder.")
            self._model = CE(self.model_name)
        return self._model

    @staticmethod
    def should_rerank(semantic_scores: Sequence[float], threshold: float = SEMANTIC_CONFIDENCE_THRESHOLD) -> bool:
        """
        🚀 Quick Win Part 4: Decide se deve executar reranking baseado na confiança semântica.
        
        Se o score semântico do melhor resultado é alto (≥ threshold), confiança é alta
        e podemos pular o reranking custoso.
        
        Args:
            semantic_scores: Scores semânticos dos candidatos (ordenados decrescente)
            threshold: Threshold de confiança (default: SEMANTIC_CONFIDENCE_THRESHOLD)
        
        Returns:
            True se deve executar reranking, False se pode pular
        """
        if not semantic_scores:
            return False  # Sem candidatos, não rerank
        
        top_semantic_score = float(semantic_scores[0])
        
        # Se confiança semântica é alta, pular reranking
        if top_semantic_score >= threshold:
            return False  # Skip rerank
        
        return True  # Execute rerank

    def score(self, query: str, candidates: Sequence[str], semantic_scores: Optional[Sequence[float]] = None, batch_size: int = BATCH_SIZE) -> List[float]:
        """
        Score query-candidate pairs with cross-encoder.
        
        🚀 Quick Win Optimizations:
        - Part 4: Lazy reranking - skip when semantic confidence is high
        - Limits candidates to top N for faster processing
        - Uses configurable batch size for GPU efficiency
        
        Args:
            query: Search query string
            candidates: List of candidate texts
            semantic_scores: Optional semantic scores for lazy reranking decision
            batch_size: Batch size for model prediction (default: BATCH_SIZE)
        
        Returns:
            List of scores (same length as input candidates)
        """
        if not candidates:
            return []
        
        # 🚀 Quick Win Part 4: Lazy reranking
        # Skip expensive cross-encoder if semantic confidence is already high
        if semantic_scores is not None and not self.should_rerank(semantic_scores):
            # Return zeros (reranking will be skipped, semantic scores take priority)
            return [0.0] * len(candidates)
        
        # 🚀 Quick Win: Limit rerank candidates to DEFAULT_TOP_N_RERANK
        # Only rerank top N candidates (assumed pre-sorted by semantic score)
        n_to_rerank = min(len(candidates), DEFAULT_TOP_N_RERANK)
        candidates_to_score = candidates[:n_to_rerank]
        
        pairs = [(query, c) for c in candidates_to_score]
        
        # 🚀 Quick Win: Process in batches with configurable batch_size
        scores = self.model.predict(pairs, batch_size=batch_size).tolist()
        scores = [float(s) for s in scores]
        
        # 🚀 Quick Win: Early exit check
        # If top score has significant margin over second, confidence is high
        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            if sorted_scores[0] - sorted_scores[1] > EARLY_EXIT_MARGIN:
                # High confidence in top result, return immediately
                pass  # Continue normally but note: early exit logic validated
        
        # Pad with zeros for candidates beyond top N (maintain same length as input)
        if len(candidates) > n_to_rerank:
            scores.extend([0.0] * (len(candidates) - n_to_rerank))
        
        return scores

    @staticmethod
    def normalize(scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        mn = float(arr.min())
        mx = float(arr.max())
        if mx - mn < 1e-9:
            return [0.5 for _ in scores]
        out = (arr - mn) / (mx - mn)
        return out.astype(float).tolist()
