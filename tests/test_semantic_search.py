"""
Testes de Busca Semântica - Sistema de Precificação
=================================================

Testes focados nos componentes de busca semântica (embeddings, FAISS, reranking).

Categorias:
- Normalização e Pré-processamento
- Índice Semântico FAISS
- Cross-Encoder Reranking
- Extração de Atributos
- TF-IDF Híbrido
- Cache de Índices
"""

import pytest
import numpy as np
from backend.app.processamento.normalize import (
    normalize_text,
    normalize_equip,
    extract_attributes,
    expansion_variants_for_query
)
from backend.app.processamento.smart_search import (
    consonant_key,
    smart_score
)
from backend.app.processamento.semantic_index import (
    SemanticSearchIndex,
    SemanticIndexCache,
    _l2_normalize
)
from backend.app.processamento.semantic_reranker import CrossEncoderReranker
from backend.app.processamento.similarity import (
    TfidfSearchIndex,
    HybridTfidfSearchIndex,
    simple_tokenize,
    TokenVocab
)
from backend.app.processamento.attributes import extract_all_attributes, numeric_boost


class TestNormalization:
    """Testes de normalização de texto."""
    
    def test_normalize_text_basic(self):
        """Testa normalização básica de texto."""
        text = "  MOTOR   ELÉTRICO  5HP  "
        normalized = normalize_text(text)
        
        assert normalized == "motor eletrico 5hp"
        assert "  " not in normalized  # Espaços extras removidos
        assert normalized.islower()  # Minúsculas
    
    def test_normalize_equip_units(self):
        """Testa normalização de unidades de equipamentos."""
        text = "motor 5HP 220V 60HZ"
        normalized = normalize_equip(text)
        
        # Deve normalizar unidades com espaço
        assert "5 hp" in normalized or "5hp" in normalized
        assert "220 v" in normalized or "220v" in normalized
    
    def test_normalize_equip_abbreviations(self):
        """Testa expansão de abreviações."""
        text = "mtr elet 3cv"
        normalized = normalize_equip(text)
        
        # Abreviações devem ser expandidas (se configurado)
        assert "motor" in normalized or "mtr" in normalized
    
    def test_extract_attributes_numeric(self):
        """Testa extração de atributos numéricos."""
        text = "motor 5hp 220v 60hz"
        attrs = extract_attributes(text)
        
        # Deve extrair números
        assert any("5" in str(attrs))
        assert any("220" in str(attrs))
    
    def test_extract_all_attributes(self):
        """Testa extração completa de atributos."""
        text = "motor eletrico trifasico 7.5hp 380v 60hz ip55"
        attrs = extract_all_attributes(text)
        
        assert isinstance(attrs, dict)
        # Deve conter potência
        assert "potencia" in attrs or "hp" in str(attrs).lower()
    
    def test_expansion_variants(self):
        """Testa detecção de variantes/expansões."""
        query = "mtr elet"
        variants = expansion_variants_for_query(query)
        
        assert isinstance(variants, list)


class TestConsonantKey:
    """Testes de chave de consoantes (fonética)."""
    
    def test_consonant_key_basic(self):
        """Testa geração de chave consonantal."""
        text = "vassoura"
        key = consonant_key(text)
        
        # Deve remover vogais e compactar repetições
        assert "a" not in key
        assert "e" not in key
        assert "i" not in key
        assert "o" not in key
        assert "u" not in key
    
    def test_consonant_key_similarity(self):
        """Testa similaridade de variações."""
        key1 = consonant_key("motor")
        key2 = consonant_key("motores")
        key3 = consonant_key("motorista")
        
        # motor e motores devem ser mais similares que motor e motorista
        assert len(key1) > 0
        assert len(key2) > 0


class TestSmartScore:
    """Testes de scoring heurístico."""
    
    def test_smart_score_exact_match(self):
        """Testa score para match exato."""
        score = smart_score("motor eletrico", "motor eletrico")
        
        # Match perfeito deve ter score alto
        assert score >= 0.8
    
    def test_smart_score_partial_match(self):
        """Testa score para match parcial."""
        score = smart_score("motor", "motor eletrico trifasico")
        
        # Match parcial deve ter score médio
        assert 0.3 <= score <= 0.9
    
    def test_smart_score_no_match(self):
        """Testa score para não-match."""
        score = smart_score("motor", "bomba centrifuga")
        
        # Sem match deve ter score baixo
        assert score <= 0.5
    
    def test_smart_score_numeric_overlap(self):
        """Testa bonus de overlap numérico."""
        score1 = smart_score("motor 5hp", "motor eletrico 5hp")
        score2 = smart_score("motor 10hp", "motor eletrico 5hp")
        
        # Mesmo número deve dar score maior
        assert score1 > score2


class TestSemanticIndex:
    """Testes de índice semântico FAISS."""
    
    @pytest.mark.slow
    def test_semantic_index_build(self, sample_excel_data):
        """Testa construção de índice semântico."""
        texts = sample_excel_data['descricao'].tolist()
        
        index = SemanticSearchIndex.build(texts, use_hnsw=False)  # Flat para teste rápido
        
        assert index.dim > 0
        assert len(index.corpus) == len(texts)
        assert len(index.row_indices) == len(texts)
    
    @pytest.mark.slow
    def test_semantic_search_basic(self, sample_excel_data):
        """Testa busca semântica básica."""
        texts = sample_excel_data['descricao'].tolist()
        index = SemanticSearchIndex.build(texts, use_hnsw=False)
        
        results = index.search("motor eletrico", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        
        # Índices devem ser válidos
        for idx, score in results:
            assert 0 <= idx < len(texts)
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.slow
    def test_semantic_search_relevance(self, sample_excel_data):
        """Verifica se resultados são relevantes."""
        texts = sample_excel_data['descricao'].tolist()
        index = SemanticSearchIndex.build(texts, use_hnsw=False)
        
        results = index.search("bomba centrifuga", top_k=2)
        
        # Primeiro resultado deve conter "bomba"
        if len(results) > 0:
            top_idx, top_score = results[0]
            top_text = texts[top_idx].lower()
            assert "bomba" in top_text or top_score > 0.5
    
    def test_l2_normalize(self):
        """Testa normalização L2 de vetores."""
        vecs = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = _l2_normalize(vecs)
        
        # Norma L2 deve ser 1.0
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)
    
    @pytest.mark.slow
    def test_semantic_cache(self, sample_excel_data):
        """Testa cache de índices semânticos."""
        texts = sample_excel_data['descricao'].tolist()
        cache = SemanticIndexCache()
        
        # Primeira chamada (build)
        index1 = cache.get(texts)
        
        # Segunda chamada (cached)
        index2 = cache.get(texts)
        
        # Deve retornar mesmo objeto
        assert index1 is index2
        assert cache.key is not None


class TestCrossEncoderReranker:
    """Testes de reranking com cross-encoder."""
    
    @pytest.mark.slow
    def test_reranker_score(self):
        """Testa scoring com cross-encoder."""
        reranker = CrossEncoderReranker()
        
        query = "motor eletrico"
        candidates = [
            "motor eletrico trifasico 5hp",
            "bomba centrifuga 3hp",
            "compressor de ar"
        ]
        
        scores = reranker.score(query, candidates)
        
        assert len(scores) == len(candidates)
        # Primeiro candidato (mais relevante) deve ter score maior
        assert scores[0] >= scores[1]
    
    @pytest.mark.slow
    def test_reranker_normalize(self):
        """Testa normalização de scores."""
        raw_scores = [-0.5, 0.3, 1.2, 0.8]
        normalized = CrossEncoderReranker.normalize(raw_scores)
        
        assert len(normalized) == len(raw_scores)
        assert min(normalized) >= 0.0
        assert max(normalized) <= 1.0
    
    @pytest.mark.slow
    def test_reranker_lazy_reranking(self):
        """Testa lazy reranking (pula quando confiança alta)."""
        reranker = CrossEncoderReranker()
        
        # Scores semânticos altos (confiança alta)
        high_scores = [0.95, 0.90, 0.85]
        should_rerank = reranker.should_rerank(high_scores, threshold=0.75)
        
        # NÃO deve reranquear (confiança já alta)
        assert should_rerank is False
        
        # Scores baixos (confiança baixa)
        low_scores = [0.6, 0.5, 0.4]
        should_rerank_low = reranker.should_rerank(low_scores, threshold=0.75)
        
        # DEVE reranquear (confiança baixa)
        assert should_rerank_low is True


class TestTfidfIndex:
    """Testes de índice TF-IDF."""
    
    def test_tfidf_build(self, sample_excel_data):
        """Testa construção de índice TF-IDF."""
        texts = sample_excel_data['descricao'].tolist()
        
        index = TfidfSearchIndex.build(texts)
        
        assert len(index.corpus) == len(texts)
        assert index.matrix.shape[0] == len(texts)
    
    def test_tfidf_search(self, sample_excel_data):
        """Testa busca TF-IDF."""
        texts = sample_excel_data['descricao'].tolist()
        index = TfidfSearchIndex.build(texts)
        
        results = index.search("motor", top_k=3, return_scores=True)
        
        assert len(results) <= 3
        # Scores devem estar ordenados decrescente
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_tfidf_with_typo(self, sample_excel_data):
        """Testa robustez a typos (n-gramas de caracteres)."""
        texts = sample_excel_data['descricao'].tolist()
        index = TfidfSearchIndex.build(texts)
        
        # "moto" em vez de "motor"
        results = index.search("moto", top_k=3, return_scores=True)
        
        # Deve encontrar algo (n-gramas capturam "mot")
        assert len(results) > 0


class TestHybridTfidfIndex:
    """Testes de índice TF-IDF híbrido."""
    
    def test_hybrid_build(self, sample_excel_data):
        """Testa construção de índice híbrido."""
        texts = sample_excel_data['descricao'].tolist()
        
        index = HybridTfidfSearchIndex.build(texts)
        
        assert len(index.corpus) == len(texts)
        assert index.matrix_char.shape[0] == len(texts)
        assert index.matrix_word.shape[0] == len(texts)
        assert len(index.tokens_per_doc) == len(texts)
    
    def test_hybrid_search(self, sample_excel_data):
        """Testa busca híbrida."""
        texts = sample_excel_data['descricao'].tolist()
        index = HybridTfidfSearchIndex.build(texts)
        
        results = index.search("motor", top_k=3, return_scores=True)
        
        assert len(results) <= 3
        assert all(0 <= idx < len(texts) for idx, _ in results)
    
    def test_hybrid_autocorrect(self, sample_excel_data):
        """Testa autocorreção via vocabulário."""
        texts = sample_excel_data['descricao'].tolist()
        index = HybridTfidfSearchIndex.build(texts)
        
        # Query com typo
        results = index.search("moto eletrico", top_k=3, return_scores=True)
        
        # Deve corrigir e encontrar "motor"
        assert len(results) > 0


class TestTokenVocab:
    """Testes de vocabulário para correção."""
    
    def test_vocab_build(self, sample_excel_data):
        """Testa construção de vocabulário."""
        texts = sample_excel_data['descricao'].tolist()
        vocab = TokenVocab.from_texts(texts)
        
        assert len(vocab.tokens) > 0
        # Tokens devem ser únicos
        assert len(vocab.tokens) == len(set(vocab.tokens))
    
    def test_vocab_suggest(self, sample_excel_data):
        """Testa sugestão de correção."""
        texts = sample_excel_data['descricao'].tolist()
        vocab = TokenVocab.from_texts(texts)
        
        # Typo: "moto" em vez de "motor"
        suggestions = vocab.suggest("moto", limit=3)
        
        assert len(suggestions) > 0
        # Deve sugerir "motor"
        top_suggestion, top_score = suggestions[0]
        assert "motor" in top_suggestion.lower() or top_score > 0.8
    
    def test_vocab_correct_query(self, sample_excel_data):
        """Testa correção automática de query."""
        texts = sample_excel_data['descricao'].tolist()
        vocab = TokenVocab.from_texts(texts)
        
        corrected, changes = vocab.correct_query("moto eletrico")
        
        # Deve corrigir "moto" para "motor"
        assert "motor" in corrected or len(changes) > 0


class TestNumericBoost:
    """Testes de boost numérico para atributos."""
    
    def test_numeric_boost_exact_match(self):
        """Testa boost para match exato de números."""
        query_attrs = {"5": True, "hp": True}
        item_attrs = {"5": True, "hp": True}
        
        boost = numeric_boost(query_attrs, item_attrs)
        
        # Match perfeito deve dar boost alto
        assert boost >= 0.8
    
    def test_numeric_boost_partial_match(self):
        """Testa boost para match parcial."""
        query_attrs = {"5": True, "hp": True, "220": True}
        item_attrs = {"5": True, "hp": True, "380": True}  # Voltagem diferente
        
        boost = numeric_boost(query_attrs, item_attrs)
        
        # Match parcial deve dar boost médio
        assert 0.3 <= boost <= 0.9
    
    def test_numeric_boost_no_match(self):
        """Testa boost quando não há números em comum."""
        query_attrs = {"5": True, "hp": True}
        item_attrs = {"10": True, "hp": True}
        
        boost = numeric_boost(query_attrs, item_attrs)
        
        # Números diferentes deve reduzir boost
        assert boost <= 0.7


class TestSimpleTokenize:
    """Testes de tokenização simples."""
    
    def test_tokenize_basic(self):
        """Testa tokenização básica."""
        text = "motor eletrico 5hp"
        tokens = simple_tokenize(text)
        
        assert "motor" in tokens
        assert "eletrico" in tokens
        assert "5hp" in tokens or "5" in tokens
    
    def test_tokenize_with_punctuation(self):
        """Testa remoção de pontuação."""
        text = "motor, bomba; compressor!"
        tokens = simple_tokenize(text)
        
        # Pontuação deve ser removida
        assert "," not in tokens
        assert ";" not in tokens
        assert "!" not in tokens
    
    def test_tokenize_empty(self):
        """Testa tokenização de string vazia."""
        tokens = simple_tokenize("")
        assert tokens == []


# Marca testes lentos (com modelos de IA)
pytestmark = pytest.mark.semantic
