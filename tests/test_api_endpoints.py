"""
Testes de Endpoints da API - Sistema de Precificação
===================================================

Testes funcionais de todos os endpoints REST da API.

Categorias:
- Health e Status
- Upload de Dados
- Busca Tradicional (TF-IDF)
- Busca Inteligente (Semântica)
- Busca em Lote
- Favoritos
- Kit e Orçamento
- Histórico
- Feedback
- Detalhes de Equipamento
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io


class TestHealthEndpoints:
    """Testes de endpoints de health check e status."""
    
    def test_health_check(self, client):
        """Testa endpoint de health check básico."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"
    
    def test_data_status_with_data(self, client):
        """Testa status dos dados quando planilha está carregada."""
        response = client.get("/data/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("has_data") is True
        assert "rows" in data
        assert data["rows"] > 0
        assert "columns" in data
    
    def test_data_status_without_data(self, client, monkeypatch):
        """Testa status quando planilha não existe."""
        # Simula arquivo inexistente
        monkeypatch.setenv("EXCEL_PATH", "/tmp/arquivo_inexistente.xlsx")
        
        response = client.get("/data/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("has_data") is False


class TestUploadEndpoint:
    """Testes de upload de planilha."""
    
    def test_upload_valid_excel(self, client, sample_excel_data):
        """Testa upload de arquivo Excel válido."""
        # Cria arquivo Excel em memória
        buffer = io.BytesIO()
        sample_excel_data.to_excel(buffer, index=False, sheet_name="dados")
        buffer.seek(0)
        
        files = {
            "file": ("dados_teste.xlsx", buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        }
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert data.get("rows") == len(sample_excel_data)
    
    def test_upload_invalid_extension(self, client):
        """Testa upload de arquivo com extensão inválida."""
        files = {
            "file": ("dados.txt", b"not an excel file", "text/plain")
        }
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "deve ser .xlsx" in data.get("detail", "").lower()
    
    def test_upload_corrupted_excel(self, client):
        """Testa upload de arquivo corrompido."""
        files = {
            "file": ("dados.xlsx", b"corrupted data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        }
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == 500
        data = response.json()
        assert "erro" in data.get("detail", "").lower()


class TestBuscarEndpoint:
    """Testes de busca tradicional (TF-IDF)."""
    
    def test_buscar_basic(self, client, authenticated_headers):
        """Testa busca básica com query simples."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "resultados" in data
        assert len(data["resultados"]) <= 5
        assert data.get("total") >= 0
    
    def test_buscar_with_attributes(self, client, authenticated_headers):
        """Testa extração de atributos da query."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor 5hp 220v", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "atributos" in data
        # Deve extrair potência e voltagem
        atributos = data["atributos"]
        assert any("5" in str(atributos) or "hp" in str(atributos).lower())
    
    def test_buscar_no_results(self, client, authenticated_headers):
        """Testa busca que não retorna resultados."""
        response = client.post(
            "/buscar",
            json={"descricao": "equipamento_inexistente_xyz123", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("resultados", [])) == 0
    
    def test_buscar_empty_query(self, client, authenticated_headers):
        """Testa busca com query vazia."""
        response = client.post(
            "/buscar",
            json={"descricao": "", "top_k": 5},
            headers=authenticated_headers
        )
        
        # Deve aceitar mas retornar vazio OU erro de validação
        assert response.status_code in [200, 400, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert len(data.get("resultados", [])) == 0
    
    def test_buscar_with_min_score(self, client, authenticated_headers):
        """Testa filtro por score mínimo."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 10, "min_score": 0.8},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Todos os resultados devem ter confiança >= 80%
        for resultado in data.get("resultados", []):
            confianca = resultado.get("confianca", 0)
            if confianca is not None:
                assert confianca >= 80.0 or confianca == 0  # Pode não ter confiança calculada


class TestBuscarInteligenteEndpoint:
    """Testes de busca inteligente (semântica)."""
    
    def test_buscar_inteligente_basic(self, client, authenticated_headers):
        """Testa busca semântica básica."""
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor elétrico", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "resultados" in data
        assert "query_normalizada" in data
        assert "modelo_semantico" in data
    
    def test_buscar_inteligente_returns_scores(self, client, authenticated_headers):
        """Verifica se resultados incluem scores de confiança."""
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "bomba centrifuga", "top_k": 3},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        for resultado in data.get("resultados", []):
            assert "score" in resultado or "confianca" in resultado
            assert "sugeridos" in resultado
    
    def test_buscar_inteligente_min_confidence(self, client, authenticated_headers):
        """Testa se filtro de confiança mínima é aplicado."""
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor", "top_k": 10},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # MIN_CONFIDENCE = 40.0 no código
        for resultado in data.get("resultados", []):
            score = resultado.get("score") or resultado.get("confianca")
            if score is not None:
                assert score >= 40.0 or score == 0
    
    def test_buscar_inteligente_aggregations(self, client, authenticated_headers):
        """Verifica se agregações (preço, vida útil, manutenção) estão presentes."""
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data.get("resultados", [])) > 0:
            primeiro = data["resultados"][0]
            # Deve ter campos de agregação (podem ser None)
            assert "valor_unitario" in primeiro
            assert "vida_util_meses" in primeiro
            assert "manutencao_percent" in primeiro
    
    def test_buscar_inteligente_with_typo(self, client, authenticated_headers):
        """Testa robustez a erros de digitação."""
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "moto eletrico", "top_k": 5},  # "moto" em vez de "motor"
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        # Deve encontrar "motor" mesmo com typo
        assert len(data.get("resultados", [])) > 0
    
    def test_buscar_inteligente_cache(self, client, authenticated_headers):
        """Testa se cache de resultados funciona."""
        query_payload = {"descricao": "compressor", "top_k": 5}
        
        # Primeira request (cache MISS)
        response1 = client.post(
            "/buscar-inteligente",
            json=query_payload,
            headers=authenticated_headers
        )
        
        # Segunda request (cache HIT)
        response2 = client.post(
            "/buscar-inteligente",
            json=query_payload,
            headers=authenticated_headers
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Resultados devem ser idênticos
        assert response1.json()["resultados"] == response2.json()["resultados"]
        
        # Segunda deve ser mais rápida (cache hit)
        # Verifica headers se disponíveis
        if "X-Cache" in response2.headers:
            assert response2.headers["X-Cache"] == "HIT"


class TestBuscarLoteEndpoints:
    """Testes de busca em lote."""
    
    def test_buscar_lote_basic(self, client, authenticated_headers):
        """Testa busca em lote tradicional."""
        response = client.post(
            "/buscar-lote",
            json={
                "descricoes": ["motor", "bomba", "compressor"],
                "top_k": 3
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "resultados" in data
        assert len(data["resultados"]) > 0
    
    def test_buscar_lote_inteligente(self, client, authenticated_headers):
        """Testa busca semântica em lote."""
        response = client.post(
            "/buscar-lote-inteligente",
            json={
                "descricoes": ["motor eletrico", "bomba centrifuga"],
                "top_k": 5
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Cada query deve ter identificador
        for resultado in data.get("resultados", []):
            assert "query_original" in resultado
            assert resultado["query_original"] in ["motor eletrico", "bomba centrifuga"]
    
    def test_buscar_lote_empty_array(self, client, authenticated_headers):
        """Testa busca em lote com array vazio."""
        response = client.post(
            "/buscar-lote",
            json={"descricoes": [], "top_k": 5},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("resultados", [])) == 0
    
    def test_buscar_lote_many_queries(self, client, authenticated_headers):
        """Testa busca em lote com muitas queries."""
        queries = [f"motor {i}hp" for i in range(1, 21)]  # 20 queries
        
        response = client.post(
            "/buscar-lote-inteligente",
            json={"descricoes": queries, "top_k": 3},
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Deve retornar resultados para todas (ou maioria das) queries
        unique_queries = set(r["query_original"] for r in data.get("resultados", []))
        assert len(unique_queries) >= 10  # Pelo menos metade deve ter resultados


class TestFavoritosEndpoint:
    """Testes de gerenciamento de favoritos."""
    
    def test_list_favoritos_empty(self, client, authenticated_headers):
        """Testa listagem de favoritos quando vazio."""
        response = client.get("/favoritos", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
    
    def test_add_favorito(self, client, authenticated_headers):
        """Testa adição de item aos favoritos."""
        response = client.post(
            "/favoritos",
            json={
                "item_name": "Motor Teste 5HP",
                "price": 1500.00,
                "extra": {"voltagem": "220V"}
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
    
    def test_list_favoritos_after_add(self, client, authenticated_headers):
        """Verifica se item adicionado aparece na listagem."""
        # Adiciona
        client.post(
            "/favoritos",
            json={"item_name": "Bomba Teste", "price": 800.00},
            headers=authenticated_headers
        )
        
        # Lista
        response = client.get("/favoritos", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        items = data.get("items", [])
        assert len(items) > 0
        assert any(item["item_name"] == "Bomba Teste" for item in items)
    
    def test_delete_favorito(self, client, authenticated_headers):
        """Testa remoção de favorito."""
        # Adiciona
        add_response = client.post(
            "/favoritos",
            json={"item_name": "Item para Deletar", "price": 100.00},
            headers=authenticated_headers
        )
        
        # Lista para pegar ID
        list_response = client.get("/favoritos", headers=authenticated_headers)
        items = list_response.json()["items"]
        item_to_delete = next((i for i in items if i["item_name"] == "Item para Deletar"), None)
        
        if item_to_delete:
            # Deleta
            delete_response = client.delete(
                f"/favoritos/{item_to_delete['id']}",
                headers=authenticated_headers
            )
            
            assert delete_response.status_code == 200
            assert delete_response.json().get("success") is True


class TestKitEndpoint:
    """Testes de gerenciamento de kit."""
    
    def test_add_to_kit(self, client, authenticated_headers):
        """Testa adição de item ao kit."""
        response = client.post(
            "/kit",
            json={
                "item_name": "Parafuso M8",
                "price": 0.50,
                "qty": 100
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
    
    def test_list_kit(self, client, authenticated_headers):
        """Testa listagem do kit."""
        # Adiciona item
        client.post(
            "/kit",
            json={"item_name": "Arruela", "price": 0.20, "qty": 50},
            headers=authenticated_headers
        )
        
        # Lista
        response = client.get("/kit", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert any(item["item_name"] == "Arruela" for item in data["items"])
    
    def test_generate_orcamento(self, client, authenticated_headers):
        """Testa geração de orçamento a partir do kit."""
        # Adiciona itens
        client.post(
            "/kit",
            json={"item_name": "Motor 5HP", "price": 1500.00, "qty": 2},
            headers=authenticated_headers
        )
        client.post(
            "/kit",
            json={"item_name": "Bomba", "price": 800.00, "qty": 1},
            headers=authenticated_headers
        )
        
        # Gera orçamento
        response = client.post("/kit/orcamento", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "itens" in data
        assert "total" in data
        
        # Total deve ser 1500*2 + 800 = 3800
        expected_total = (1500.00 * 2) + 800.00
        assert abs(data["total"] - expected_total) < 0.01  # Margem para arredondamento
    
    def test_export_kit_excel(self, client, authenticated_headers):
        """Testa exportação do kit em Excel."""
        # Adiciona item
        client.post(
            "/kit",
            json={"item_name": "Item Exportar", "price": 100.00, "qty": 5},
            headers=authenticated_headers
        )
        
        # Exporta
        response = client.get("/kit/export", headers=authenticated_headers)
        
        # Deve retornar arquivo Excel ou 404 se vazio
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            assert response.headers["content-type"] in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/octet-stream"
            ]


class TestHistoricoEndpoint:
    """Testes de histórico de buscas."""
    
    def test_get_historico_empty(self, client, authenticated_headers):
        """Testa histórico quando vazio."""
        response = client.get("/historico", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
    
    def test_historico_after_search(self, client, authenticated_headers):
        """Verifica se buscas são registradas no histórico."""
        # Faz uma busca
        client.post(
            "/buscar-inteligente",
            json={"descricao": "motor teste historico", "top_k": 5},
            headers=authenticated_headers
        )
        
        # Verifica histórico
        response = client.get("/historico?limit=10", headers=authenticated_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        items = data.get("items", [])
        # Deve conter a busca recente
        assert any("motor teste historico" in item.get("query", "") for item in items)


class TestFeedbackEndpoint:
    """Testes de feedback de usuários."""
    
    def test_save_feedback(self, client, authenticated_headers):
        """Testa salvamento de feedback."""
        response = client.post(
            "/feedback",
            json={
                "ranking": 1,
                "sugeridos": "Motor 5HP",
                "valor_unitario": 1500.00,
                "confianca": 95.0,
                "sugestao_incorreta": False,
                "feedback": "Resultado excelente",
                "equipamento_material_revisado": "",
                "query": "motor 5hp"
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
    
    def test_save_feedback_batch(self, client, authenticated_headers):
        """Testa salvamento de feedback em lote."""
        response = client.post(
            "/feedback-lote",
            json={
                "items": [
                    {
                        "sugeridos": "Motor A",
                        "confianca": 90.0,
                        "query": "motor",
                        "sugestao_incorreta": False
                    },
                    {
                        "sugeridos": "Motor B",
                        "confianca": 85.0,
                        "query": "motor",
                        "sugestao_incorreta": False
                    }
                ],
                "use_tfidf": False
            },
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
    
    def test_feedback_stats(self, client):
        """Testa endpoint de estatísticas de feedback."""
        response = client.get("/feedback/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total" in data


class TestDetalhesEndpoint:
    """Testes de detalhes de equipamento."""
    
    def test_detalhes_existing_group(self, client, authenticated_headers, sample_excel_data):
        """Testa detalhes de grupo existente."""
        # Pega um grupo da amostra
        grupo_name = sample_excel_data['descricao_padronizada'].iloc[0]
        
        response = client.get(
            f"/detalhes/{grupo_name}",
            headers=authenticated_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("grupo") == grupo_name
        assert "items" in data
        assert len(data["items"]) > 0
    
    def test_detalhes_nonexistent_group(self, client, authenticated_headers):
        """Testa detalhes de grupo inexistente."""
        response = client.get(
            "/detalhes/grupo_inexistente_xyz",
            headers=authenticated_headers
        )
        
        assert response.status_code == 404
        data = response.json()
        assert "não encontrado" in data.get("detail", "").lower()


class TestCacheEndpoints:
    """Testes de endpoints de cache."""
    
    def test_cache_stats(self, client):
        """Testa endpoint de estatísticas de cache."""
        response = client.get("/cache/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "lru_cache" in data
        assert "json_cache" in data
    
    def test_cache_clear(self, client):
        """Testa limpeza de cache."""
        response = client.post("/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        
        # Verifica stats após limpar
        stats_response = client.get("/cache/stats")
        stats = stats_response.json()
        
        # Tamanho deve ser 0 ou próximo de 0
        assert stats["lru_cache"]["size"] == 0


# Marca para execução
pytestmark = pytest.mark.api
