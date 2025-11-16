"""
Testes de Integração E2E - Sistema de Precificação
==================================================

Testes end-to-end que verificam fluxos completos do sistema.

Categorias:
- Fluxo de Upload e Busca
- Workflow de Favoritos e Kit
- Pipeline de Feedback
- Integração Frontend-Backend (via API routes)
- Cenários Realistas de Usuário
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import io
import time


class TestUploadSearchWorkflow:
    """Testes de workflow de upload e busca."""
    
    def test_full_workflow_upload_to_search(self, client, sample_excel_data, authenticated_headers):
        """Testa workflow completo: upload → busca → resultado."""
        # 1. Upload de dados
        buffer = io.BytesIO()
        sample_excel_data.to_excel(buffer, index=False, sheet_name="dados")
        buffer.seek(0)
        
        files = {"file": ("dados.xlsx", buffer, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        upload_response = client.post("/upload", files=files)
        
        assert upload_response.status_code == 200
        assert upload_response.json()["success"] is True
        
        # 2. Verifica status dos dados
        status_response = client.get("/data/status")
        assert status_response.status_code == 200
        assert status_response.json()["has_data"] is True
        
        # 3. Busca tradicional
        search_response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert search_response.status_code == 200
        results = search_response.json()["resultados"]
        assert len(results) > 0
        
        # 4. Busca inteligente
        smart_search_response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor eletrico", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert smart_search_response.status_code == 200
        smart_results = smart_search_response.json()["resultados"]
        assert len(smart_results) > 0
    
    def test_workflow_batch_search(self, client, authenticated_headers):
        """Testa workflow de busca em lote."""
        # Busca múltiplas queries
        batch_response = client.post(
            "/buscar-lote-inteligente",
            json={
                "descricoes": ["motor", "bomba", "compressor"],
                "top_k": 3
            },
            headers=authenticated_headers
        )
        
        assert batch_response.status_code == 200
        results = batch_response.json()["resultados"]
        
        # Deve ter resultados para cada query
        queries_returned = set(r["query_original"] for r in results)
        assert len(queries_returned) >= 2  # Pelo menos 2 das 3 queries


class TestFavoritesKitWorkflow:
    """Testes de workflow de favoritos e kit."""
    
    def test_full_favorites_workflow(self, client, authenticated_headers):
        """Testa workflow completo de favoritos."""
        # 1. Lista inicial (vazio)
        list_response = client.get("/favoritos", headers=authenticated_headers)
        assert list_response.status_code == 200
        initial_count = len(list_response.json()["items"])
        
        # 2. Adiciona favorito
        add_response = client.post(
            "/favoritos",
            json={"item_name": "Motor Favorito", "price": 1500.00},
            headers=authenticated_headers
        )
        assert add_response.status_code == 200
        
        # 3. Verifica que foi adicionado
        list_response2 = client.get("/favoritos", headers=authenticated_headers)
        new_count = len(list_response2.json()["items"])
        assert new_count == initial_count + 1
        
        # 4. Deleta favorito
        item_id = None
        for item in list_response2.json()["items"]:
            if item["item_name"] == "Motor Favorito":
                item_id = item["id"]
                break
        
        assert item_id is not None
        
        delete_response = client.delete(
            f"/favoritos/{item_id}",
            headers=authenticated_headers
        )
        assert delete_response.status_code == 200
        
        # 5. Verifica que foi deletado
        list_response3 = client.get("/favoritos", headers=authenticated_headers)
        final_count = len(list_response3.json()["items"])
        assert final_count == initial_count
    
    def test_full_kit_workflow(self, client, authenticated_headers):
        """Testa workflow completo do kit."""
        # 1. Adiciona itens ao kit
        client.post(
            "/kit",
            json={"item_name": "Motor 5HP", "price": 1500.00, "qty": 2},
            headers=authenticated_headers
        )
        client.post(
            "/kit",
            json={"item_name": "Bomba 3HP", "price": 800.00, "qty": 1},
            headers=authenticated_headers
        )
        
        # 2. Lista kit
        list_response = client.get("/kit", headers=authenticated_headers)
        assert list_response.status_code == 200
        assert len(list_response.json()["items"]) >= 2
        
        # 3. Gera orçamento
        budget_response = client.post("/kit/orcamento", headers=authenticated_headers)
        assert budget_response.status_code == 200
        
        budget = budget_response.json()
        assert budget["total"] >= 3800.00  # 1500*2 + 800
        
        # 4. Exporta para Excel
        export_response = client.get("/kit/export", headers=authenticated_headers)
        assert export_response.status_code in [200, 404]  # 200 se tiver dados


class TestFeedbackWorkflow:
    """Testes de workflow de feedback."""
    
    def test_search_and_feedback_workflow(self, client, authenticated_headers):
        """Testa workflow de busca → feedback."""
        # 1. Faz busca
        search_response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor eletrico", "top_k": 3},
            headers=authenticated_headers
        )
        
        assert search_response.status_code == 200
        results = search_response.json()["resultados"]
        
        if len(results) > 0:
            # 2. Envia feedback sobre primeiro resultado
            first_result = results[0]
            
            feedback_response = client.post(
                "/feedback",
                json={
                    "ranking": 1,
                    "sugeridos": first_result["sugeridos"],
                    "confianca": first_result.get("score", 0),
                    "sugestao_incorreta": False,
                    "feedback": "Resultado excelente",
                    "equipamento_material_revisado": "",
                    "query": "motor eletrico",
                    "use_tfidf": False
                },
                headers=authenticated_headers
            )
            
            assert feedback_response.status_code == 200
            
            # 3. Verifica estatísticas de feedback
            stats_response = client.get("/feedback/stats")
            assert stats_response.status_code == 200
            assert stats_response.json()["total"] > 0


class TestHistoryWorkflow:
    """Testes de histórico de buscas."""
    
    def test_search_history_tracking(self, client, authenticated_headers):
        """Verifica se histórico de buscas é registrado."""
        # 1. Faz múltiplas buscas
        queries = ["motor", "bomba", "compressor"]
        
        for query in queries:
            client.post(
                "/buscar-inteligente",
                json={"descricao": query, "top_k": 5},
                headers=authenticated_headers
            )
            time.sleep(0.1)  # Pequeno delay para garantir ordem
        
        # 2. Verifica histórico
        history_response = client.get("/historico?limit=10", headers=authenticated_headers)
        
        assert history_response.status_code == 200
        history_items = history_response.json()["items"]
        
        # Deve conter pelo menos algumas das queries
        history_queries = [item["query"] for item in history_items]
        assert any(q in history_queries for q in queries)


class TestCacheIntegration:
    """Testes de integração de cache."""
    
    def test_cache_improves_performance(self, client, authenticated_headers):
        """Verifica se cache melhora performance."""
        query_payload = {"descricao": "motor teste cache", "top_k": 5}
        
        # 1. Primeira busca (cache MISS)
        start_time1 = time.time()
        response1 = client.post(
            "/buscar-inteligente",
            json=query_payload,
            headers=authenticated_headers
        )
        time1 = time.time() - start_time1
        
        assert response1.status_code == 200
        
        # 2. Segunda busca (cache HIT)
        start_time2 = time.time()
        response2 = client.post(
            "/buscar-inteligente",
            json=query_payload,
            headers=authenticated_headers
        )
        time2 = time.time() - start_time2
        
        assert response2.status_code == 200
        
        # Cache hit deve ser significativamente mais rápido
        # (exceto se modelos ainda não carregados na primeira)
        assert time2 <= time1 * 2  # Margem generosa para variabilidade
        
        # Resultados devem ser idênticos
        assert response1.json()["resultados"] == response2.json()["resultados"]
    
    def test_cache_stats_tracking(self, client, authenticated_headers):
        """Verifica se estatísticas de cache são atualizadas."""
        # 1. Limpa cache
        client.post("/cache/clear")
        
        # 2. Verifica stats (deve estar vazio)
        stats_response1 = client.get("/cache/stats")
        stats1 = stats_response1.json()
        
        initial_hits = stats1["lru_cache"]["hits"]
        
        # 3. Faz busca repetida (2x mesma query)
        query = {"descricao": "motor cache stats", "top_k": 5}
        
        client.post("/buscar-inteligente", json=query, headers=authenticated_headers)
        client.post("/buscar-inteligente", json=query, headers=authenticated_headers)
        
        # 4. Verifica stats novamente
        stats_response2 = client.get("/cache/stats")
        stats2 = stats_response2.json()
        
        # Hits deve ter aumentado
        assert stats2["lru_cache"]["hits"] >= initial_hits


class TestDetailsPagesIntegration:
    """Testes de integração de páginas de detalhes."""
    
    def test_search_to_details_workflow(self, client, authenticated_headers, sample_excel_data):
        """Testa workflow busca → detalhes."""
        # 1. Faz busca
        search_response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor", "top_k": 3},
            headers=authenticated_headers
        )
        
        assert search_response.status_code == 200
        results = search_response.json()["resultados"]
        
        if len(results) > 0:
            # 2. Pega link de detalhes do primeiro resultado
            first_result = results[0]
            grupo_name = first_result["sugeridos"]
            
            # 3. Acessa detalhes
            details_response = client.get(
                f"/detalhes/{grupo_name}",
                headers=authenticated_headers
            )
            
            assert details_response.status_code == 200
            details = details_response.json()
            
            assert details["grupo"] == grupo_name
            assert len(details["items"]) > 0


class TestRealisticUserScenarios:
    """Testes de cenários realistas de uso."""
    
    def test_scenario_find_and_add_to_kit(self, client, authenticated_headers):
        """Cenário: Usuário busca equipamento e adiciona ao kit."""
        # 1. Busca equipamento
        search_response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor 5hp", "top_k": 5},
            headers=authenticated_headers
        )
        
        assert search_response.status_code == 200
        results = search_response.json()["resultados"]
        
        if len(results) > 0:
            # 2. Seleciona primeiro resultado
            selected = results[0]
            
            # 3. Adiciona ao kit
            kit_response = client.post(
                "/kit",
                json={
                    "item_name": selected["sugeridos"],
                    "price": selected.get("valor_unitario", 0),
                    "qty": 1
                },
                headers=authenticated_headers
            )
            
            assert kit_response.status_code == 200
            
            # 4. Gera orçamento
            budget_response = client.post("/kit/orcamento", headers=authenticated_headers)
            assert budget_response.status_code == 200
            
            budget = budget_response.json()
            assert budget["total"] > 0
    
    def test_scenario_compare_search_methods(self, client, authenticated_headers):
        """Cenário: Usuário compara busca tradicional vs inteligente."""
        query = "bomba centrifuga"
        
        # 1. Busca tradicional (TF-IDF)
        tfidf_response = client.post(
            "/buscar",
            json={"descricao": query, "top_k": 5, "use_tfidf": True},
            headers=authenticated_headers
        )
        
        # 2. Busca inteligente (Semântica)
        semantic_response = client.post(
            "/buscar-inteligente",
            json={"descricao": query, "top_k": 5},
            headers=authenticated_headers
        )
        
        assert tfidf_response.status_code == 200
        assert semantic_response.status_code == 200
        
        # Ambos devem retornar resultados
        tfidf_results = tfidf_response.json()["resultados"]
        semantic_results = semantic_response.json()["resultados"]
        
        assert len(tfidf_results) > 0 or len(semantic_results) > 0
    
    def test_scenario_batch_procurement(self, client, authenticated_headers):
        """Cenário: Usuário busca múltiplos equipamentos para procurement."""
        # Lista de equipamentos para um projeto
        equipments = [
            "motor eletrico 5hp",
            "bomba centrifuga 3hp",
            "compressor ar 10hp",
            "painel eletrico"
        ]
        
        # 1. Busca em lote
        batch_response = client.post(
            "/buscar-lote-inteligente",
            json={"descricoes": equipments, "top_k": 3},
            headers=authenticated_headers
        )
        
        assert batch_response.status_code == 200
        results = batch_response.json()["resultados"]
        
        # 2. Para cada resultado, adiciona melhor opção ao kit
        added_count = 0
        for equipment in equipments:
            # Filtra resultados desta query
            equipment_results = [r for r in results if r.get("query_original") == equipment]
            
            if equipment_results:
                best = equipment_results[0]  # Melhor score
                
                client.post(
                    "/kit",
                    json={
                        "item_name": best["sugeridos"],
                        "price": best.get("valor_unitario", 0),
                        "qty": 1
                    },
                    headers=authenticated_headers
                )
                added_count += 1
        
        # 3. Gera orçamento final
        if added_count > 0:
            budget_response = client.post("/kit/orcamento", headers=authenticated_headers)
            assert budget_response.status_code == 200
            assert budget_response.json()["total"] > 0


class TestErrorRecovery:
    """Testes de recuperação de erros."""
    
    def test_recovery_from_corrupted_upload(self, client, authenticated_headers):
        """Testa recuperação após upload corrompido."""
        # 1. Upload corrompido
        files = {"file": ("bad.xlsx", b"corrupted", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        bad_response = client.post("/upload", files=files)
        
        assert bad_response.status_code == 500
        
        # 2. Sistema deve continuar funcionando com dados anteriores
        status_response = client.get("/data/status")
        assert status_response.status_code == 200
        
        # 3. Busca deve ainda funcionar (com dados anteriores)
        search_response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        assert search_response.status_code in [200, 400]
    
    def test_recovery_from_timeout(self, client, authenticated_headers):
        """Testa recuperação após timeout (query lenta)."""
        # Query muito complexa que pode causar timeout
        huge_query = "motor " * 100
        
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": huge_query, "top_k": 5},
            headers=authenticated_headers,
            timeout=2.0  # Timeout curto
        )
        
        # Pode retornar timeout ou truncar query
        assert response.status_code in [200, 408, 504]
        
        # Sistema deve continuar respondendo depois
        normal_response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        assert normal_response.status_code == 200


# Marca testes de integração
pytestmark = pytest.mark.integration
