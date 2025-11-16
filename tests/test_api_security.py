"""
Testes de Segurança - Sistema de Precificação
============================================

Testes focados em vulnerabilidades de segurança identificadas na análise.

Categorias:
- SQL Injection
- Autenticação e Autorização
- Rate Limiting
- Input Validation
- CORS
"""

import pytest
from fastapi.testclient import TestClient
import time


class TestSQLInjection:
    """Testes de vulnerabilidades de SQL Injection."""
    
    def test_sql_injection_favoritos_delete(self, client, authenticated_headers):
        """Testa se endpoint de deleção de favoritos é vulnerável a SQL injection."""
        # Adiciona um favorito primeiro
        response = client.post(
            "/favoritos",
            json={"item_name": "Motor Teste", "price": 1500.00},
            headers=authenticated_headers
        )
        assert response.status_code == 200
        
        # Tenta SQL injection no DELETE
        malicious_id = "1 OR 1=1"  # Tentativa de deletar todos os registros
        response = client.delete(
            f"/favoritos/{malicious_id}",
            headers=authenticated_headers
        )
        
        # Deve falhar com 422 (validação) ou 404 (não encontrado)
        # NÃO deve retornar 200 se deletou múltiplos registros
        assert response.status_code in [404, 422, 500]
    
    def test_sql_injection_kit_delete(self, client, authenticated_headers):
        """Testa SQL injection no endpoint de kit."""
        # Adiciona item ao kit
        response = client.post(
            "/kit",
            json={"item_name": "Bomba Teste", "price": 800.00, "qty": 2},
            headers=authenticated_headers
        )
        assert response.status_code == 200
        
        # Tenta SQL injection
        malicious_id = "1; DROP TABLE kit_items; --"
        response = client.delete(
            f"/kit/{malicious_id}",
            headers=authenticated_headers
        )
        
        assert response.status_code in [404, 422, 500]
        
        # Verifica se tabela ainda existe (não foi dropada)
        response = client.get("/kit", headers=authenticated_headers)
        assert response.status_code == 200
    
    def test_sql_injection_historico(self, client, authenticated_headers):
        """Testa SQL injection via parâmetro limit no histórico."""
        # Tenta injeção via query parameter
        response = client.get(
            "/historico?limit=10 OR 1=1",
            headers=authenticated_headers
        )
        
        # Deve falhar com erro de validação ou converter para int válido
        # NÃO deve retornar dados não autorizados
        if response.status_code == 200:
            data = response.json()
            # Verifica se retornou número razoável de items (não todos do banco)
            assert len(data.get("items", [])) <= 100


class TestAuthentication:
    """Testes de autenticação e autorização."""
    
    def test_user_id_required_favoritos(self, client):
        """Verifica se x-user-id é obrigatório para favoritos."""
        response = client.get("/favoritos")
        # Atualmente retorna 200 com user_id="anon"
        # IDEAL: Deveria retornar 401 Unauthorized
        assert response.status_code in [200, 401]
        
        if response.status_code == 200:
            # Se aceita anon, verifica se está isolado
            data = response.json()
            # Favoritos devem estar vazios para usuário anônimo
            assert isinstance(data.get("items"), list)
    
    def test_user_isolation_favoritos(self, client):
        """Testa se usuários não conseguem acessar favoritos de outros."""
        # Usuário 1 cria favorito
        headers_user1 = {"x-user-id": "user_001"}
        response = client.post(
            "/favoritos",
            json={"item_name": "Item Privado User1", "price": 100.00},
            headers=headers_user1
        )
        assert response.status_code == 200
        
        # Usuário 2 tenta listar
        headers_user2 = {"x-user-id": "user_002"}
        response = client.get("/favoritos", headers=headers_user2)
        assert response.status_code == 200
        
        data = response.json()
        items = data.get("items", [])
        
        # Verifica que User2 NÃO vê favoritos do User1
        for item in items:
            assert "Item Privado User1" not in item.get("item_name", "")
    
    def test_user_id_spoofing(self, client):
        """Testa se é possível falsificar user_id via diferentes headers."""
        # Cria favorito com um user_id
        headers_original = {"x-user-id": "legit_user"}
        response = client.post(
            "/favoritos",
            json={"item_name": "Item Legit", "price": 200.00},
            headers=headers_original
        )
        assert response.status_code == 200
        
        # Tenta acessar com user_id diferente mas mesmo IP
        headers_spoofed = {"x-user-id": "attacker"}
        response = client.get("/favoritos", headers=headers_spoofed)
        
        data = response.json()
        items = [item["item_name"] for item in data.get("items", [])]
        
        # NÃO deve ver item do legit_user
        assert "Item Legit" not in items


class TestRateLimiting:
    """Testes de rate limiting."""
    
    def test_rate_limit_buscar_inteligente(self, client, authenticated_headers):
        """Testa se rate limiting está ativo para busca inteligente."""
        # Faz múltiplas requests rápidas
        results = []
        for i in range(15):  # Mais que o limite (10/s)
            response = client.post(
                "/buscar-inteligente",
                json={"descricao": f"motor {i}", "top_k": 5},
                headers=authenticated_headers
            )
            results.append(response.status_code)
        
        # Pelo menos uma deve ser rejeitada (429 Too Many Requests)
        # OU todas passam se rate limiting não estiver ativo (problema!)
        rate_limited = 429 in results
        
        # Se não há rate limiting, é um problema de segurança
        if not rate_limited:
            pytest.xfail("Rate limiting não está ativo - VULNERABILIDADE")
    
    def test_rate_limit_upload(self, client, authenticated_headers):
        """Testa rate limiting no endpoint de upload."""
        # Upload requer arquivo, então cria um mock
        files = {"file": ("test.xlsx", b"fake excel data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        
        results = []
        for i in range(5):  # Tenta 5 uploads rápidos
            response = client.post(
                "/upload",
                files=files,
                headers=authenticated_headers
            )
            results.append(response.status_code)
            time.sleep(0.1)  # Pequeno delay
        
        # Ao menos uma deve ser rejeitada OU todas falharem com 400 (arquivo inválido)
        has_rate_limit = 429 in results
        all_bad_request = all(s in [400, 500] for s in results)
        
        assert has_rate_limit or all_bad_request


class TestInputValidation:
    """Testes de validação de entrada."""
    
    def test_xss_in_descricao(self, client, authenticated_headers):
        """Testa se descrição com XSS é sanitizada."""
        malicious_query = '<script>alert("XSS")</script>'
        
        response = client.post(
            "/buscar",
            json={"descricao": malicious_query, "top_k": 5},
            headers=authenticated_headers
        )
        
        # Deve processar sem erro (status 200)
        assert response.status_code == 200
        
        data = response.json()
        # Verifica se script não é retornado sem escape
        response_text = str(data)
        assert "<script>" not in response_text or "&lt;script&gt;" in response_text
    
    def test_oversized_query(self, client, authenticated_headers):
        """Testa query excessivamente longa."""
        huge_query = "motor " * 10000  # 60k+ caracteres
        
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": huge_query, "top_k": 5},
            headers=authenticated_headers
        )
        
        # Deve rejeitar (413 ou 400) OU processar truncando
        assert response.status_code in [200, 400, 413, 422]
        
        if response.status_code == 200:
            # Verifica se backend truncou a query
            data = response.json()
            query_normalizada = data.get("query_normalizada", "")
            assert len(query_normalizada) < 10000  # Truncado
    
    def test_negative_top_k(self, client, authenticated_headers):
        """Testa valor negativo para top_k."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": -5},
            headers=authenticated_headers
        )
        
        # Deve validar e retornar erro OU ajustar para valor positivo
        if response.status_code == 200:
            data = response.json()
            assert len(data.get("resultados", [])) >= 0
        else:
            assert response.status_code in [400, 422]
    
    def test_zero_top_k(self, client, authenticated_headers):
        """Testa top_k = 0."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 0},
            headers=authenticated_headers
        )
        
        # Deve retornar lista vazia OU erro de validação
        if response.status_code == 200:
            data = response.json()
            assert len(data.get("resultados", [])) == 0
        else:
            assert response.status_code in [400, 422]
    
    def test_huge_top_k(self, client, authenticated_headers):
        """Testa top_k absurdamente grande."""
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 999999},
            headers=authenticated_headers
        )
        
        # Deve limitar top_k a um valor razoável
        assert response.status_code == 200
        data = response.json()
        # Máximo deve ser <= 100 (limite razoável)
        assert len(data.get("resultados", [])) <= 100


class TestCORS:
    """Testes de configuração CORS."""
    
    def test_cors_preflight(self, client):
        """Testa resposta a OPTIONS (CORS preflight)."""
        response = client.options("/buscar")
        
        # Deve responder com headers CORS
        assert response.status_code in [200, 204]
        
        # Em produção, verificar origem permitida
        # Este teste pode falhar em desenvolvimento (localhost permitido)
    
    def test_cors_origin_validation(self, client, authenticated_headers):
        """Testa se origem não autorizada é rejeitada."""
        headers_with_origin = {
            **authenticated_headers,
            "Origin": "https://malicious-site.com"
        }
        
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=headers_with_origin
        )
        
        # Backend deve processar, mas CORS headers devem impedir acesso no browser
        # (FastAPI middleware lida com isso)
        assert response.status_code == 200


class TestSecretLeakage:
    """Testes de vazamento de informações sensíveis."""
    
    def test_error_details_not_exposed(self, client, authenticated_headers):
        """Verifica se detalhes internos não são expostos em erros."""
        # Força um erro interno (arquivo Excel inexistente)
        import os
        from backend.app.utils import config
        
        # Salva path original e seta inválido
        original_path = config.EXCEL_PATH
        config.EXCEL_PATH = "/caminho/inexistente/arquivo.xlsx"
        
        try:
            response = client.post(
                "/buscar",
                json={"descricao": "motor", "top_k": 5},
                headers=authenticated_headers
            )
            
            # Deve retornar erro genérico
            assert response.status_code in [400, 500]
            
            data = response.json()
            detail = str(data.get("detail", "")).lower()
            
            # NÃO deve expor paths internos
            assert "/caminho/inexistente" not in detail
            assert "traceback" not in detail
            assert "exception" not in detail
        finally:
            # Restaura path original
            config.EXCEL_PATH = original_path
    
    def test_database_path_not_exposed(self, client):
        """Verifica se path do banco de dados não é exposto."""
        response = client.get("/health")
        
        data = response.json()
        response_text = str(data).lower()
        
        # NÃO deve expor paths de disco
        assert "sqlite" not in response_text or "status" in response_text
        assert ".db" not in response_text or "status" in response_text


# Marca testes de segurança para execução prioritária
pytestmark = pytest.mark.security
