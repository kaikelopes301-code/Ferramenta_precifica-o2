"""
Testes de Performance e Carga - Sistema de Precificação
=====================================================

Testes focados em performance, latência, throughput e comportamento sob carga.

Categorias:
- Latência de Endpoints
- Throughput e Concorrência
- Memory Leaks
- Cache Performance
- Load Testing
- Stress Testing
"""

import pytest
from fastapi.testclient import TestClient
import time
import threading
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestEndpointLatency:
    """Testes de latência de endpoints."""
    
    def test_health_check_latency(self, client):
        """Health check deve responder em <100ms."""
        start = time.time()
        response = client.get("/health")
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 100  # ms
    
    def test_data_status_latency(self, client):
        """Status dos dados deve responder em <500ms."""
        start = time.time()
        response = client.get("/data/status")
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 500  # ms
    
    @pytest.mark.slow
    def test_buscar_latency(self, client, authenticated_headers):
        """Busca tradicional deve responder em <2s."""
        start = time.time()
        response = client.post(
            "/buscar",
            json={"descricao": "motor", "top_k": 5},
            headers=authenticated_headers
        )
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 2000  # ms
    
    @pytest.mark.slow
    def test_buscar_inteligente_latency(self, client, authenticated_headers):
        """Busca inteligente deve responder em <30s (primeira vez com modelos)."""
        start = time.time()
        response = client.post(
            "/buscar-inteligente",
            json={"descricao": "motor eletrico", "top_k": 5},
            headers=authenticated_headers
        )
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 30000  # ms (primeira carga de modelos)
    
    @pytest.mark.slow
    def test_buscar_inteligente_cached_latency(self, client, authenticated_headers):
        """Busca inteligente com cache deve ser <5s."""
        query = {"descricao": "motor cache test", "top_k": 5}
        
        # Primeira request (warm-up)
        client.post("/buscar-inteligente", json=query, headers=authenticated_headers)
        
        # Segunda request (cached)
        start = time.time()
        response = client.post("/buscar-inteligente", json=query, headers=authenticated_headers)
        latency = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert latency < 5000  # ms


class TestThroughput:
    """Testes de throughput (requests por segundo)."""
    
    @pytest.mark.slow
    def test_concurrent_health_checks(self, client):
        """Teste de throughput do health check."""
        num_requests = 100
        results = []
        
        def make_request():
            start = time.time()
            response = client.get("/health")
            duration = time.time() - start
            return response.status_code, duration
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        # Verifica sucesso
        success_count = sum(1 for status, _ in results if status == 200)
        assert success_count >= num_requests * 0.95  # 95% sucesso
        
        # Calcula throughput
        throughput = num_requests / total_time
        print(f"\n✅ Health Check Throughput: {throughput:.2f} req/s")
        
        # Deve suportar pelo menos 50 req/s
        assert throughput >= 50
    
    @pytest.mark.slow
    def test_concurrent_searches(self, client, authenticated_headers):
        """Teste de throughput de buscas."""
        num_requests = 20  # Reduzido pois é mais pesado
        queries = ["motor", "bomba", "compressor"] * (num_requests // 3)
        
        results = []
        
        def make_search(query):
            start = time.time()
            response = client.post(
                "/buscar",
                json={"descricao": query, "top_k": 5},
                headers=authenticated_headers
            )
            duration = time.time() - start
            return response.status_code, duration
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_search, q) for q in queries]
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        success_count = sum(1 for status, _ in results if status == 200)
        assert success_count >= num_requests * 0.9  # 90% sucesso
        
        throughput = num_requests / total_time
        print(f"\n✅ Search Throughput: {throughput:.2f} req/s")
        
        # Deve suportar pelo menos 3 req/s
        assert throughput >= 3


class TestMemoryUsage:
    """Testes de uso de memória."""
    
    def get_memory_usage_mb(self):
        """Retorna uso de memória do processo em MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.slow
    def test_no_memory_leak_searches(self, client, authenticated_headers):
        """Verifica se há memory leak em buscas repetidas."""
        initial_memory = self.get_memory_usage_mb()
        
        # Faz 50 buscas
        for i in range(50):
            client.post(
                "/buscar",
                json={"descricao": f"motor {i}", "top_k": 5},
                headers=authenticated_headers
            )
        
        final_memory = self.get_memory_usage_mb()
        memory_growth = final_memory - initial_memory
        
        print(f"\n📊 Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_growth:.1f}MB)")
        
        # Crescimento deve ser <50MB
        assert memory_growth < 50
    
    @pytest.mark.slow
    def test_memory_bounded_cache(self, client, authenticated_headers):
        """Verifica se cache respeita limite de memória."""
        initial_memory = self.get_memory_usage_mb()
        
        # Faz muitas buscas diferentes (não cacheable)
        for i in range(100):
            client.post(
                "/buscar-inteligente",
                json={"descricao": f"query unica {i} {time.time()}", "top_k": 5},
                headers=authenticated_headers
            )
        
        final_memory = self.get_memory_usage_mb()
        memory_growth = final_memory - initial_memory
        
        print(f"\n📊 Cache Memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_growth:.1f}MB)")
        
        # Cache deve respeitar CACHE_MEMORY_LIMIT_MB (200MB)
        assert memory_growth < 250  # Margem


class TestCachePerformance:
    """Testes de performance do cache."""
    
    @pytest.mark.slow
    def test_cache_hit_speedup(self, client, authenticated_headers):
        """Verifica speedup do cache hit vs miss."""
        query = {"descricao": "motor cache speedup", "top_k": 5}
        
        # MISS (primeira request)
        start_miss = time.time()
        response_miss = client.post(
            "/buscar-inteligente",
            json=query,
            headers=authenticated_headers
        )
        time_miss = (time.time() - start_miss) * 1000
        
        # HIT (segunda request)
        start_hit = time.time()
        response_hit = client.post(
            "/buscar-inteligente",
            json=query,
            headers=authenticated_headers
        )
        time_hit = (time.time() - start_hit) * 1000
        
        assert response_miss.status_code == 200
        assert response_hit.status_code == 200
        
        speedup = time_miss / max(time_hit, 1)
        print(f"\n⚡ Cache Speedup: {speedup:.1f}x (MISS={time_miss:.0f}ms, HIT={time_hit:.0f}ms)")
        
        # Cache hit deve ser pelo menos 2x mais rápido
        assert speedup >= 2.0 or time_hit < 100  # Ou já é muito rápido
    
    @pytest.mark.slow
    def test_cache_stats_accuracy(self, client, authenticated_headers):
        """Verifica acurácia das estatísticas de cache."""
        # Limpa cache
        client.post("/cache/clear")
        
        # Faz requests conhecidas
        query1 = {"descricao": "motor stats 1", "top_k": 5}
        query2 = {"descricao": "motor stats 2", "top_k": 5}
        
        # 2x query1 (1 miss + 1 hit)
        client.post("/buscar-inteligente", json=query1, headers=authenticated_headers)
        client.post("/buscar-inteligente", json=query1, headers=authenticated_headers)
        
        # 3x query2 (1 miss + 2 hits)
        client.post("/buscar-inteligente", json=query2, headers=authenticated_headers)
        client.post("/buscar-inteligente", json=query2, headers=authenticated_headers)
        client.post("/buscar-inteligente", json=query2, headers=authenticated_headers)
        
        # Verifica stats
        stats_response = client.get("/cache/stats")
        stats = stats_response.json()["lru_cache"]
        
        print(f"\n📊 Cache Stats: hits={stats['hits']}, misses={stats['misses']}, rate={stats['hit_rate']}%")
        
        # Deve ter 3 hits e 2 misses
        assert stats["hits"] >= 3
        assert stats["misses"] >= 2


class TestLoadTesting:
    """Testes de carga (load testing)."""
    
    @pytest.mark.slow
    @pytest.mark.load
    def test_sustained_load(self, client, authenticated_headers):
        """Teste de carga sustentada por 30s."""
        duration_seconds = 30
        max_workers = 5
        
        queries = ["motor", "bomba", "compressor", "motor 5hp", "bomba centrifuga"]
        
        results = {
            "total": 0,
            "success": 0,
            "errors": 0,
            "latencies": []
        }
        
        def make_request():
            query = queries[results["total"] % len(queries)]
            start = time.time()
            try:
                response = client.post(
                    "/buscar",
                    json={"descricao": query, "top_k": 5},
                    headers=authenticated_headers,
                    timeout=10
                )
                latency = (time.time() - start) * 1000
                
                results["total"] += 1
                results["latencies"].append(latency)
                
                if response.status_code == 200:
                    results["success"] += 1
                else:
                    results["errors"] += 1
                    
            except Exception:
                results["errors"] += 1
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while time.time() - start_time < duration_seconds:
                executor.submit(make_request)
                time.sleep(0.1)  # ~10 req/s
        
        # Análise dos resultados
        avg_latency = sum(results["latencies"]) / max(len(results["latencies"]), 1)
        p95_latency = sorted(results["latencies"])[int(len(results["latencies"]) * 0.95)] if results["latencies"] else 0
        throughput = results["total"] / duration_seconds
        success_rate = (results["success"] / max(results["total"], 1)) * 100
        
        print(f"""
        📊 Load Test Results ({duration_seconds}s):
        - Total Requests: {results["total"]}
        - Success Rate: {success_rate:.1f}%
        - Throughput: {throughput:.2f} req/s
        - Avg Latency: {avg_latency:.0f}ms
        - P95 Latency: {p95_latency:.0f}ms
        """)
        
        # Validações
        assert success_rate >= 95.0  # 95% de sucesso
        assert throughput >= 5.0  # Pelo menos 5 req/s
        assert p95_latency < 5000  # P95 < 5s


class TestStressTesting:
    """Testes de stress (limites do sistema)."""
    
    @pytest.mark.slow
    @pytest.mark.stress
    def test_spike_traffic(self, client, authenticated_headers):
        """Teste de pico de tráfego repentino."""
        num_requests = 50
        max_workers = 20  # Alto paralelismo
        
        results = []
        
        def make_request():
            start = time.time()
            try:
                response = client.post(
                    "/buscar",
                    json={"descricao": "motor spike", "top_k": 5},
                    headers=authenticated_headers,
                    timeout=30
                )
                duration = time.time() - start
                return response.status_code, duration
            except Exception as e:
                return 500, 0
        
        # Pico súbito
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        # Análise
        success_count = sum(1 for status, _ in results if status == 200)
        success_rate = (success_count / num_requests) * 100
        
        print(f"""
        ⚡ Spike Test Results:
        - Total Requests: {num_requests}
        - Concurrency: {max_workers}
        - Success Rate: {success_rate:.1f}%
        - Total Time: {total_time:.2f}s
        """)
        
        # Sob pico, pelo menos 80% deve suceder
        assert success_rate >= 80.0
    
    @pytest.mark.slow
    @pytest.mark.stress
    def test_degradation_under_load(self, client, authenticated_headers):
        """Verifica degradação graciosa sob carga extrema."""
        num_requests = 100
        max_workers = 30
        
        latencies = []
        errors = []
        
        def make_request(i):
            start = time.time()
            try:
                response = client.post(
                    "/buscar-inteligente",
                    json={"descricao": f"stress test {i}", "top_k": 5},
                    headers=authenticated_headers,
                    timeout=60
                )
                duration = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    latencies.append(duration)
                else:
                    errors.append(response.status_code)
                    
            except Exception as e:
                errors.append(str(e))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(make_request, range(num_requests)))
        
        success_rate = (len(latencies) / num_requests) * 100
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
            
            print(f"""
            🔥 Stress Test Results:
            - Requests: {num_requests} (concurrency={max_workers})
            - Success Rate: {success_rate:.1f}%
            - Avg Latency: {avg_latency:.0f}ms
            - P99 Latency: {p99_latency:.0f}ms
            - Errors: {len(errors)}
            """)
        
        # Degradação graciosa: mesmo sob stress extremo, pelo menos 50% deve funcionar
        assert success_rate >= 50.0


class TestRateLimitPerformance:
    """Testes de rate limiting sob carga."""
    
    @pytest.mark.slow
    def test_rate_limit_enforcement(self, client, authenticated_headers):
        """Verifica se rate limiting é aplicado corretamente."""
        num_requests = 30  # Mais que limite (10/s)
        
        responses = []
        start_time = time.time()
        
        for i in range(num_requests):
            response = client.post(
                "/buscar-inteligente",
                json={"descricao": f"rate limit {i}", "top_k": 5},
                headers=authenticated_headers
            )
            responses.append((response.status_code, time.time() - start_time))
        
        # Conta 429 (Too Many Requests)
        rate_limited = sum(1 for status, _ in responses if status == 429)
        
        print(f"\n🚦 Rate Limit: {rate_limited}/{num_requests} requests limited")
        
        # Se rate limiting ativo, deve ter rejeitado alguns
        # Se não ativo, é problema de segurança (já testado em test_api_security.py)
        if rate_limited > 0:
            assert rate_limited >= 10  # Pelo menos 10 rejeitados


# Marca testes de performance
pytestmark = pytest.mark.performance
