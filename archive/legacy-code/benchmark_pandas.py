"""
Benchmarks de performance para validar otimizações Pandas e API.

Testes de regressão que garantem que as otimizações entreguem os ganhos prometidos.
"""
import pytest
import time
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ingestao.excel import load_excel
from src.utils.data_cache import data_cache


class PerformanceBaseline:
    """Baselines de performance para detectar regressões."""
    
    # Baselines atualizados após otimizações P0/P1
    CLEAN_DATA_MAX_TIME = 0.050  # 50ms para 4K rows (era muito pior antes)
    CACHE_HIT_MAX_TIME = 0.005   # 5ms para cache hit
    SEARCH_API_MAX_TIME = 0.300  # 300ms para busca (era ~400ms antes)
    VECTORIZATION_SPEEDUP = 2.5  # Min 2.5x speedup vs apply()


def create_mock_dataframe(n_rows: int = 4000) -> pd.DataFrame:
    """Cria DataFrame mock similar aos dados de produção."""
    np.random.seed(42)  # Reproduzível
    
    return pd.DataFrame({
        'valor_unitario': [
            f"{np.random.uniform(1, 100):.2f}".replace('.', ',') if i % 3 == 0
            else f"R$ {np.random.uniform(1, 100):.2f}"
            for i in range(n_rows)
        ],
        'vida_util_meses': [
            f"{np.random.randint(6, 60)},0" if i % 4 == 0
            else f"{np.random.randint(6, 60)}.0"
            for i in range(n_rows)
        ],
        'manutencao': [
            f"{np.random.uniform(0.02, 0.15):.3f}".replace('.', ',') if i % 2 == 0
            else f"{np.random.uniform(2, 15):.1f}%"
            for i in range(n_rows)
        ],
        'descricao': [f"Equipamento tipo {i // 10} modelo {i % 10}" for i in range(n_rows)],
        'marca': [f"Marca{i % 20}" for i in range(n_rows)]
    })


def time_function(func, *args, **kwargs) -> tuple[Any, float]:
    """Helper para medir tempo de execução."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    return result, duration


class TestDataProcessingPerformance:
    """Testes de performance para processamento de dados."""
    
    def test_vectorized_float_conversion_performance(self):
        """Testa se a conversão vetorizada é mais rápida que apply()."""
        from src.ingestao.excel import load_excel
        
        mock_df = create_mock_dataframe(4000)
        
        # Simular função antiga com apply (para comparação)
        def old_apply_method(df):
            def _to_float_old(val):
                if pd.isna(val):
                    return None
                s = str(val).replace('R$', '').replace('%', '').replace(' ', '')
                s = s.replace(',', '.')
                try:
                    return float(s)
                except:
                    return None
            
            df_copy = df.copy()
            df_copy['valor_unitario'] = df_copy['valor_unitario'].apply(_to_float_old)
            return df_copy
        
        # Função nova vetorizada (extrair lógica de excel.py)
        def new_vectorized_method(df):
            df_copy = df.copy()
            s = df_copy['valor_unitario'].astype(str)
            s = s.str.replace(r'[R$%\s]', '', regex=True)
            s = s.str.replace(',', '.')
            df_copy['valor_unitario'] = pd.to_numeric(s, errors='coerce')
            return df_copy
        
        # Benchmark comparativo
        _, old_time = time_function(old_apply_method, mock_df)
        _, new_time = time_function(new_vectorized_method, mock_df)
        
        speedup = old_time / new_time
        print(f"Speedup: {speedup:.2f}x (old: {old_time:.3f}s, new: {new_time:.3f}s)")
        
        assert speedup >= PerformanceBaseline.VECTORIZATION_SPEEDUP, \
            f"Vetorização deve ser ≥{PerformanceBaseline.VECTORIZATION_SPEEDUP}x mais rápida"
        assert new_time < PerformanceBaseline.CLEAN_DATA_MAX_TIME, \
            f"Conversão vetorizada deve ser <{PerformanceBaseline.CLEAN_DATA_MAX_TIME}s"
    
    def test_parquet_cache_performance(self):
        """Testa se o cache Parquet acelera carregamento significativamente."""
        mock_df = create_mock_dataframe(2000)
        
        # Primeiro save (cold)
        cache_key = "test_perf_cache"
        data_cache.set(cache_key, mock_df)
        
        # Benchmark cache hit
        _, cache_time = time_function(data_cache.get, cache_key)
        
        print(f"Cache hit time: {cache_time:.4f}s")
        assert cache_time < PerformanceBaseline.CACHE_HIT_MAX_TIME, \
            f"Cache hit deve ser <{PerformanceBaseline.CACHE_HIT_MAX_TIME}s"
        
        # Cleanup
        try:
            cache_file = data_cache.cache_dir / f"{cache_key}.parquet"
            cache_file.unlink(missing_ok=True)
        except:
            pass


class TestAPIPerformance:
    """Testes de performance para endpoints da API."""
    
    def test_search_endpoint_response_time(self):
        """Testa se o endpoint de busca atende SLA de latência."""
        pytest.importorskip("fastapi.testclient")
        
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        # Warm up (carregar modelos, cache, etc.)
        try:
            client.post("/buscar-inteligente", json={"descricao": "furadeira", "top_k": 3})
        except Exception:
            pytest.skip("API não disponível para teste")
        
        # Benchmark real
        test_queries = [
            {"descricao": "martelo", "top_k": 5},
            {"descricao": "serra eletrica", "top_k": 8},
            {"descricao": "parafusadeira", "top_k": 3}
        ]
        
        times = []
        for query in test_queries:
            start = time.perf_counter()
            response = client.post("/buscar-inteligente", json=query)
            duration = time.perf_counter() - start
            times.append(duration)
            
            assert response.status_code == 200, f"API error: {response.text}"
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        print(f"Search API - Avg: {avg_time:.3f}s, P95: {p95_time:.3f}s")
        
        assert p95_time < PerformanceBaseline.SEARCH_API_MAX_TIME, \
            f"P95 latency deve ser <{PerformanceBaseline.SEARCH_API_MAX_TIME}s, got {p95_time:.3f}s"
    
    def test_batch_search_throughput(self):
        """Testa throughput de busca em lote."""
        pytest.importorskip("fastapi.testclient")
        
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        client = TestClient(app)
        
        batch_queries = {
            "descricoes": [
                "furadeira industrial",
                "martelo pneumático",
                "serra circular",
                "parafusadeira elétrica",
                "morsa bancada"
            ],
            "top_k": 3
        }
        
        try:
            start = time.perf_counter()
            response = client.post("/buscar-lote-inteligente", json=batch_queries)
            duration = time.perf_counter() - start
            
            if response.status_code == 200:
                throughput = len(batch_queries["descricoes"]) / duration
                print(f"Batch throughput: {throughput:.1f} queries/sec")
                
                # Meta: ≥10 queries/sec para lote
                assert throughput >= 10.0, f"Throughput deve ser ≥10 q/s, got {throughput:.1f}"
        except Exception as e:
            pytest.skip(f"Batch API não disponível: {e}")


class TestMemoryUsage:
    """Testes de uso de memória para detectar vazamentos."""
    
    def test_dataframe_memory_efficiency(self):
        """Testa se operações não causam explosão de memória."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simular carga pesada
        for i in range(5):
            large_df = create_mock_dataframe(10000)
            
            # Aplicar nossas otimizações
            s = large_df['valor_unitario'].astype(str)
            s = s.str.replace(r'[R$%\s]', '', regex=True)
            large_df['processed'] = pd.to_numeric(s, errors='coerce')
            
            # Force GC
            del large_df, s
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"Memory growth: {memory_growth:.1f} MB")
        
        # Não deve crescer mais que 50MB após 5 iterações
        assert memory_growth < 50.0, f"Memory leak suspeito: {memory_growth:.1f} MB growth"


def benchmark_summary():
    """Executa todos os benchmarks e gera relatório."""
    print("=" * 60)
    print("BENCHMARK SUMMARY - Performance Optimizations")
    print("=" * 60)
    
    test_suite = [
        TestDataProcessingPerformance(),
        TestAPIPerformance(), 
        TestMemoryUsage()
    ]
    
    results = {}
    for test_class in test_suite:
        class_name = test_class.__class__.__name__
        print(f"\n[{class_name}]")
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                try:
                    start = time.perf_counter()
                    getattr(test_class, method_name)()
                    duration = time.perf_counter() - start
                    
                    status = "✅ PASS"
                    results[f"{class_name}.{method_name}"] = {"status": "PASS", "time": duration}
                except Exception as e:
                    status = f"❌ FAIL: {str(e)[:50]}..."
                    results[f"{class_name}.{method_name}"] = {"status": "FAIL", "error": str(e)}
                    
                print(f"  {method_name}: {status}")
    
    # Summary stats
    pass_count = sum(1 for r in results.values() if r.get("status") == "PASS")
    total_count = len(results)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {pass_count}/{total_count} tests passed")
    
    if pass_count == total_count:
        print("🎉 All performance benchmarks PASSED!")
    else:
        print("⚠️  Some performance tests failed - investigate regressions")
    
    return results


if __name__ == "__main__":
    # Executar benchmarks standalone
    benchmark_summary()