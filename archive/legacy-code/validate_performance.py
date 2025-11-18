#!/usr/bin/env python3
"""
Script para validação completa de performance do backend otimizado
"""
import time
import os
import sys
import json
import requests
import subprocess
import psutil
from datetime import datetime

# Adicionar o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class PerformanceValidator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "startup_time": None,
            "memory_usage": {},
            "search_performance": {},
            "baseline_comparison": {}
        }
        
    def measure_startup_time(self):
        """Medir tempo de startup do backend"""
        print("🚀 Testando tempo de startup...")
        
        # Usar backend path
        backend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        start_time = time.time()
        try:
            from app.api.main import app
            startup_time = time.time() - start_time
            
            self.results["startup_time"] = {
                "seconds": round(startup_time, 2),
                "status": "✅ SUCESSO" if startup_time < 3.0 else "⚠️ ACIMA DO TARGET"
            }
            
            print(f"   Tempo de startup: {startup_time:.2f}s")
            return startup_time
        except Exception as e:
            print(f"   ❌ Erro no startup: {e}")
            self.results["startup_time"] = {"error": str(e)}
            return None
    
    def measure_memory_usage(self):
        """Medir uso de memória"""
        print("\n💾 Analisando uso de memória...")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        self.results["memory_usage"] = {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
            "percent": round(process.memory_percent(), 2)
        }
        
        print(f"   Memória RSS: {self.results['memory_usage']['rss_mb']} MB")
        print(f"   Percentual: {self.results['memory_usage']['percent']}%")
        
    def start_test_server(self):
        """Iniciar servidor de teste"""
        print("\n🌐 Iniciando servidor de teste...")
        
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "app.api.main:app", 
            "--host", "127.0.0.1", 
            "--port", "8001"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = "backend"
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env,
            cwd=os.getcwd()
        )
        
        # Aguardar servidor inicializar
        time.sleep(3)
        
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Servidor iniciado com sucesso")
                return process
        except:
            pass
        
        print("   ⚠️ Servidor não respondeu, tentando novamente...")
        time.sleep(2)
        
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Servidor inicializado")
                return process
        except:
            print("   ❌ Falha ao inicializar servidor")
            process.terminate()
            return None
    
    def test_search_performance(self, server_process):
        """Testar performance da busca inteligente"""
        print("\n🔍 Testando busca inteligente...")
        
        test_queries = [
            "mop industrial",
            "equipamento limpeza",
            "ferramenta jardinagem",
            "produto químico",
            "material escritório"
        ]
        
        search_results = []
        
        for query in test_queries:
            try:
                start_time = time.time()
                
                response = requests.post(
                    "http://127.0.0.1:8001/buscar-inteligente",
                    json={"descricao": query, "top_k": 5},
                    timeout=10
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    result = {
                        "query": query,
                        "response_time_ms": round(response_time * 1000, 2),
                        "results_count": len(data.get("resultados", [])),
                        "status": "success"
                    }
                    
                    if data.get("resultados"):
                        result["top_score"] = data["resultados"][0].get("score", 0)
                    
                    search_results.append(result)
                    print(f"   {query}: {response_time*1000:.0f}ms ({len(data.get('resultados', []))} resultados)")
                
            except Exception as e:
                search_results.append({
                    "query": query,
                    "error": str(e),
                    "status": "error"
                })
                print(f"   {query}: ❌ Erro - {e}")
        
        self.results["search_performance"] = {
            "tests": search_results,
            "avg_response_time_ms": round(
                sum(r.get("response_time_ms", 0) for r in search_results if r.get("response_time_ms")) 
                / max(1, len([r for r in search_results if r.get("response_time_ms")])), 2
            ),
            "success_rate": round(
                len([r for r in search_results if r.get("status") == "success"]) / len(search_results) * 100, 1
            )
        }
        
        print(f"   Tempo médio: {self.results['search_performance']['avg_response_time_ms']}ms")
        print(f"   Taxa de sucesso: {self.results['search_performance']['success_rate']}%")
    
    def compare_with_baseline(self):
        """Comparar com baseline anterior"""
        print("\n📊 Comparando com baseline...")
        
        baseline_file = "performance_baseline.json"
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)
                
                if self.results["startup_time"] and baseline.get("startup_time"):
                    old_startup = baseline["startup_time"].get("seconds", 0)
                    new_startup = self.results["startup_time"]["seconds"]
                    improvement = ((old_startup - new_startup) / old_startup * 100) if old_startup > 0 else 0
                    
                    self.results["baseline_comparison"]["startup_improvement"] = {
                        "old_seconds": old_startup,
                        "new_seconds": new_startup,
                        "improvement_percent": round(improvement, 1)
                    }
                    
                    print(f"   Startup: {old_startup}s → {new_startup}s ({improvement:+.1f}%)")
            
            except Exception as e:
                print(f"   ⚠️ Erro ao carregar baseline: {e}")
        else:
            print("   ℹ️ Nenhum baseline anterior encontrado")
    
    def generate_report(self):
        """Gerar relatório final"""
        print("\n📋 Gerando relatório...")
        
        # Salvar resultados
        with open("performance_validation.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Imprimir resumo
        print("\n" + "="*50)
        print("🎯 RESUMO DA VALIDAÇÃO DE PERFORMANCE")
        print("="*50)
        
        if self.results["startup_time"] and "seconds" in self.results["startup_time"]:
            startup = self.results["startup_time"]["seconds"]
            target_met = "✅" if startup < 3.0 else "❌"
            print(f"Tempo de Startup: {startup}s {target_met}")
        elif self.results["startup_time"] and "error" in self.results["startup_time"]:
            print(f"Startup: ❌ Erro - {self.results['startup_time']['error']}")
        
        if self.results["memory_usage"]:
            mem = self.results["memory_usage"]["rss_mb"]
            print(f"Uso de Memória: {mem} MB")
        
        if self.results["search_performance"]:
            avg_time = self.results["search_performance"]["avg_response_time_ms"]
            success_rate = self.results["search_performance"]["success_rate"]
            print(f"Busca Média: {avg_time}ms ({success_rate}% sucesso)")
        
        if self.results["baseline_comparison"].get("startup_improvement"):
            improvement = self.results["baseline_comparison"]["startup_improvement"]["improvement_percent"]
            print(f"Melhoria: {improvement:+.1f}% no startup")
        
        print("\n📁 Relatório salvo em: performance_validation.json")
    
    def run_full_validation(self):
        """Executar validação completa"""
        print("🧪 INICIANDO VALIDAÇÃO COMPLETA DE PERFORMANCE")
        print("="*60)
        
        # 1. Medir startup
        self.measure_startup_time()
        
        # 2. Medir memória
        self.measure_memory_usage()
        
        # 3. Testar servidor
        server = self.start_test_server()
        if server:
            try:
                # 4. Testar busca
                self.test_search_performance(server)
            finally:
                # Finalizar servidor
                print("\n🛑 Finalizando servidor de teste...")
                server.terminate()
                server.wait()
        
        # 5. Comparar com baseline
        self.compare_with_baseline()
        
        # 6. Gerar relatório
        self.generate_report()

if __name__ == "__main__":
    validator = PerformanceValidator()
    validator.run_full_validation()