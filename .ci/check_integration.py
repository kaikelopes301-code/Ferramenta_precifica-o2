#!/usr/bin/env python3
"""
🧪 Integration Healthcheck Script
Next.js 15.x + FastAPI Enterprise Integration Test
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor

class IntegrationTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.results = {}
        
    def test_backend_direct(self) -> Dict[str, Any]:
        """Testa FastAPI diretamente"""
        print("🔄 Testing FastAPI direct...")
        
        try:
            # Health check
            start = time.time()
            response = requests.get(f"{self.backend_url}/api/data/status", timeout=5)
            latency = (time.time() - start) * 1000
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "response": response.json() if response.status_code == 200 else None
            }
            
            if response.status_code == 200:
                print(f"✅ Backend direct: {latency:.1f}ms")
            else:
                print(f"❌ Backend direct: HTTP {response.status_code}")
                
            return result
            
        except Exception as e:
            print(f"❌ Backend direct failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_smart_search_direct(self) -> Dict[str, Any]:
        """Testa busca inteligente diretamente no backend"""
        print("🔄 Testing Smart Search direct...")
        
        try:
            start = time.time()
            response = requests.post(
                f"{self.backend_url}/buscar-inteligente",
                json={"descricao": "mop industrial", "top_k": 5},
                timeout=10
            )
            latency = (time.time() - start) * 1000
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
            }
            
            if response.status_code == 200:
                data = response.json()
                result["results_count"] = len(data.get("resultados", []))
                print(f"✅ Smart Search direct: {latency:.1f}ms, {result['results_count']} results")
            else:
                print(f"❌ Smart Search direct: HTTP {response.status_code}")
                
            return result
            
        except Exception as e:
            print(f"❌ Smart Search direct failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_frontend_proxy(self) -> Dict[str, Any]:
        """Testa proxy do Next.js (/backend/*)"""
        print("🔄 Testing Next.js proxy...")
        
        try:
            # Via proxy
            start = time.time()
            response = requests.get(f"{self.frontend_url}/backend/api/data/status", timeout=10)
            latency = (time.time() - start) * 1000
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "response": response.json() if response.status_code == 200 else None
            }
            
            if response.status_code == 200:
                print(f"✅ Frontend proxy: {latency:.1f}ms")
            else:
                print(f"❌ Frontend proxy: HTTP {response.status_code}")
                
            return result
            
        except Exception as e:
            print(f"❌ Frontend proxy failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def test_api_route(self) -> Dict[str, Any]:
        """Testa route handler do Next.js"""
        print("🔄 Testing Next.js API route...")
        
        try:
            start = time.time()
            response = requests.post(
                f"{self.frontend_url}/api/smart-search",
                json={"descricao": "mop industrial", "top_k": 5},
                timeout=15
            )
            latency = (time.time() - start) * 1000
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
            }
            
            if response.status_code == 200:
                data = response.json()
                result["results_count"] = len(data.get("resultados", []))
                print(f"✅ API Route: {latency:.1f}ms, {result['results_count']} results")
            else:
                print(f"❌ API Route: HTTP {response.status_code}")
                
            return result
            
        except Exception as e:
            print(f"❌ API Route failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os testes"""
        print("🧪 INTEGRATION HEALTHCHECK")
        print("=" * 50)
        
        # Execução paralela dos testes independentes
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "backend_health": executor.submit(self.test_backend_direct),
                "backend_search": executor.submit(self.test_smart_search_direct),
                "frontend_proxy": executor.submit(self.test_frontend_proxy),
                "api_route": executor.submit(self.test_api_route),
            }
            
            results = {}
            for test_name, future in futures.items():
                try:
                    results[test_name] = future.result(timeout=20)
                except Exception as e:
                    results[test_name] = {"status": "error", "error": f"Test timeout: {e}"}
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera relatório final"""
        
        # Contar sucessos
        successful_tests = sum(1 for r in results.values() if r.get("status") == "success")
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100
        
        # Calcular latências médias
        latencies = [r.get("latency_ms", 0) for r in results.values() if r.get("latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_rate": round(success_rate, 1),
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "avg_latency_ms": round(avg_latency, 2),
            "tests": results,
            "status": "PASS" if successful_tests >= 3 else "FAIL"  # Pelo menos 3/4 devem passar
        }
        
        return report

def main():
    """Função principal"""
    tester = IntegrationTester()
    
    # Executar testes
    results = tester.run_all_tests()
    
    # Gerar relatório
    report = tester.generate_report(results)
    
    # Exibir resultados
    print("\n" + "="*50)
    print("📊 INTEGRATION REPORT")
    print("="*50)
    print(f"Status: {report['status']}")
    print(f"Success Rate: {report['success_rate']}%")
    print(f"Average Latency: {report['avg_latency_ms']}ms")
    print(f"Tests Passed: {report['successful_tests']}/{report['total_tests']}")
    
    # Detalhes dos testes que falharam
    failed_tests = [name for name, result in results.items() if result.get("status") != "success"]
    if failed_tests:
        print(f"\n❌ Failed Tests: {', '.join(failed_tests)}")
    
    # Salvar relatório
    report_file = Path("integration_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Full report saved to: {report_file}")
    
    # Exit code para CI/CD
    exit_code = 0 if report['status'] == 'PASS' else 1
    
    # Verificações específicas para CI
    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        # No CI, todos os testes devem passar
        if report['success_rate'] < 100:
            print(f"\n🚨 CI FAILURE: Success rate {report['success_rate']}% < 100%")
            exit_code = 1
        
        # Verificar latência
        if report['avg_latency_ms'] > 1000:  # 1 segundo
            print(f"\n🚨 CI FAILURE: Average latency {report['avg_latency_ms']}ms > 1000ms")
            exit_code = 1
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()