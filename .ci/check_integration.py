#!/usr/bin/env python3
"""
🧪 Integration Healthcheck Script
Next.js 15.x + FastAPI Enterprise Integration Test
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

import requests

# Limites de latência para CI (em milissegundos)
MAX_AVG_LATENCY_MS = 3000        # média das rotas críticas (proxy + api_route)
MAX_SINGLE_LATENCY_MS = 5000     # nenhuma chamada crítica pode passar disso

# Testes que são obrigatórios para a CI (caminho real de usuário)
REQUIRED_TESTS_FOR_CI = ["frontend_proxy", "api_route"]


class IntegrationTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"

    def test_backend_direct(self) -> Dict[str, Any]:
        """Testa FastAPI diretamente"""
        print("🔄 Testing FastAPI direct...")

        try:
            start = time.time()
            response = requests.get(f"{self.backend_url}/data/status", timeout=20)
            latency = (time.time() - start) * 1000

            result: Dict[str, Any] = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "response": response.json() if response.status_code == 200 else None,
            }

            if response.status_code == 200:
                print(f"✅ Backend direct: {latency:.1f}ms")
            else:
                print(f"❌ Backend direct: HTTP {response.status_code}")

            return result

        except Exception as e:  # noqa: BLE001
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
                timeout=30,
            )
            latency = (time.time() - start) * 1000

            result: Dict[str, Any] = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
            }

            if response.status_code == 200:
                data = response.json()
                result["results_count"] = len(data.get("resultados", []))
                print(
                    f"✅ Smart Search direct: {latency:.1f}ms, "
                    f"{result['results_count']} results"
                )
            else:
                print(f"❌ Smart Search direct: HTTP {response.status_code}")

            return result

        except Exception as e:  # noqa: BLE001
            print(f"❌ Smart Search direct failed: {e}")
            return {"status": "error", "error": str(e)}

    def test_frontend_proxy(self) -> Dict[str, Any]:
        """Testa proxy do Next.js (/backend/*)"""
        print("🔄 Testing Next.js proxy...")

        try:
            start = time.time()
            response = requests.get(f"{self.frontend_url}/backend/data/status", timeout=10)
            latency = (time.time() - start) * 1000

            result: Dict[str, Any] = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "response": response.json() if response.status_code == 200 else None,
            }

            if response.status_code == 200:
                print(f"✅ Frontend proxy: {latency:.1f}ms")
            else:
                print(f"❌ Frontend proxy: HTTP {response.status_code}")

            return result

        except Exception as e:  # noqa: BLE001
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
                timeout=15,
            )
            latency = (time.time() - start) * 1000

            result: Dict[str, Any] = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
            }

            if response.status_code == 200:
                data = response.json()
                result["results_count"] = len(data.get("resultados", []))
                print(
                    f"✅ API Route: {latency:.1f}ms, "
                    f"{result['results_count']} results"
                )
            else:
                print(f"❌ API Route: HTTP {response.status_code}")

            return result

        except Exception as e:  # noqa: BLE001
            print(f"❌ API Route failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os testes"""
        print("🧪 INTEGRATION HEALTHCHECK")
        print("=" * 50)

        # Warm-up explícito da pipeline pesada (não entra no relatório)
        print("🔥 Warming up Smart Search pipeline (not counted in report)...")
        try:
            _ = self.test_smart_search_direct()
        except Exception as e:  # noqa: BLE001
            # Se o warm-up falhar, ainda assim seguimos com os testes formais
            print(f"⚠️ Warm-up failed, continuing tests anyway: {e}")

        # Agora executa os testes "oficiais"
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "backend_health": executor.submit(self.test_backend_direct),
                "backend_search": executor.submit(self.test_smart_search_direct),
                "frontend_proxy": executor.submit(self.test_frontend_proxy),
                "api_route": executor.submit(self.test_api_route),
            }

            results: Dict[str, Any] = {}
            for test_name, future in futures.items():
                try:
                    results[test_name] = future.result(timeout=35)
                except Exception as e:  # noqa: BLE001
                    results[test_name] = {
                        "status": "error",
                        "error": f"Test timeout: {e}",
                    }

        return results


    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera relatório final"""

        successful_tests = sum(
            1 for r in results.values() if r.get("status") == "success"
        )
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests else 0.0

        latencies = [
            r.get("latency_ms", 0)
            for r in results.values()
            if r.get("status") == "success" and r.get("latency_ms") is not None
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_rate": round(success_rate, 1),
            "successful_tests": successful_tests,
            "total_tests": total_tests,
            "avg_latency_ms": round(avg_latency, 2),
            "tests": results,
            "status": "PASS" if successful_tests >= 3 else "FAIL",  # Pelo menos 3/4
        }

        return report


def main() -> None:
    tester = IntegrationTester()

    results = tester.run_all_tests()
    report = tester.generate_report(results)

    print("\n" + "=" * 50)
    print("📊 INTEGRATION REPORT")
    print("=" * 50)
    print(f"Status: {report['status']}")
    print(f"Success Rate: {report['success_rate']}%")
    print(f"Average Latency: {report['avg_latency_ms']}ms")
    print(
        f"Tests Passed: {report['successful_tests']}/"
        f"{report['total_tests']}"
    )

    failed_tests = [
        name for name, result in results.items()
        if result.get("status") != "success"
    ]
    if failed_tests:
        print(f"\n❌ Failed Tests: {', '.join(failed_tests)}")

    report_file = Path("integration_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Full report saved to: {report_file}")

    exit_code = 0 if report["status"] == "PASS" else 1

    if len(sys.argv) > 1 and sys.argv[1] == "--ci":
        # 1) Testes críticos (fluxo real de usuário) DEVEM passar
        missing_required = [
            name for name in REQUIRED_TESTS_FOR_CI
            if results.get(name, {}).get("status") != "success"
        ]
        if missing_required:
            print(
                "\n🚨 CI FAILURE: Critical integration tests failed: "
                + ", ".join(missing_required)
            )
            exit_code = 1

        # 2) Latência máxima por teste crítico
        slow_critical = {
            name: results[name].get("latency_ms")
            for name in REQUIRED_TESTS_FOR_CI
            if results.get(name, {}).get("status") == "success"
            and results[name].get("latency_ms") is not None
            and results[name]["latency_ms"] > MAX_SINGLE_LATENCY_MS
        }
        if slow_critical:
            print(
                f"\n🚨 CI FAILURE: One or more critical integration calls "
                f"exceeded {MAX_SINGLE_LATENCY_MS}ms:"
            )
            for name, ms in slow_critical.items():
                print(f"   - {name}: {ms:.2f}ms")
            exit_code = 1

        # 3) Latência média somente dos testes críticos
        critical_latencies = [
            results[name]["latency_ms"]
            for name in REQUIRED_TESTS_FOR_CI
            if results.get(name, {}).get("status") == "success"
            and results[name].get("latency_ms") is not None
        ]
        if critical_latencies:
            avg_critical = sum(critical_latencies) / len(critical_latencies)
        else:
            avg_critical = 0.0

        if avg_critical > MAX_AVG_LATENCY_MS:
            print(
                f"\n🚨 CI FAILURE: Average critical latency "
                f"{avg_critical:.2f}ms > {MAX_AVG_LATENCY_MS}ms"
            )
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
