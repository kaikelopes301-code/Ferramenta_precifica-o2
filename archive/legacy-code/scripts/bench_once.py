import subprocess, sys, time, json, os, signal
import requests

ROOT = os.path.abspath(os.path.dirname(__file__) + '/..')
os.chdir(ROOT)

def main():
    # activate venv if present in PATH resolution (best effort)
    env = os.environ.copy()
    uvicorn_args = [sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '127.0.0.1', '--port', '8001']
    # Não suprimir logs para diagnosticar falhas de startup
    srv = subprocess.Popen(uvicorn_args)
    try:
        time.sleep(4.0)
        url = 'http://127.0.0.1:8001/buscar-inteligente'
        payload = {"descricao": "mop industrial", "top_k": 8}
        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=10)
        dt = (time.perf_counter() - t0) * 1000.0
        try:
            data = r.json()
        except Exception:
            data = None
        print(f"elapsed_ms={int(round(dt))}")
        if data and isinstance(data, dict):
            resultados = data.get('resultados') or []
            if resultados:
                top1 = resultados[0]
                sug = top1.get('sugeridos', 'N/A')
                score = top1.get('score') or top1.get('confianca')
                print(f"top1={sug}; score={score}")
            else:
                print("no_results=true")
        else:
            print("parse_error=true")
    finally:
        try:
            if os.name == 'nt':
                srv.terminate()
            else:
                os.kill(srv.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            srv.wait(timeout=3)
        except Exception:
            try:
                srv.kill()
            except Exception:
                pass

if __name__ == '__main__':
    main()
