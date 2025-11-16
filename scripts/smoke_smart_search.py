import requests
import json

URL = "http://localhost:8000/buscar-inteligente"

def test(q: str):
    r = requests.post(URL, json={"descricao": q, "top_k": 5})
    try:
        data = r.json()
    except Exception:
        print(q, '-> erro parse', r.text)
        return
    print('\nQUERY:', q)
    print('NORMALIZADA:', data.get('query_normalizada'))
    for item in data.get('resultados', []):
        print(f"  #{item['ranking']} {item['sugeridos']} score={item['score']}")

if __name__ == '__main__':
    queries = [
        'enceradeira 510',
        'enceradeira modelo 350',
        'enceradeira 220v',
        'enceradeira 110v',
        'enceradeira industrial',
    ]
    for q in queries:
        test(q)
