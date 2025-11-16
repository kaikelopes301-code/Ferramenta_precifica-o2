import sys
import os

# Garante que a raiz do projeto esteja no sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ingestao.excel import load_excel
from src.processamento.normalize import extract_attributes, is_duplicate
from src.processamento.similarity import TfidfSearchIndex, HybridTfidfSearchIndex

print("Smoke test starting...")

try:
    # Excel: não falhar se não existir, apenas avisa
    try:
        df = load_excel()
        print(f"Excel ok: {len(df)} linhas")
    except Exception as e:
        print(f"Excel não testado: {e}")

    # NLP helpers
    attrs = extract_attributes("Furadeira 220V 10mm profissional")
    print("Atributos extraídos:", attrs)

    print("Similaridade:", is_duplicate("Furadeira Makita 220V", "Furadeira 220V Makita"))

    # TF-IDF n-grams char_wb: pequena demo
    corpus = [
        "Furadeira Makita 220V 10mm",
        "Parafusadeira Bosch 12V",
        "Furadeira de Impacto 127V 13mm",
        "Serra Circular 220V 185mm",
        "Furadeira 220V Black+Decker 10mm",
    ]
    index = TfidfSearchIndex.build(corpus, analyzer="char_wb", ngram_range=(3, 5))
    q = "furadeira 10 mm 220v"
    res = index.search(q, top_k=3)
    print("TF-IDF similares para:", q)
    for i, score in res:
        print(f" - {corpus[i]} (score={score:.3f})")

    # Híbrido com âncoras: diferencia 'mop' de 'cadeira giratória'
    corpus2 = [
        "Mop Fit Giratório Flash Limp Mop5010",
        "Cadeira giratória de escritório",
        "Mop pó 60 cm completo",
        "Refil mop microfibra 60cm",
        "Cadeira fixa visitante",
        "Balde Mop com torção",
    ]
    index_h = HybridTfidfSearchIndex.build(corpus2)
    q2 = "Mop Fit Giratório Flash Limp Mop5010"
    res2 = index_h.search(q2, top_k=3)
    print("Híbrido similares para:", q2)
    for i, score in res2:
        print(f" - {corpus2[i]} (score={score:.3f})")

    # Consulta de 1 palavra: trazer variações (ex.: vassoura)
    corpus3 = [
        "Vassoura de nylon",
        "Vassoura piaçava",
        "Vassoura de teto",
        "Vassourão industrial",
        "Pá de lixo",
        "Rodo 60cm",
        "Vassoura feiticeira",
    ]
    index_v = HybridTfidfSearchIndex.build(corpus3)
    q3 = "vassoura"
    res3 = index_v.search(q3, top_k=5)
    print("Híbrido (consulta genérica) para:", q3)
    for i, score in res3:
        print(f" - {corpus3[i]} (score={score:.3f})")
    # typo
    q3b = "vasoura"
    res3b = index_v.search(q3b, top_k=5)
    print("Híbrido (consulta com typo) para:", q3b)
    for i, score in res3b:
        print(f" - {corpus3[i]} (score={score:.3f})")

    print("Smoke test pass.")
    sys.exit(0)
except Exception as e:
    print("Smoke test FAIL:", e)
    sys.exit(1)
