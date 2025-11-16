import sys, os
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.processamento.semantic_index import SemanticSearchIndex


def run_tests():
    corpus = [
        "lavadora de piso automática 50 litros",
        "lavadora de piso ride-on 70L",
        "encereadeira / polidora de piso industrial",
        "lavadora de alta pressão 3000 psi profissional",
        "aspirador de pó industrial seco",
        "extratora de carpetes 30L",
    ]
    idx = SemanticSearchIndex.build(corpus)

    positives: List[Tuple[str, int]] = [
        ("automatic floor scrubber 50L tank", 0),
        ("ride-on scrubber dryer", 1),
        ("floor polisher industrial", 2),
        ("high-pressure washer 3000 PSI", 3),
        ("industrial vacuum cleaner dry", 4),
        ("carpet extractor 30 liters", 5),
    ]

    lows: List[str] = [
        "small battery charger",
        "office chair wheels",
        "paper towels dispenser",
    ]

    ok = 0
    for q, target_idx in positives:
        res = idx.search(q, top_k=3)
        got = [i for i, s in res]
        if target_idx in got:
            ok += 1
        print(f"Q: {q} -> {res}", ("OK" if target_idx in got else "MISS"))

    # baixa similaridade esperada (apenas checagem de que top score não é muito alto)
    for q in lows:
        res = idx.search(q, top_k=1)
        score = res[0][1] if res else 0.0
        print(f"LOW Q: {q} -> score={score:.3f}")

    print(f"positives hit: {ok}/{len(positives)}")


if __name__ == "__main__":
    run_tests()
