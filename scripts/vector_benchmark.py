"""
Benchmark de vetorização para validação de performance em CI.
Objetivo: garantir que o método vetorizado seja consistentemente
mais rápido que o método baseado em apply.
"""

import time
import re
import pandas as pd

# Meta mínima de ganho (ex: 1.02 = 2% mais rápido)
TARGET_SPEEDUP = 1.02


def make_data(n_rows: int = 200_000) -> pd.DataFrame:
    """Gera dados sintéticos com formatos típicos BR."""
    base = [
        "1.234,56",
        "2,3",
        "R$ 4,50",
        "7.890,12",
        "0,99",
        "-123,45",
        "R$ -6,70",
    ]
    data = (base * ((n_rows // len(base)) + 1))[:n_rows]
    return pd.DataFrame({"val": data})


def old_method(series: pd.Series) -> pd.Series:
    """Método antigo baseado em apply (não vetorizado)."""

    def parse(x):
        s = str(x)
        # Remove tudo que não for dígito, vírgula, ponto ou sinal
        s = re.sub(r"[^\d,.\-]", "", s)
        # Remove ponto de milhar e troca vírgula por ponto
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    return series.apply(parse)


def new_method(series: pd.Series) -> pd.Series:
    """Método vetorizado usando operações de string do pandas."""
    s = series.astype(str)
    # Limpa caracteres indesejados em uma passada
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)
    # Remove ponto de milhar e troca vírgula por ponto
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def bench(fn, series, repeat: int = 5) -> float:
    """Roda a função `repeat` vezes e retorna o menor tempo."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(series)
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    df = make_data()
    # Warm-up do método novo para evitar efeitos de primeira execução
    new_method(df["val"])

    old_t = bench(old_method, df["val"])
    new_t = bench(new_method, df["val"])
    speedup = old_t / new_t if new_t > 0 else float("inf")

    print(f"Old time: {old_t:.4f}s | New time: {new_t:.4f}s | Speedup: {speedup:.2f}x")

    # 1) Garantia forte: método novo PRECISA ser mais rápido
    assert new_t < old_t, (
        f"Método novo está mais lento ou igual "
        f"({new_t:.4f}s vs {old_t:.4f}s)"
    )

    # 2) Garantia de ganho mínimo configurável
    assert speedup >= TARGET_SPEEDUP, (
        f"Speedup {speedup:.2f}x abaixo da meta "
        f"{TARGET_SPEEDUP:.2f}x"
    )

    print("✅ Performance benchmarks passed")


if __name__ == "__main__":
    main()
