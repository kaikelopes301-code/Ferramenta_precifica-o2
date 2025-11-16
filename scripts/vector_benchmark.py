"""
Benchmark de vetorização para validação de performance em CI.
Objetivo: garantir que o método vetorizado seja mais rápido que o antigo.
"""

import time
import re
import pandas as pd

# Meta mínima de ganho (ex: 1.02 = 2% mais rápido)
TARGET_SPEEDUP = 1.02


def make_data(n_rows: int = 200_000) -> pd.DataFrame:
    base = ["1.234,56", "2,3", "R$ 4,50", "7.890,12", "0,99", "-123,45", "R$ -6,70"]
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
    s = s.str.replace(r"[^\d,.\-]", "", regex=True)  # limpa em uma passada
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)  # milhar/vírgula
    return pd.to_numeric(s, errors="coerce")


def bench(fn, series, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(series)
        times.append(time.perf_counter() - t0)
    return min(times)  # menor tempo para reduzir ruído


def main():
    df = make_data()
    new_method(df["val"])  # warm-up

    old_t = bench(old_method, df["val"])
    new_t = bench(new_method, df["val"])
    speedup = old_t / new_t if new_t > 0 else float("inf")

    print(f"Old time: {old_t:.4f}s | New time: {new_t:.4f}s | Speedup: {speedup:.2f}x")

    # Garante que o método novo é mais rápido
    assert new_t < old_t, f"Método novo está mais lento ({new_t:.4f}s vs {old_t:.4f}s)"

    # Garante um ganho mínimo configurável
    assert (
        speedup >= TARGET_SPEEDUP
    ), f"Speedup {speedup:.2f}x abaixo da meta {TARGET_SPEEDUP:.2f}x"

    print("✅ Performance benchmarks passed")


if __name__ == "__main__":
    main()
