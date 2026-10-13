"""Complemento da F2: como a serie de cada praca se distribui no tempo.

Objetivo: separar dois casos que exigem tratamentos DIFERENTES na F3:
  (a) praca que entrou tarde / saiu cedo -> buraco nas PONTAS (basta usar o trecho ativo);
  (b) praca com buracos NO MEIO (serie picada) -> exige decisao explicita (imputar/mascarar/excluir).

Uso:
    python src/eda_cobertura.py
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PARQUET = BASE / "data" / "trafego_pedagio.parquet"


def main() -> None:
    df = pd.read_parquet(PARQUET, columns=["concessionaria", "praca", "mes_ano"])
    df["data"] = pd.to_datetime(df["mes_ano"], format="%d/%m/%Y")
    df["praca_id"] = df["concessionaria"].astype(str) + " | " + df["praca"].astype(str)

    dias_possiveis = df["data"].nunique()

    linhas = []
    for pid, grp in df[["praca_id", "data"]].drop_duplicates().groupby("praca_id"):
        d = grp["data"].sort_values()
        obs = d.nunique()
        span = (d.max() - d.min()).days + 1     # extensao do trecho ativo
        linhas.append((pid, obs, d.min(), d.max(), span, span - obs))

    c = pd.DataFrame(linhas, columns=["praca_id", "obs", "inicio", "fim", "span", "buraco_meio"])

    print("=" * 72)
    print("COBERTURA POR PRACA (345 pracas; periodo:", dias_possiveis, "dias)")
    print(c["obs"].describe().to_string())

    print("\npracas por faixa de dias observados:")
    for limite in [974, 730, 365, 180, 90, 30]:
        n = int((c["obs"] >= limite).sum())
        print(f"  obs >= {limite:>4}: {n:>3} pracas")

    com_buraco = c[c["buraco_meio"] > 0]
    print("\npracas com buraco NO MEIO (dentro do trecho ativo):", len(com_buraco))
    print("total de dias faltantes no meio:", int(c["buraco_meio"].sum()))
    print("desses, quantas com > 30 dias faltando no meio:", int((c["buraco_meio"] > 30).sum()))

    print("\ntop 10 praticas com mais buraco no meio:")
    print(com_buraco.sort_values("buraco_meio", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
