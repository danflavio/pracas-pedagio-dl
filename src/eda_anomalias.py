"""F2 - Valores anomalos (PDF Sec.2.1): criterio e decisao.

Pergunta do enunciado: existem picos/quedas atipicas? O que fazer com eles?

Criterio adotado: **IQR por praca**, com limite de **3x o IQR** (marca apenas
extremos, nao o dia a dia). Calculado sobre o Total diario de cada praca.

Decisao: **manter** as linhas. Justificativa: um volume atipico aqui
provavelmente e um evento real (feriado, obra, acidente, operacao especial), e
remover isso apagaria justamente os dias mais dificeis de prever. O criterio e
usado para MEDIR o fenomeno, nao para apagar dado.

Uso:
    python src/eda_anomalias.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
SERIE = BASE / "data" / "series_praca_dia.parquet"
RESULTS = BASE / "results"
K = 3.0          # multiplicador do IQR


def main() -> None:
    s = pd.read_parquet(SERIE)[["praca_id", "data", "Total", "observado"]]
    s = s[s["observado"]].copy()

    # limites por praca
    g = s.groupby("praca_id")["Total"]
    q1 = g.transform(lambda x: x.quantile(0.25))
    q3 = g.transform(lambda x: x.quantile(0.75))
    iqr = (q3 - q1).replace(0, np.nan)
    s["limite_inf"] = q1 - K * iqr
    s["limite_sup"] = q3 + K * iqr
    s["anomalo"] = (s["Total"] < s["limite_inf"]) | (s["Total"] > s["limite_sup"])
    s["anomalo"] = s["anomalo"].fillna(False)

    n = len(s)
    n_an = int(s["anomalo"].sum())
    pracas_com = int(s.loc[s["anomalo"], "praca_id"].nunique())

    print("=" * 68)
    print("VALORES ANOMALOS - criterio IQR por praca (limite de %.0fx o IQR)" % K)
    print("dias observados ..............:", n)
    print("dias marcados como anomalos ..:", n_an, "(%.2f%%)" % (100 * n_an / n))
    print("pracas com ao menos 1 anomalia:", pracas_com, "de", s["praca_id"].nunique())

    vol_an = s.loc[s["anomalo"], "Total"].sum()
    vol_tot = s["Total"].sum()
    print("participacao no volume total .: %.3f%%" % (100 * vol_an / vol_tot))

    print("\n(dos anomalos, quantos sao picos para cima x quedas)")
    ac = s[s["anomalo"]]
    print("  acima do limite superior:", int((ac["Total"] > ac["limite_sup"]).sum()))
    print("  abaixo do limite inferior:", int((ac["Total"] < ac["limite_inf"]).sum()))

    print("\n20 maiores desvios (Total vs limite superior):")
    top = ac.sort_values("Total", ascending=False).head(20).copy()
    top["razao"] = (top["Total"] / top["limite_sup"]).round(2)
    print(top[["praca_id", "data", "Total", "limite_sup", "razao"]].to_string(index=False))

    # decisoes resumidas por praca, para o relatorio
    resumo = (s.groupby("praca_id")
                .agg(dias=("Total", "size"), anomalos=("anomalo", "sum"))
                .reset_index())
    resumo = resumo[resumo["anomalos"] > 0].sort_values("anomalos", ascending=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(RESULTS / "anomalias_por_praca.csv", index=False)
    print("\nresumo por praca gravado em results/anomalias_por_praca.csv")

    print("\nDECISAO: manter as linhas (eventos reais), sem winsorizar e sem remover.")


if __name__ == "__main__":
    main()
