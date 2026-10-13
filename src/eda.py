"""F2 - EDA minima que gera a justificativa do relatorio.

Le data/trafego_pedagio.parquet (gerado por src/preprocess.py) e produz:
  - estatisticas no console (estrutura, alvo, cobertura por praca);
  - figuras em results/figures/ (histograma, serie temporal, dia da semana, heatmap).

O objetivo NAO e explorar por explorar: e produzir exatamente a evidencia que o
relatorio (Sec.2.1 do PDF) exige: entradas/alvo, distribuicao, cobertura, faltantes.

Uso:
    python src/eda.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sem janela: gera arquivos, nao abre plot
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
PARQUET = BASE / "data" / "trafego_pedagio.parquet"
FIG = BASE / "results" / "figures"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def carregar() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    # mes_ano chega como texto dd/mm/yyyy (data diaria) -> datetime (ver D-003/D-007)
    df["data"] = pd.to_datetime(df["mes_ano"], format="%d/%m/%Y", errors="coerce")
    # identidade da praca = concessionaria + praca (a chave da unidade de observacao)
    df["praca_id"] = df["concessionaria"].astype(str) + " | " + df["praca"].astype(str)
    return df


def estrutura(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("ESTRUTURA")
    print("linhas:", len(df))
    print("colunas:", list(df.columns))
    print(df.dtypes.to_string())
    print("datas nulas:", int(df["data"].isna().sum()))
    print("periodo:", df["data"].min().date(), "->", df["data"].max().date())
    print("dias distintos no periodo:", df["data"].nunique())
    print("concessionarias:", df["concessionaria"].nunique())
    print("pracas (so o nome):", df["praca"].nunique())
    print("pracas (par concessionaria|praca):", df["praca_id"].nunique())
    for col in ["tipo_de_veiculo", "sentido", "tipo_cobranca", "categoria_eixo"]:
        print(f"\n[{col}] {df[col].nunique()} valores")
        print(df[col].value_counts().head(12).to_string())


def alvo(df: pd.DataFrame) -> None:
    v = df["volume_total"]
    print("=" * 72)
    print("ALVO no nivel de linha: volume_total")
    print(v.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_string())
    print("zeros:", int((v == 0).sum()), f"({(v == 0).mean():.2%})")
    print("negativos:", int((v < 0).sum()))
    print("ausentes:", int(v.isna().sum()))
    print("assimetria (skew):", round(float(v.skew()), 3))


def serie_praca_dia(df: pd.DataFrame):
    """Agrega para praca x dia (canal total) - a base da serie temporal."""
    s = df.groupby(["praca_id", "data"], as_index=False)["volume_total"].sum()
    s = s.sort_values(["praca_id", "data"])

    total_linhas = df["volume_total"].sum()
    print("=" * 72)
    print("AGREGACAO praca x dia")
    print("linhas agregadas:", len(s), "| soma:", round(s["volume_total"].sum(), 2))
    print("soma conferida com o nivel de linha:", np.isclose(s["volume_total"].sum(), total_linhas))

    dias = df["data"].nunique()
    cnt = s.groupby("praca_id")["data"].nunique()
    print("dias possiveis no periodo:", dias)
    print("dias por praca -> describe:")
    print(cnt.describe().to_string())
    print("pracas com < 90% dos dias:", int((cnt < 0.9 * dias).sum()))
    print("pracas com < 50% dos dias:", int((cnt < 0.5 * dias).sum()))

    tot = s.groupby("praca_id")["volume_total"].sum().sort_values(ascending=False)
    print("top 10 pracas por volume total:")
    print(tot.head(10).to_string())
    return s, tot


def figuras(df: pd.DataFrame, s: pd.DataFrame, tot: pd.Series) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # 1) distribuicao do volume (log1p para caber tudo num grafico so)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(np.log1p(df["volume_total"].clip(lower=0)), bins=80, color="#3b6ea5")
    ax.set_title("Distribuicao de volume_total (log1p, nivel de linha)")
    ax.set_xlabel("log1p(volume)"); ax.set_ylabel("frequencia")
    fig.tight_layout(); fig.savefig(FIG / "01_hist_volume.png", dpi=120); plt.close(fig)

    # 2) serie temporal: total diario + top 4 pracas
    pivot = s.pivot(index="data", columns="praca_id", values="volume_total")
    top4 = tot.head(4).index
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(pivot.index, pivot.sum(axis=1), color="#111")
    axes[0].set_title("Total diario (todas as pracas)")
    for ax, pid in zip(axes[1:], top4):
        ax.plot(pivot.index, pivot[pid], color="#3b6ea5")
        ax.set_title(str(pid)[:75])
    fig.tight_layout(); fig.savefig(FIG / "02_serie_temporal.png", dpi=120); plt.close(fig)

    # 3) volume medio por dia da semana (motiva as features de calendario - D-007)
    dd = s.copy()
    dd["dow"] = dd["data"].dt.dayofweek
    media = dd.groupby("dow")["volume_total"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["seg", "ter", "qua", "qui", "sex", "sab", "dom"], media.values, color="#3b6ea5")
    ax.set_title("Volume medio por dia da semana")
    ax.set_ylabel("volume medio")
    fig.tight_layout(); fig.savefig(FIG / "03_dia_semana.png", dpi=120); plt.close(fig)

    # 4) heatmap praca (top 30) x mes
    top30 = tot.head(30).index
    hm = s[s["praca_id"].isin(top30)].copy()
    hm["mes"] = hm["data"].dt.to_period("M").astype(str)
    mat = hm.pivot_table(index="praca_id", columns="mes", values="volume_total", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([str(i)[:45] for i in mat.index], fontsize=6)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=6)
    ax.set_title("Heatmap de volume: praca (top 30) x mes")
    fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(FIG / "04_heatmap_praca_mes.png", dpi=120); plt.close(fig)

    print("=" * 72)
    print("figuras salvas em", FIG)


def main() -> None:
    df = carregar()
    estrutura(df)
    alvo(df)
    s, tot = serie_praca_dia(df)
    figuras(df, s, tot)


if __name__ == "__main__":
    main()
