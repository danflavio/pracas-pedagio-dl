"""Gera a figura de apoio dos slides: volume medio por dia da semana.

Diferente das figuras da EDA (feitas para inspecao em tela), esta e desenhada para
PROJECAO: fonte grande, poucos elementos, uma ideia so.

Ela sustenta o achado "a serie e diaria": se o movimento cai no fim de semana,
modelar por mes jogaria essa informacao fora.

Uso:
    python src/figura_dia_semana.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
SERIE = BASE / "data" / "series_praca_dia.parquet"
DESTINO = BASE / "results" / "figures" / "slide_dia_semana.png"


def main() -> None:
    s = pd.read_parquet(SERIE)[["data", "Total", "observado"]]
    s = s[s["observado"]]
    media = s.assign(dow=s["data"].dt.dayofweek).groupby("dow")["Total"].mean()

    rotulos = ["seg", "ter", "qua", "qui", "sex", "s\u00e1b", "dom"]
    cores = ["#0E7A3C"] * 5 + ["#C8102E", "#C8102E"]

    fig, ax = plt.subplots(figsize=(11, 3.0), dpi=110)
    barras = ax.bar(rotulos, media.values, color=cores, width=0.68)
    for b, v in zip(barras, media.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}".replace(",", "."),
                ha="center", va="bottom", fontsize=21, fontweight="bold", color="#1A231E")

    ax.set_ylim(0, media.max() * 1.18)
    ax.set_ylabel("ve\u00edculos por dia", fontsize=17, color="#6C7A72")
    ax.tick_params(axis="x", labelsize=22)
    ax.tick_params(axis="y", labelsize=15, colors="#6C7A72")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color("#E1EAE4")
    ax.spines["bottom"].set_color("#E1EAE4")
    ax.grid(axis="y", color="#EDF3EF")
    ax.set_axisbelow(True)
    fig.tight_layout()
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINO)
    plt.close(fig)
    print("gravado:", DESTINO)


if __name__ == "__main__":
    main()
