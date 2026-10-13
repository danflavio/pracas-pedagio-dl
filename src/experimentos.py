"""F6 - Experimentos exigidos pela frente 3.2 (redes recorrentes).

Cobre, com tabelas/figuras em results/:
  1. SimpleRNN vs GRU vs LSTM sob a MESMA config (so muda `model_type`);
  2. tamanhos de janela (7, 14, 30 dias) -> resposta a dependencias de longo prazo;
  3. profundidade x largura (n_layers x units) -> desempenho vs no de parametros;
  4. custo: tempo de treino, tempo de inferencia (ms/predicao) e no de parametros.

Estrategia de previsao: 1 passo (X = window x 13 canais -> Total do dia seguinte).
Multi-step (t+2, t+3) NAO foi implementado - declarado como limitacao no relatorio.

Regra de comparacao justa: dentro de cada experimento, TODOS os modelos usam os
mesmos dados, a mesma seed e as mesmas epocas. Muda so a dimensao em estudo.

Uso:
    python src/experimentos.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds   # noqa: E402
import train as tr     # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
FIG = RESULTS / "figures"
EPOCAS = 50
PACIENCIA = 5

_PREP: dict[int, dict] = {}


def prep_de(window: int) -> dict:
    if window not in _PREP:
        _PREP[window] = ds.preparar(window=window)
    return _PREP[window]


def melhor_config() -> dict:
    resumo = json.loads((RESULTS / "optuna_resumo.json").read_text(encoding="utf-8"))
    return dict(resumo["melhores_params"])


def avaliar(prep: dict, params: dict) -> dict:
    """Treina e mede: metricas na validacao + custo (treino, inferencia, parametros)."""
    r = tr.treinar(prep, params, verbose=0)
    t0 = time.perf_counter()
    pred = r["model"].predict(prep["X_val"], verbose=0)
    dt = time.perf_counter() - t0
    return {
        "params": r["n_params"],
        "tempo_treino_s": round(r["tempo_treino_s"], 1),
        "ms_predicao": round(1000 * dt / max(1, len(pred)), 3),
        "epocas": r["epochs_rodadas"],
        **r["metrics_val"],
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    base = melhor_config()
    print("config de referencia (melhor trial do Optuna):", base, flush=True)

    registros = []

    # 1) FAMILIAS - mesma config, so muda o tipo de camada recorrente
    for fam in ("SimpleRNN", "GRU", "LSTM"):
        p = {**base, "model_type": fam, "epochs": EPOCAS, "patience": PACIENCIA}
        print(f"[familia] {fam} (window={p['window']}) ...", flush=True)
        registros.append({"experimento": "familia", "variante": fam,
                          "window": p["window"], "units": p["units"], "n_layers": p["n_layers"],
                          **avaliar(prep_de(p["window"]), p)})

    # 2) JANELAS - mesma familia/config, so muda a janela
    fam = base["model_type"]
    for w in (7, 14, 30):
        p = {**base, "model_type": fam, "window": w, "epochs": EPOCAS, "patience": PACIENCIA}
        print(f"[janela] {w} dias ...", flush=True)
        registros.append({"experimento": "janela", "variante": f"w{w}",
                          "window": w, "units": p["units"], "n_layers": p["n_layers"],
                          **avaliar(prep_de(w), p)})

    # 3a) PROFUNDIDADE (camadas recorrentes)
    for nl in (1, 2, 3):
        p = {**base, "n_layers": nl, "epochs": EPOCAS, "patience": PACIENCIA}
        print(f"[profundidade] n_layers={nl} ...", flush=True)
        registros.append({"experimento": "profundidade", "variante": f"L{nl}",
                          "window": p["window"], "units": p["units"], "n_layers": nl,
                          **avaliar(prep_de(p["window"]), p)})

    # 3b) LARGURA (unidades por camada)
    for u in (32, 64, 128):
        p = {**base, "units": u, "epochs": EPOCAS, "patience": PACIENCIA}
        print(f"[largura] units={u} ...", flush=True)
        registros.append({"experimento": "largura", "variante": f"U{u}",
                          "window": p["window"], "units": u, "n_layers": p["n_layers"],
                          **avaliar(prep_de(p["window"]), p)})

    df = pd.DataFrame(registros)
    df.to_csv(RESULTS / "experimentos.csv", index=False)
    print("\nEXPERIMENTOS (validacao, unidade: veiculos)", flush=True)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.3f}"), flush=True)

    _plotar(df)


def _plotar(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for exp, titulo in (("familia", "Familias: MAE na validacao"),
                        ("janela", "Tamanho da janela: MAE na validacao"),
                        ("largura", "Largura (units): MAE x parametros")):
        d = df[df["experimento"] == exp]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(d["variante"].astype(str), d["MAE"], color="#3b6ea5")
        ax.set_title(titulo)
        ax.set_ylabel("MAE (veiculos)")
        fig.tight_layout()
        fig.savefig(FIG / f"exp_{exp}.png", dpi=120)
        plt.close(fig)

    d = df[df["experimento"] == "profundidade"]
    if not d.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(d["params"], d["MAE"], "o-", color="#3b6ea5")
        for _, r in d.iterrows():
            ax.annotate(str(r["variante"]), (r["params"], r["MAE"]))
        ax.set_xlabel("no de parametros")
        ax.set_ylabel("MAE (veiculos)")
        ax.set_title("Custo x desempenho (profundidade)")
        fig.tight_layout()
        fig.savefig(FIG / "exp_custo_parametros.png", dpi=120)
        plt.close(fig)

    print("figuras salvas em", FIG, flush=True)


if __name__ == "__main__":
    main()
