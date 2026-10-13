"""F4/F5 - Protocolo de treino, baselines e metricas na unidade do alvo (veiculos).

Pontos que a rubrica exige e que estao explicitos aqui:
  - baselines de referencia: persistencia (y = ultimo valor) e media movel;
  - early stopping + ReduceLROnPlateau (custo controlado, sem overfitting descontrolado);
  - metricas MAE / RMSE / R2 interpretadas em veiculos (desfazendo a padronizacao);
  - curvas de treino/validacao salvas em results/figures/.

Uso:
    python src/train.py                 # rodada-piloto: 3 familias com config de referencia
    python src/train.py --smoke         # 2 epocas, so para validar o pipeline
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds          # noqa: E402
import model as mdl           # noqa: E402

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from tensorflow import keras  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
FIG = RESULTS / "figures"

CONFIG_REFERENCIA = {
    "units": 64, "n_layers": 1, "dropout": 0.0, "recurrent_dropout": 0.0,
    "learning_rate": 1e-3, "optimizer": "adam", "clipnorm": 1.0,
    "batch_size": 64, "epochs": 60, "patience": 6, "seed": ds.GLOBAL_SEED,
}


# --------------------------------------------------------------------------- #
def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE e R2 - todos na unidade do alvo (veiculos), nao na escala padronizada."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def escalonar(y: np.ndarray, prep: dict) -> np.ndarray:
    idx = ds.TOTAL_IDX
    return (y - prep["media"][idx]) / prep["desvio"][idx]


def desescalonar(y: np.ndarray, prep: dict) -> np.ndarray:
    idx = ds.TOTAL_IDX
    return y * prep["desvio"][idx] + prep["media"][idx]


def canais_reais(X: np.ndarray, prep: dict) -> np.ndarray:
    """Desfaz a padronizacao dos canais de volume de um tensor X (n, W, F)."""
    X = X.copy()
    n_vol = len(ds.CANAIS_VOL)
    X[:, :, :n_vol] = X[:, :, :n_vol] * prep["desvio"] + prep["media"]
    return X


def baselines(prep: dict) -> pd.DataFrame:
    """Referencias minimas: persistencia e media movel, nos 3 conjuntos."""
    linhas = []
    for nome in ("train", "val", "test"):
        vol = canais_reais(prep[f"X_{nome}"], prep)
        y = prep[f"y_{nome}"]
        persist = vol[:, -1, ds.TOTAL_IDX]                 # ultimo valor observado
        media_mov = vol[:, :, ds.TOTAL_IDX].mean(axis=1)   # media da janela
        for rotulo, pred in (("persistencia", persist), ("media_movel", media_mov)):
            linhas.append({"conjunto": nome, "baseline": rotulo, **metricas(y, pred)})
    return pd.DataFrame(linhas)


def treinar(prep: dict, params: dict, verbose: int = 0, extra_callbacks=None) -> dict:
    """Treina um modelo com early stopping e devolve historico + metricas na validacao."""
    keras.backend.clear_session()
    m = mdl.construir(params, prep["window"], prep["X_train"].shape[-1])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(params.get("patience", 6)),
                                      restore_best_weights=True, min_delta=1e-4),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    callbacks += list(extra_callbacks or [])

    t0 = time.perf_counter()
    h = m.fit(
        prep["X_train"], escalonar(prep["y_train"], prep),
        validation_data=(prep["X_val"], escalonar(prep["y_val"], prep)),
        epochs=int(params.get("epochs", 60)),
        batch_size=int(params.get("batch_size", 64)),
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,                    # ordem temporal preservada (convencao do projeto)
    )
    tempo = time.perf_counter() - t0

    pred = desescalonar(m.predict(prep["X_val"], verbose=0).ravel(), prep)
    return {
        "model": m,
        "history": h.history,
        "metrics_val": metricas(prep["y_val"], pred),
        "epochs_rodadas": len(h.history["loss"]),
        "tempo_treino_s": tempo,
        "n_params": int(m.count_params()),
    }


# --------------------------------------------------------------------------- #
def _plot_curvas(historico: dict, titulo: str, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(historico["loss"], label="treino")
    ax.plot(historico["val_loss"], label="validacao")
    ax.set_xlabel("epoca"); ax.set_ylabel("MSE (escala padronizada)")
    ax.set_title(titulo); ax.legend()
    fig.tight_layout(); fig.savefig(destino, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 epocas, so valida o pipeline")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--split", default="chrono", choices=["chrono", "grupo"])
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    prep = ds.preparar(window=args.window, split=args.split)

    base = baselines(prep)
    print("BASELINES (referencia minima)")
    print(base.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    base.to_csv(RESULTS / f"baselines_{args.split}.csv", index=False)

    registros = []
    for familia in ("SimpleRNN", "GRU", "LSTM"):
        params = {**CONFIG_REFERENCIA, "model_type": familia}
        if args.smoke:
            params.update(epochs=2, patience=2)
        print(f"\n>>> treinando {familia} ...")
        r = treinar(prep, params, verbose=0)
        _plot_curvas(r["history"], f"{familia} - perda treino/validacao",
                     FIG / f"curva_{familia.lower()}.png")
        registros.append({
            "familia": familia, "window": args.window, "split": args.split,
            "units": params["units"], "n_layers": params["n_layers"],
            "epocas": r["epochs_rodadas"], "params": r["n_params"],
            "tempo_s": round(r["tempo_treino_s"], 1), **r["metrics_val"],
        })
        print("   ", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["metrics_val"].items()},
              f"| epocas={r['epochs_rodadas']} | {r['tempo_treino_s']:.0f}s")

    tabela = pd.DataFrame(registros)
    tabela.to_csv(RESULTS / f"piloto_{args.split}_w{args.window}.csv", index=False)
    print("\nRESULTADO NA VALIDACAO (unidade: veiculos)")
    print(tabela.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
