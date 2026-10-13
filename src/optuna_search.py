"""F5 - Busca de hiperparametros com Optuna.

Busca DELIBERADAMENTE enxuta: cada dimensao precisa ser explicavel em uma frase.

Decisoes (justificadas no relatorio - PDF Sec.2.4):
  - metrica otimizada: RMSE na VALIDACAO (o teste nunca participa da busca);
  - pruner MedianPruner: trials ruins param cedo -> cabe no orcamento de CPU;
  - paciencia folgada no early stopping: o piloto mostrou que as RNNs precisam de
    ~50 epocas para convergir; cortar epocas demais faria TODOS os trials sob
    underfit (a busca ranquearia quem aprende rapido, nao quem e melhor);
  - orcamento de CPU: a busca roda sobre uma FRACAO do treino (subamostragem
    sistematica, sem embaralhar). A config vencedora e retreinada com o treino
    INTEIRO na F6/F7 -> a busca serve para RANQUEAR, nao e o resultado final;
  - storage SQLite em data/ (fora do Git); export legivel em results/.

Espaco de busca:
  model_type {SimpleRNN,GRU,LSTM} x window {7,14,30} x units 32-128 (log) x
  n_layers {1,2} x dropout 0-0.3 x lr 1e-4..5e-3 (log) x batch {128,256} x
  otimizador {adam,rmsprop} x clipnorm=1.0
  (batch 64 descartado: medido em 21,1s/epoca contra 10,2s do batch 256)

Uso:
    python src/optuna_search.py --trials 15 --epochs 40 --fracao 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import optuna                  # noqa: E402
import pandas as pd            # noqa: E402
from tensorflow import keras   # noqa: E402

import dataset as ds           # noqa: E402
import train as tr             # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
STORAGE = f"sqlite:///{(BASE / 'data' / 'optuna_study.db').as_posix()}"
STUDY = "rnn_trafego"

# cache: window -> prep (construir as janelas e caro; reusar entre trials)
_PREP: dict[int, dict] = {}


def prep_para(window: int, split: str = "chrono") -> dict:
    if window not in _PREP:
        _PREP[window] = ds.preparar(window=window, split=split)
    return _PREP[window]


def subamostrar(prep: dict, fracao: float) -> dict:
    """Reduz SO o treino, por passos regulares (cobertura temporal preservada).

    Subamostragem sistematica e determinista - nao embaralha nada. Usada apenas na
    busca; o modelo final e treinado com o treino completo.
    """
    if fracao >= 1.0:
        return prep
    p = dict(prep)
    n = len(p["y_train"])
    idx = np.linspace(0, n - 1, max(1, int(fracao * n))).astype(int)
    p["X_train"] = p["X_train"][idx]
    p["y_train"] = p["y_train"][idx]
    return p


class PodarPorEpoca(keras.callbacks.Callback):
    """Reporta o val_loss ao Optuna a cada epoca e interrompe trial ruim."""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(float(logs["val_loss"]), epoch)
        if self.trial.should_prune():
            self.model.stop_training = True
            raise optuna.TrialPruned()


def objetivo(trial: optuna.Trial, epochs: int, patience: int, fracao: float) -> float:
    window = trial.suggest_categorical("window", [7, 14, 30])
    model_type = trial.suggest_categorical("model_type", ["SimpleRNN", "GRU", "LSTM"])
    units = trial.suggest_int("units", 32, 128, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch = trial.suggest_categorical("batch_size", [128, 256])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

    params = {
        "model_type": model_type, "units": units, "n_layers": n_layers,
        "dropout": dropout, "recurrent_dropout": 0.0,
        "learning_rate": lr, "optimizer": optimizer, "clipnorm": 1.0,
        "batch_size": batch, "epochs": epochs, "patience": patience,
        "seed": ds.GLOBAL_SEED,
    }
    prep = subamostrar(prep_para(window), fracao)
    r = tr.treinar(prep, params, verbose=0, extra_callbacks=[PodarPorEpoca(trial)])
    trial.set_user_attr("n_params", r["n_params"])
    trial.set_user_attr("tempo_s", round(r["tempo_treino_s"], 1))
    trial.set_user_attr("epocas", r["epochs_rodadas"])
    return r["metrics_val"]["RMSE"]


def exportar(study: optuna.Study) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = study.trials_dataframe()
    df.to_csv(RESULTS / "optuna_trials.csv", index=False)

    completos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    melhor_por_familia = {}
    for fam in ("SimpleRNN", "GRU", "LSTM"):
        fs = [t for t in completos if t.params.get("model_type") == fam]
        if fs:
            b = min(fs, key=lambda t: t.value)
            melhor_por_familia[fam] = {"RMSE_val": b.value, "params": b.params, "trial": b.number}

    best = study.best_trial
    resumo = {
        "n_trials_total": len(study.trials),
        "n_trials_completos": len(completos),
        "melhor_trial": best.number,
        "melhor_RMSE_val": best.value,
        "melhores_params": best.params,
        "melhor_por_familia": melhor_por_familia,
    }
    (RESULTS / "optuna_resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--fracao", type=float, default=0.5,
                    help="fracao do treino usada na BUSCA (0<f<=1)")
    ap.add_argument("--timeout", type=int, default=0, help="segundos (0 = sem limite)")
    args = ap.parse_args()

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(
        direction="minimize", study_name=STUDY, storage=STORAGE,
        load_if_exists=True, pruner=pruner,
    )
    print(f"trials ja no estudo: {len(study.trials)} | novos: {args.trials}", flush=True)
    print(f"config: epochs={args.epochs} patience={args.patience} fracao_treino={args.fracao}", flush=True)

    def relatar(study, trial):
        print(f"  trial {trial.number}: {trial.state.name} valor={trial.value}"
              f" | epocas={trial.user_attrs.get('epocas')}"
              f" | {trial.user_attrs.get('tempo_s')}s", flush=True)

    t0 = time.perf_counter()
    study.optimize(
        lambda t: objetivo(t, args.epochs, args.patience, args.fracao),
        n_trials=args.trials, timeout=(args.timeout or None),
        callbacks=[relatar],
    )
    print(f"tempo total: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    exportar(study)


if __name__ == "__main__":
    main()
