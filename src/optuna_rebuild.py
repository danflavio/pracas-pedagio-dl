"""Reconstroi o estudo do Optuna a partir do export versionado.

Motivo: o banco do estudo (data/optuna_study.db) NAO entra no Git (data/ esta no
.gitignore). Este script recria o estudo no SQLite a partir do CSV que ESTA
versionado (results/optuna_trials.csv), para o Optuna Dashboard funcionar em
qualquer maquina - sem retreinar nada.

Uso:
    python src/optuna_rebuild.py            # cria data/optuna_study.db (erro se existir)
    python src/optuna_rebuild.py --force    # apaga e recria

Depois:
    optuna-dashboard sqlite:///data/optuna_study.db
    # abra http://127.0.0.1:8080

Observacao: o create_trial do Optuna nao aceita datetime_start/complete, entao as
linhas do tempo ficam com o horario da reconstrucao (as demais telas nao usam isso).
"""

import argparse
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState

BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "results" / "optuna_trials.csv"
DB = BASE / "data" / "optuna_study.db"
STORAGE = f"sqlite:///{DB.as_posix()}"
STUDY = "rnn_trafego"

# Mesmo espaco de busca de src/optuna_search.py
DISTRIBUICOES = {
    "window": optuna.distributions.CategoricalDistribution([7, 14, 30]),
    "model_type": optuna.distributions.CategoricalDistribution(["SimpleRNN", "GRU", "LSTM"]),
    "units": optuna.distributions.IntDistribution(32, 128, log=True),
    "n_layers": optuna.distributions.IntDistribution(1, 2),
    "dropout": optuna.distributions.FloatDistribution(0.0, 0.3),
    "learning_rate": optuna.distributions.FloatDistribution(1e-4, 5e-3, log=True),
    "batch_size": optuna.distributions.CategoricalDistribution([128, 256]),
    "optimizer": optuna.distributions.CategoricalDistribution(["adam", "rmsprop"]),
}


def _num(valor):
    return None if pd.isna(valor) else float(valor)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recria o estudo do Optuna a partir do CSV.")
    ap.add_argument("--force", action="store_true", help="apaga o banco existente antes de recriar")
    args = ap.parse_args()

    if DB.exists():
        if not args.force:
            raise SystemExit(f"{DB} ja existe. Use --force para recriar.")
        DB.unlink()

    df = pd.read_csv(CSV)
    study = optuna.create_study(study_name=STUDY, storage=STORAGE, direction="minimize")

    n_ok = n_podados = 0
    for _, row in df.iterrows():
        params = {
            "window": int(row["params_window"]),
            "model_type": str(row["params_model_type"]),
            "units": int(row["params_units"]),
            "n_layers": int(row["params_n_layers"]),
            "dropout": float(row["params_dropout"]),
            "learning_rate": float(row["params_learning_rate"]),
            "batch_size": int(row["params_batch_size"]),
            "optimizer": str(row["params_optimizer"]),
        }
        user_attrs = {}
        for chave, col in (("epocas", "user_attrs_epocas"),
                           ("n_params", "user_attrs_n_params"),
                           ("tempo_s", "user_attrs_tempo_s")):
            v = _num(row.get(col))
            if v is not None:
                user_attrs[chave] = v

        if str(row["state"]).upper() == "COMPLETE":
            trial = optuna.trial.create_trial(
                state=TrialState.COMPLETE, value=float(row["value"]),
                params=params, distributions=DISTRIBUICOES, user_attrs=user_attrs,
            )
            n_ok += 1
        else:
            trial = optuna.trial.create_trial(
                state=TrialState.PRUNED,
                params=params, distributions=DISTRIBUICOES, user_attrs=user_attrs,
            )
            n_podados += 1
        study.add_trial(trial)

    print(f"estudo recriado: {n_ok} completos + {n_podados} podados = {len(study.trials)} trials")
    print(f"melhor RMSE (validacao): {study.best_value:.1f}  (trial {study.best_trial.number})")
    print(f"banco: {DB}")
    print("rode:  optuna-dashboard sqlite:///data/optuna_study.db")


if __name__ == "__main__":
    main()
