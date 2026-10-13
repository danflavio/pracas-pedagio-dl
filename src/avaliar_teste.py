"""F7 - Avaliacao final no conjunto de TESTE (uma unica vez).

Protocolo: treina a config vencedora do Optuna no TREINO, usa a VALIDACAO apenas
durante o treino (early stopping) e reporta o TESTE uma unica vez - sem ajustar
nada depois de ver o teste. Compara contra os baselines obrigatorios.

Uso:
    python src/avaliar_teste.py --split chrono
    python src/avaliar_teste.py --split grupo
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as ds   # noqa: E402
import train as tr     # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
EPOCAS = 60
PACIENCIA = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="chrono", choices=["chrono", "grupo"])
    args = ap.parse_args()

    cfg = RESULTS / "config_final.json"
    if cfg.exists():
        base = json.loads(cfg.read_text(encoding="utf-8"))
        base.pop("origem", None)
    else:
        resumo = json.loads((RESULTS / "optuna_resumo.json").read_text(encoding="utf-8"))
        base = dict(resumo["melhores_params"])
    base.update(epochs=EPOCAS, patience=PACIENCIA, seed=ds.GLOBAL_SEED)

    print(f"config final: {base}", flush=True)
    print(f"split: {args.split}", flush=True)

    prep = ds.preparar(window=int(base["window"]), split=args.split)

    # baselines de referencia, no MESMO conjunto de teste
    base_linhas = []
    for nome in ("val", "test"):
        vol = tr.canais_reais(prep[f"X_{nome}"], prep)
        y = prep[f"y_{nome}"]
        for rotulo, pred in (("persistencia", vol[:, -1, ds.TOTAL_IDX]),
                             ("media_movel", vol[:, :, ds.TOTAL_IDX].mean(axis=1))):
            base_linhas.append({"modelo": rotulo, "conjunto": nome, **tr.metricas(y, pred)})

    print("treinando a config final ...", flush=True)
    r = tr.treinar(prep, base, verbose=0)

    linhas = [{"modelo": base["model_type"], "conjunto": "val", **r["metrics_val"]}]
    pred_te = tr.desescalonar(r["model"].predict(prep["X_test"], verbose=0).ravel(), prep)
    linhas.append({"modelo": base["model_type"], "conjunto": "test",
                   **tr.metricas(prep["y_test"], pred_te)})

    tabela = pd.DataFrame(base_linhas + linhas)
    tabela["params"] = r["n_params"]
    tabela["tempo_treino_s"] = round(r["tempo_treino_s"], 1)
    destino = RESULTS / f"teste_final_{args.split}.csv"
    tabela.to_csv(destino, index=False)

    print(f"\nRESULTADO FINAL ({args.split}) - unidade: veiculos", flush=True)
    print(tabela.to_string(index=False, float_format=lambda x: f"{x:,.2f}"), flush=True)
    print(f"\nsalvo em {destino}", flush=True)


if __name__ == "__main__":
    main()
