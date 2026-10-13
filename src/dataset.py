"""F3 - Serie multivariada por praca + janelas deslizantes + dois splits.

O que este modulo faz, em ordem:
  1. agrega o bruto para `praca x dia`, com canais de volume e features de calendario;
  2. filtra pracas com historico suficiente (decisao: >= 180 dias);
  3. marca o que e observado (serie pode ter buracos no meio - ver eda_cobertura);
  4. monta janelas deslizantes: X = W dias x canais, y = Total do dia SEGUINTE;
  5. aplica os dois protocolos de split: cronologico global e por grupos (pracas);
  6. padroniza os canais de volume usando estatisticas SOMENTE do treino.

Nada aqui embaralha a ordem temporal. O alvo de uma janela de treino e sempre
anterior ao alvo de qualquer janela de validacao/teste (checado em testes_rapidos).

Uso:
    python src/dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PARQUET = BASE / "data" / "trafego_pedagio.parquet"
SERIE = BASE / "data" / "series_praca_dia.parquet"

# --- Decisoes de desenho (validadas em docs/decisoes_dados.md) ---
GLOBAL_SEED = 42
MIN_DIAS = 180                       # praça precisa de ao menos 180 dias observados
TIPOS = ["Passeio", "Comercial", "Moto"]        # D-004: "Veiculo Pequeno" excluido
CANAIS_VOL = TIPOS + ["Total"]                  # canais de volume (o Total e o alvo defasado)
CORTE_TREINO = pd.Timestamp("2025-11-30")       # split cronologico
CORTE_VAL = pd.Timestamp("2026-04-30")
PROP_GRUPO = (0.70, 0.15, 0.15)                 # split por grupos (pracas)
TOTAL_IDX = CANAIS_VOL.index("Total")


# --------------------------------------------------------------------------- #
# 1) Construcao da serie praca x dia
# --------------------------------------------------------------------------- #
def construir_serie(min_dias: int = MIN_DIAS) -> pd.DataFrame:
    """Agrega o bruto para `praca x dia` e adiciona os canais de calendario."""
    df = pd.read_parquet(PARQUET)
    df["data"] = pd.to_datetime(df["mes_ano"], format="%d/%m/%Y", errors="raise")
    df["praca_id"] = df["concessionaria"].astype(str) + " | " + df["praca"].astype(str)

    # filtra pracas com historico suficiente
    dias_por_praca = df.groupby("praca_id")["data"].nunique()
    pracas_ok = dias_por_praca[dias_por_praca >= min_dias].index
    df = df[df["praca_id"].isin(pracas_ok)]
    df = df[df["tipo_de_veiculo"].isin(TIPOS)]      # D-004

    # soma sobre sentido / cobranca / categoria_eixo -> uma linha por (praca, dia, tipo)
    longo = df.groupby(["praca_id", "data", "tipo_de_veiculo"], as_index=False)["volume_total"].sum()
    largo = longo.pivot(index=["praca_id", "data"], columns="tipo_de_veiculo", values="volume_total")
    largo = largo.reindex(columns=TIPOS)
    # dia observado SEM aquele tipo = volume 0 (as combinacoes ausentes nao existem no bruto)
    largo = largo.fillna(0.0).reset_index()

    # reconstroi o calendario diario de cada praca (decisoes 3 e 4)
    partes = []
    for pid, g in largo.groupby("praca_id", sort=False):
        idx = pd.date_range(g["data"].min(), g["data"].max(), freq="D")
        g = g.set_index("data").reindex(idx)
        g["praca_id"] = pid
        g["data"] = idx
        # observado = o dia tinha ao menos uma linha no bruto (antes do fillna)
        g["observado"] = g[TIPOS].notna().any(axis=1)
        partes.append(g.reset_index(drop=True))
    serie = pd.concat(partes, ignore_index=True)

    serie["Total"] = serie[TIPOS].sum(axis=1, skipna=False)  # NaN quando o dia nao existe

    # features de calendario (D-007)
    dow = serie["data"].dt.dayofweek
    for i in range(7):
        serie[f"dow_{i}"] = (dow == i).astype("float32")
    serie["mes_sin"] = np.sin(2 * np.pi * serie["data"].dt.month / 12).astype("float32")
    serie["mes_cos"] = np.cos(2 * np.pi * serie["data"].dt.month / 12).astype("float32")

    serie = serie.sort_values(["praca_id", "data"]).reset_index(drop=True)
    return serie


def carregar_serie(min_dias: int = MIN_DIAS) -> pd.DataFrame:
    """Le a serie cacheada (data/series_praca_dia.parquet); se nao existir, constroi.

    O cache importa: a F5 roda dezenas de trials, e reconstruir a serie a cada
    trial custaria minutos sem necessidade.
    """
    if SERIE.exists():
        return pd.read_parquet(SERIE)
    return construir_serie(min_dias)


def colunas_calendario() -> list[str]:
    return [f"dow_{i}" for i in range(7)] + ["mes_sin", "mes_cos"]


# --------------------------------------------------------------------------- #
# 2) Janelas deslizantes (sem embaralhar)
# --------------------------------------------------------------------------- #
def montar_janelas(serie: pd.DataFrame, window: int) -> dict:
    """Gera X (n, window, n_features), y (n,) = Total do dia seguinte.

    Regra (decisao 3): a janela so e valida se TODOS os dias - entrada e alvo -
    estao observados. Assim nunca preenchemos buraco com dado inventado.
    """
    cols = CANAIS_VOL + colunas_calendario()
    Xs, ys, pids, datas = [], [], [], []
    for pid, g in serie.groupby("praca_id", sort=False):
        vol = g[CANAIS_VOL].to_numpy("float32")
        cal = g[colunas_calendario()].to_numpy("float32")
        feat = np.hstack([vol, cal])                 # (T, n_features)
        obs = g["observado"].to_numpy()
        dt = g["data"].to_numpy()
        for i in range(len(g) - window):
            j = i + window                            # indice do alvo (dia seguinte)
            if not obs[i:j + 1].all():
                continue
            Xs.append(feat[i:j])
            ys.append(vol[j, TOTAL_IDX])
            pids.append(pid)
            datas.append(dt[j])
    return {
        "X": np.stack(Xs).astype("float32"),
        "y": np.array(ys, dtype="float32"),
        "praca_id": np.array(pids),
        "data_alvo": pd.to_datetime(np.array(datas)),
        "window": window,
        "features": cols,
    }


# --------------------------------------------------------------------------- #
# 3) Splits
# --------------------------------------------------------------------------- #
def mascara_cronologica(data_alvo: pd.Series) -> dict:
    treino = data_alvo <= CORTE_TREINO
    val = (data_alvo > CORTE_TREINO) & (data_alvo <= CORTE_VAL)
    teste = data_alvo > CORTE_VAL
    return {"train": treino.to_numpy(), "val": val.to_numpy(), "test": teste.to_numpy()}


def mascara_grupos(praca_id: np.ndarray, seed: int = GLOBAL_SEED) -> dict:
    rng = np.random.default_rng(seed)
    pracas = np.array(sorted(set(praca_id)))
    rng.shuffle(pracas)
    n = len(pracas)
    n_tr = int(PROP_GRUPO[0] * n)
    n_va = int(PROP_GRUPO[1] * n)
    g_train = set(pracas[:n_tr])
    g_val = set(pracas[n_tr:n_tr + n_va])
    return {
        "train": np.isin(praca_id, list(g_train)),
        "val": np.isin(praca_id, list(g_val)),
        "test": np.isin(praca_id, list(pracas[n_tr + n_va:])),
    }


# --------------------------------------------------------------------------- #
# 4) Padronizacao (fit SOMENTE no treino)
# --------------------------------------------------------------------------- #
def padronizar(X: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, :, :len(CANAIS_VOL)] = (X[:, :, :len(CANAIS_VOL)] - media) / desvio
    return X


def preparar(window: int, split: str = "chrono") -> dict:
    """Devolve os conjuntos prontos para treino, com scaler ajustado no treino."""
    serie = carregar_serie()
    jan = montar_janelas(serie, window)
    X, y = jan["X"], jan["y"]

    if split == "chrono":
        m = mascara_cronologica(pd.Series(jan["data_alvo"]))
    elif split == "grupo":
        m = mascara_grupos(jan["praca_id"])
    else:
        raise ValueError("split deve ser 'chrono' ou 'grupo'")

    # estatisticas dos canais de volume, calculadas SO no treino.
    # float64 no calculo: acumular ~3 milhoes de valores em float32 gera deriva (~1e-3).
    vol_tr = (X[m["train"]][:, :, :len(CANAIS_VOL)]
              .reshape(-1, len(CANAIS_VOL)).astype("float64"))
    media = vol_tr.mean(axis=0)
    desvio = vol_tr.std(axis=0)
    desvio = np.where(desvio == 0, 1.0, desvio)

    saida = {"features": jan["features"], "window": window, "split": split,
             "media": media, "desvio": desvio}
    for nome in ("train", "val", "test"):
        saida[f"X_{nome}"] = padronizar(X[m[nome]], media, desvio)
        saida[f"y_{nome}"] = y[m[nome]]
        saida[f"praca_{nome}"] = jan["praca_id"][m[nome]]
        saida[f"data_{nome}"] = pd.Series(jan["data_alvo"])[m[nome]].to_numpy()
    return saida


def testes_rapidos(prep: dict) -> None:
    """Checagens que o plano exige: sem vazamento temporal, sem shuffle, scaler no treino."""
    print("-" * 72)
    print("TESTES RAPIDOS")
    d_tr, d_va, d_te = (pd.to_datetime(prep[f"data_{n}"]).max() for n in ("train", "val", "test"))
    t_va, t_te = (pd.to_datetime(prep[f"data_{n}"]).min() for n in ("val", "test"))
    print("max(data alvo) treino:", d_tr.date(), "| min val:", t_va.date(), "->",
          "OK" if d_tr < t_va else "FALHOU")
    print("max(data alvo) val ...:", d_va.date(), "| min teste:", t_te.date(), "->",
          "OK" if d_va < t_te else "FALHOU")
    vol_tr = prep["X_train"][:, :, :len(CANAIS_VOL)].reshape(-1, len(CANAIS_VOL))
    med = vol_tr.mean(axis=0)
    des = vol_tr.std(axis=0)
    print("media pos-scaler (treino):", np.round(med, 6).tolist())
    print("  maxima |media|:", float(np.abs(med).max()))
    print("desvio pos-scaler (treino):", np.round(des, 6).tolist())
    print("media pos-scaler ~0:", np.allclose(med, 0, atol=1e-3))
    print("desvio pos-scaler ~1:", np.allclose(des, 1, atol=1e-2))
    print(f"X_train {prep['X_train'].shape} | X_val {prep['X_val'].shape} | X_test {prep['X_test'].shape}")


def main() -> None:
    serie = construir_serie()
    serie.to_parquet(SERIE, index=False)
    print("serie gravada:", SERIE)
    print("linhas:", len(serie), "| pracas:", serie["praca_id"].nunique())
    print("dias observados:", int(serie["observado"].sum()), "| faltantes:",
          int((~serie["observado"]).sum()))
    print("canais:", CANAIS_VOL + colunas_calendario())

    prep = preparar(window=30, split="chrono")
    testes_rapidos(prep)

    prep_g = preparar(window=30, split="grupo")
    print("-" * 72)
    print("split por GRUPOS (pracas):",
          f"train={prep_g['praca_train'].shape[0]} /",
          f"val={prep_g['praca_val'].shape[0]} /",
          f"test={prep_g['praca_test'].shape[0]} janelas")


if __name__ == "__main__":
    main()
