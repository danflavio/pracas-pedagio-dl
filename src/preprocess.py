"""Une os 3 CSVs anuais (ANTT) e grava o dataset derivado em Parquet.

Uso:
    python src/preprocess.py
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]   # raiz do projeto
RAW_DIR = BASE_DIR / "data" / "raw"
SAIDA = BASE_DIR / "data" / "trafego_pedagio.parquet"

ANOS = [2024, 2025, 2026]

# TODO: complete os parametros de leitura decididos na auditoria (D-001, D-002, D-006)
LEITURA = dict(
    sep=";",
    encoding="latin-1",
    decimal=",",
    dtype={"categoria_eixo": "str"},
)


def build_dataset() -> pd.DataFrame:
    """Le os 3 CSVs, adiciona ano_origem, concatena e grava em Parquet."""
    frames = []
    for ano in ANOS:
        arq = RAW_DIR / f"volume_pedagio_{ano}.csv"
        df = pd.read_csv(arq, **LEITURA)
        df["ano_origem"] = ano
        frames.append(df)

    dados = pd.concat(frames, ignore_index=True)
    dados.to_parquet(SAIDA, index=False)
    return dados


if __name__ == "__main__":
    dados = build_dataset()
    print("gravado:", SAIDA)
    print("shape:", dados.shape)