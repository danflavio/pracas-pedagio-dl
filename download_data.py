"""Baixa os arquivos de volume de trafego (ANTT) e registra a proveniencia.

O script faz o download dos 3 arquivos anuais (2024, 2025, 2026) do Google Drive
para data/raw/ e grava um manifesto (data_manifest.json) com URL de origem, data
de download, tamanho e SHA-256 de cada arquivo - para reprodutibilidade.

Uso:
    python download_data.py            # baixa o que ainda nao existe
    python download_data.py --force    # re-baixa tudo e atualiza o manifesto
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gdown

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MANIFEST = BASE_DIR / "data_manifest.json"

# Fonte: ANTT - Volume de Trafego nas Pracas de Pedago, dados.gov.br
# Copia compartilhada entre os integrantes: Google Drive.
ARQUIVOS = [
    {"ano": 2024, "file_id": "1vUHJ-L4f9kRrRmRa5qlfdb68jSn5oOqN",
     "nome": "volume_pedagio_2024.csv"},
    {"ano": 2025, "file_id": "1hDvgQXU4UvhpE3dfSUxx72fO3Phter3t",
     "nome": "volume_pedagio_2025.csv"},
    {"ano": 2026, "file_id": "1ej0hH6aIV3HvMaoDd9jXfSZJd2hF8lzw",
     "nome": "volume_pedagio_2026.csv"},
]


def sha256(caminho: Path, chunk: int = 1 << 20) -> str:
    """Calcula o SHA-256 de um arquivo lendo em blocos (nao carrega tudo na RAM)."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(chunk), b""):
            h.update(bloco)
    return h.hexdigest()


def baixar(info: dict, force: bool = False) -> Path:
    """Baixa um arquivo do Google Drive para data/raw/ (pula se ja existir)."""
    destino = RAW_DIR / info["nome"]
    if destino.exists() and destino.stat().st_size > 0 and not force:
        print(f"[skip]    {destino.name} ja existe")
        return destino

    url = f"https://drive.google.com/uc?id={info['file_id']}"
    print(f"[baixar]  {destino.name} ...")
    gdown.download(url, str(destino), quiet=False)

    if not destino.exists() or destino.stat().st_size == 0:
        raise RuntimeError(f"falhou o download de {info['nome']}")
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description="Baixa os CSVs de trafego e registra o manifesto.")
    ap.add_argument("--force", action="store_true", help="re-baixa mesmo se o arquivo existir")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    registros = []

    for info in ARQUIVOS:
        caminho = baixar(info, force=args.force)
        registros.append({
            "ano": info["ano"],
            "arquivo": info["nome"],
            "caminho": str(caminho.relative_to(BASE_DIR)).replace("\\", "/"),
            "url_origem": f"https://drive.google.com/file/d/{info['file_id']}/view",
            "bytes": caminho.stat().st_size,
            "sha256": sha256(caminho),
            "baixado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"          size={registros[-1]['bytes']} sha256={registros[-1]['sha256'][:12]}...")

    MANIFEST.write_text(json.dumps(registros, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n[ok] manifesto gravado em {MANIFEST.name}")


if __name__ == "__main__":
    main()
