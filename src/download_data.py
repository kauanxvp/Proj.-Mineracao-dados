"""Tenta baixar os logs originais da NASA (Internet Traffic Archive) para data/raw.

Uso: python -m src.download_data

O arquivo oficial é distribuído por FTP. Se o download falhar (rede/firewall), baixe manualmente
do Kaggle o dataset "NASA access log dataset 1995" e coloque o data.csv (ou o .zip) em data/raw.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import config  # noqa: E402


def baixar() -> list[Path]:
    config.garantir_pastas()
    try:
        from .data import localizar_fontes
        existentes = localizar_fontes()
        print("Base já existe:", [p.name for p in existentes])
        return existentes
    except FileNotFoundError:
        pass
    salvos = []
    for url in config.URLS_OFICIAIS:
        destino = config.DATA_RAW_DIR / url.rsplit("/", 1)[-1]
        print(f"Baixando {url} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(destino, "wb") as f:
                f.write(resp.read())
            salvos.append(destino)
            print(f"  salvo em {destino}")
        except Exception as erro:  # noqa: BLE001
            print(f"  falhou ({erro}). Use o arquivo do Kaggle, conforme o README.")
            destino.unlink(missing_ok=True)
            break
    return salvos


if __name__ == "__main__":
    baixar()
