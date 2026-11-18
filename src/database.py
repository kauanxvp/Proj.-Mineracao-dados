"""Persistência em SQLite: série tratada, previsões, métricas e histórico de execuções.

SQLite foi escolhido por não exigir servidor (roda direto no Jupyter/Windows).
A troca por PostgreSQL exige apenas mudar a conexão (ex.: SQLAlchemy).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pandas as pd

from . import config


def conectar(caminho=None) -> sqlite3.Connection:
    config.garantir_pastas()
    return sqlite3.connect(caminho or config.DB_PATH)


def salvar_tabela(df: pd.DataFrame, nome: str, caminho=None) -> None:
    with conectar(caminho) as con:
        df.to_sql(nome, con, if_exists="replace", index=False)


def ler_tabela(nome: str, caminho=None) -> pd.DataFrame:
    with conectar(caminho) as con:
        return pd.read_sql(f"SELECT * FROM {nome}", con)


def registrar_execucao(info: dict, caminho=None) -> None:
    with conectar(caminho) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS execucoes (executado_em TEXT, melhor_modelo TEXT,"
            " hash_dados TEXT, detalhes TEXT)")
        con.execute(
            "INSERT INTO execucoes VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), info.get("melhor_modelo"),
             info.get("hash_dados"), json.dumps(info, ensure_ascii=False, default=str)))


def registrar_predicao(origem: str, entrada: dict, resultado: dict, caminho=None) -> int:
    """Grava cada previsão servida pela API, com a versão do modelo (auditoria e histórico)."""
    with conectar(caminho) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS predicoes_api (id INTEGER PRIMARY KEY AUTOINCREMENT, criado_em TEXT,"
            " origem TEXT, previsao_para TEXT, entrada TEXT, volume_previsto REAL, utilizacao_percentual REAL,"
            " risco_saturacao TEXT, volume_real REAL, modelo TEXT, versao_modelo TEXT)")
        cur = con.execute(
            "INSERT INTO predicoes_api (criado_em, origem, previsao_para, entrada, volume_previsto,"
            " utilizacao_percentual, risco_saturacao, volume_real, modelo, versao_modelo)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), origem, resultado.get("previsao_para"),
             json.dumps(entrada, ensure_ascii=False, default=str), resultado.get("volume_previsto"),
             resultado.get("utilizacao_percentual"), resultado.get("risco_saturacao"),
             resultado.get("volume_real"), resultado.get("modelo"), resultado.get("versao_modelo")))
        return int(cur.lastrowid)


def listar_predicoes(limite: int = 20, caminho=None) -> pd.DataFrame:
    with conectar(caminho) as con:
        existe = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predicoes_api'").fetchone()
        if not existe:
            return pd.DataFrame()
        return pd.read_sql("SELECT * FROM predicoes_api ORDER BY id DESC LIMIT ?", con, params=(int(limite),))


def carregar_grade(caminho=None) -> pd.DataFrame:
    """Recupera a série de 15 min tratada (com lacunas) para inferência a partir do histórico."""
    df = ler_tabela("requisicoes_15min", caminho)
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    return df.set_index("data_hora").asfreq(config.RESOLUCAO)
