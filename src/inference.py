"""Carrega o modelo treinado e gera previsões com classificação de risco."""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

from . import capacity, config, database
from .features import features_do_instante, linha_para_previsao


@lru_cache(maxsize=1)
def carregar_modelo() -> dict:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute antes: python -m src.pipeline")
    return joblib.load(config.MODEL_PATH)


def _resultado(alvo, X: pd.DataFrame) -> dict:
    b = carregar_modelo()
    volume = max(0.0, float(b["modelo"].predict(X)[0]))
    resumo = capacity.resumo_capacidade(volume, b["capacidade_referencia"])
    return {
        "previsao_para": str(alvo),
        "gerada_em": str(alvo - pd.Timedelta(minutes=b["horizonte_min"])),
        "requisicoes_agora": float(X["req_1h_atras"].iloc[0]),
        "clientes_agora": float(X["hosts_1h_atras"].iloc[0]),
        **resumo,
        "volume_previsto_req_s": round(volume / 900, 2),
        "modelo": b["nome_modelo"],
        "versao_modelo": b.get("versao"),
    }


def prever(data_hora_alvo, valores: dict) -> dict:
    """Previsão a partir de valores informados (API). `valores` usa os nomes de features.FEATURES_HISTORICO."""
    alvo = pd.Timestamp(data_hora_alvo).floor(config.RESOLUCAO)
    return _resultado(alvo, linha_para_previsao(alvo, valores))


def prever_pelo_historico(data_hora_alvo, grade: pd.DataFrame | None = None) -> dict:
    """Previsão para um intervalo do histórico salvo no SQLite, comparando com o valor real."""
    grade = grade if grade is not None else database.carregar_grade()
    alvo = pd.Timestamp(data_hora_alvo).floor(config.RESOLUCAO)
    X = features_do_instante(grade, alvo)
    if X is None:
        raise ValueError(f"Histórico incompleto para {alvo}: há lacuna nos 7 dias anteriores.")
    r = _resultado(alvo, X)
    real = grade[config.TARGET].get(alvo)
    r["volume_real"] = None if real is None or pd.isna(real) else float(real)
    if r["volume_real"] is not None:
        r["risco_real"] = capacity.classificar_risco(r["volume_real"], carregar_modelo()["capacidade_referencia"])
    return r
