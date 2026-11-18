"""Regras de capacidade e classificação do risco de saturação."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NIVEIS = ["BAIXO", "MÉDIO", "ALTO"]


def calcular_capacidade(volumes_treino: pd.Series) -> float:
    """Capacidade operacional de referência = percentil 95 do TREINO (sem olhar o teste)."""
    return float(np.quantile(volumes_treino, config.CAPACIDADE_PERCENTIL))


def classificar_risco(volume, capacidade: float):
    """Classifica pela utilização (volume / capacidade).

    BAIXO  : utilização < 70%
    MÉDIO  : 70% <= utilização < 90%
    ALTO   : utilização >= 90%
    """
    util = np.asarray(volume, dtype=float) / capacidade
    risco = np.where(util >= config.LIMIAR_ALTO, "ALTO",
                     np.where(util >= config.LIMIAR_MEDIO, "MÉDIO", "BAIXO"))
    return risco if risco.ndim else str(risco)


def resumo_capacidade(volume_previsto: float, capacidade: float) -> dict:
    util = volume_previsto / capacidade
    return {
        "volume_previsto": round(float(volume_previsto), 1),
        "capacidade_referencia": round(capacidade, 1),
        "capacidade_disponivel": round(float(capacidade - volume_previsto), 1),
        "utilizacao_percentual": round(100 * util, 1),
        "risco_saturacao": classificar_risco(volume_previsto, capacidade),
    }
