"""Engenharia de atributos compartilhada entre treino, API e dashboard.

Prevemos as requisições do intervalo t (15 min) com 1 HORA de antecedência. Por isso todo atributo
de histórico usa apenas informação disponível até t - 1h (config.HORIZONTE passos antes).
Usar a mesma função no treino e na inferência evita divergência (training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

H = config.HORIZONTE
FEATURES_CALENDARIO = ["hora", "dia_semana", "fim_de_semana", "feriado",
                       "hora_sin", "hora_cos", "dia_sin", "dia_cos"]
# nome do atributo -> quantos passos de 15 min antes do alvo
LAGS = {
    "req_1h_atras": H,
    "req_1h15_atras": H + 1,
    "req_1h30_atras": H + 2,
    "req_1h45_atras": H + 3,
    "req_24h_atras": config.PASSOS_DIA,
    "req_7d_atras": config.PASSOS_SEMANA,
}
FEATURES_JANELA = ["media_movel_1h", "media_movel_24h", "desvio_movel_24h"]
FEATURES_CONTEXTO = ["hosts_1h_atras", "taxa_erro_1h_atras", "mb_1h_atras"]
FEATURES_HISTORICO = list(LAGS) + FEATURES_JANELA + FEATURES_CONTEXTO
FEATURES = FEATURES_CALENDARIO + FEATURES_HISTORICO


def calendario(idx: pd.DatetimeIndex) -> pd.DataFrame:
    cal = pd.DataFrame(index=idx)
    fracao_dia = (idx.hour + idx.minute / 60) / 24
    cal["hora"] = idx.hour
    cal["dia_semana"] = idx.dayofweek
    cal["fim_de_semana"] = (idx.dayofweek >= 5).astype(int)
    cal["feriado"] = idx.normalize().isin(pd.to_datetime(config.FERIADOS)).astype(int)
    # Codificação cíclica: 23h45 fica "perto" de 00h00; domingo perto de segunda
    cal["hora_sin"] = np.sin(2 * np.pi * fracao_dia)
    cal["hora_cos"] = np.cos(2 * np.pi * fracao_dia)
    cal["dia_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    cal["dia_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return cal


def criar_features(grade: pd.DataFrame, remover_incompletas: bool = True) -> pd.DataFrame:
    """Recebe a série de 15 min (NaN nas lacunas) e devolve a tabela de modelagem."""
    y = grade[config.TARGET]
    df = calendario(grade.index)
    for nome, passos in LAGS.items():
        df[nome] = y.shift(passos)
    disponivel = y.shift(H)   # último valor conhecido no momento da previsão
    df["media_movel_1h"] = disponivel.rolling(config.PASSOS_HORA, min_periods=config.PASSOS_HORA).mean()
    df["media_movel_24h"] = disponivel.rolling(config.PASSOS_DIA, min_periods=config.PASSOS_DIA).mean()
    df["desvio_movel_24h"] = disponivel.rolling(config.PASSOS_DIA, min_periods=config.PASSOS_DIA).std()
    df["hosts_1h_atras"] = grade["hosts_unicos"].shift(H)
    df["taxa_erro_1h_atras"] = grade["taxa_erro"].shift(H)
    df["mb_1h_atras"] = grade["mb_transferidos"].shift(H)
    df[config.TARGET] = y
    if remover_incompletas:
        # Remove intervalos sem alvo (lacunas) e aqueles cujo histórico cai em uma lacuna
        df = df.dropna(subset=[config.TARGET] + FEATURES_HISTORICO)
    return df[FEATURES + [config.TARGET]]


def features_do_instante(grade: pd.DataFrame, data_hora) -> pd.DataFrame | None:
    """Atributos para prever o intervalo `data_hora` usando a série armazenada.

    Retorna None se o histórico necessário (até 7 dias antes) tiver lacunas.
    """
    alvo = pd.Timestamp(data_hora).floor(config.RESOLUCAO)
    passo = pd.Timedelta(config.RESOLUCAO)
    janela = grade.reindex(pd.date_range(alvo - config.PASSOS_SEMANA * passo, alvo, freq=config.RESOLUCAO))
    linha = criar_features(janela, remover_incompletas=False).loc[[alvo], FEATURES]
    return None if linha.isna().any(axis=None) else linha


def linha_para_previsao(data_hora, valores: dict) -> pd.DataFrame:
    """Monta uma linha de atributos a partir de valores informados manualmente (API)."""
    alvo = pd.Timestamp(data_hora).floor(config.RESOLUCAO)
    linha = calendario(pd.DatetimeIndex([alvo]))
    for col in FEATURES_HISTORICO:
        linha[col] = float(valores[col])
    return linha[FEATURES]
