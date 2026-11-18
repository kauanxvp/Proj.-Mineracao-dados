"""Modelo clássico de série temporal: ARIMA com sazonalidade diária e semanal por termos de Fourier.

Por que não um SARIMA "puro": com intervalos de 15 minutos, o ciclo diário tem 96 passos e o semanal 672.
Um SARIMA com período 96 fica muito lento de ajustar. A alternativa usual (regressão harmônica dinâmica,
Hyndman e Athanasopoulos) é descrever a sazonalidade por senos e cossenos (termos de Fourier) e deixar o ARIMA
modelar o que sobra (a autocorrelação de curto prazo).

O modelo segue a interface do scikit-learn para entrar na mesma comparação (GridSearchCV + TimeSeriesSplit).
Diferente dos outros, ele não usa as colunas de X como atributos: usa a própria série. O índice de X traz os
instantes, e a coluna req_1h_atras traz o valor observado 1 hora antes de cada alvo, o que permite reconstruir
o histórico conhecido no momento de cada previsão. A previsão para t usa apenas dados até t - 1h.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from . import config

PASSO = pd.Timedelta(config.RESOLUCAO)


def termos_fourier(idx: pd.DatetimeIndex, k_diario: int, k_semanal: int) -> pd.DataFrame:
    """Senos e cossenos com período de 1 dia e de 1 semana, mais constante e feriado."""
    t = np.asarray((idx - pd.Timestamp("1995-01-02")) / PASSO, dtype=float)  # 02/01/1995 foi uma segunda
    colunas = {"constante": np.ones(len(idx))}
    for periodo, k_max, nome in [(config.PASSOS_DIA, k_diario, "dia"), (config.PASSOS_SEMANA, k_semanal, "sem")]:
        for k in range(1, k_max + 1):
            colunas[f"sin_{nome}_{k}"] = np.sin(2 * np.pi * k * t / periodo)
            colunas[f"cos_{nome}_{k}"] = np.cos(2 * np.pi * k * t / periodo)
    colunas["feriado"] = idx.normalize().isin(pd.to_datetime(config.FERIADOS)).astype(float)
    return pd.DataFrame(colunas, index=idx)


class ArimaFourier(RegressorMixin, BaseEstimator):
    """ARIMA(p, 0, q) + Fourier, ajustado por máxima verossimilhança (statsmodels SARIMAX)."""

    def __init__(self, p: int = 2, q: int = 1, k_diario: int = 6, k_semanal: int = 3):
        self.p = p
        self.q = q
        self.k_diario = k_diario
        self.k_semanal = k_semanal

    def _exog(self, idx):
        return termos_fourier(idx, self.k_diario, self.k_semanal)

    def fit(self, X, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        serie = pd.Series(np.asarray(y, dtype=float), index=X.index).asfreq(config.RESOLUCAO)  # NaN nas lacunas
        modelo = SARIMAX(serie, exog=self._exog(serie.index), order=(self.p, 0, self.q), trend="n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.resultado_ = modelo.fit(disp=False, maxiter=200)
        self.serie_treino_ = serie
        return self

    def predict(self, X):
        h = config.HORIZONTE
        alvos = pd.DatetimeIndex(X.index)
        origens = alvos - h * PASSO
        # Histórico conhecido: o treino + o valor observado 1 h antes de cada alvo (coluna req_1h_atras)
        observado = pd.Series(np.asarray(X["req_1h_atras"], dtype=float), index=origens)
        historico = self.serie_treino_.combine_first(observado[~observado.index.duplicated()])
        grade = pd.date_range(historico.index.min(), alvos.max(), freq=config.RESOLUCAO)
        endog = historico.reindex(grade)
        endog[endog.index > origens.max()] = np.nan   # nada depois da última origem entra no filtro
        exog = self._exog(grade)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtrado = self.resultado_.apply(endog, exog=exog)
        # Kalman: estado filtrado na origem (usa dados até t - 1h) projetado h passos à frente
        T = filtrado.model.ssm["transition", :, :, 0]
        Z = filtrado.model.ssm["design", :, :, 0]
        Th = np.linalg.matrix_power(T, h)
        pos = grade.get_indexer(origens)
        estados = filtrado.filtered_state[:, pos]
        beta = np.asarray(filtrado.params[exog.columns])
        return (Z @ Th @ estados).ravel() + exog.loc[alvos].to_numpy() @ beta
