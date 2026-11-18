"""Definição dos modelos comparados."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import FEATURES
from .serie_temporal import ArimaFourier


class BaselineSazonal(RegressorMixin, BaseEstimator):
    """Modelo de série temporal ingênuo: repete o valor de um lag.

    req_1h_atras  = persistência: daqui a 1h a carga será igual à de agora.
    req_24h_atras = sazonal diário: igual ao mesmo horário de ontem.
    req_7d_atras  = sazonal semanal: igual ao mesmo horário da semana passada.
    Serve como referência mínima: um modelo de ML só vale a pena se superá-lo.
    """

    def __init__(self, coluna: str = "req_24h_atras"):
        self.coluna = coluna

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X[self.coluna], dtype=float)


def _preprocessador() -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), FEATURES)])


def _pipe(modelo) -> Pipeline:
    return Pipeline([("prep", _preprocessador()), ("modelo", modelo)])


def _grade(nome: str, prefixo: str) -> dict:
    """Lê a grade de hiperparâmetros do params.yaml (listas viram tuplas, como pede o scikit-learn)."""
    grade = config.PARAMS["modelos"][nome]["grade"]
    return {prefixo + k: [tuple(v) if isinstance(v, list) else v for v in valores] for k, valores in grade.items()}


def obter_modelos() -> dict:
    """Modelos comparados + grade de hiperparâmetros testada por validação cruzada temporal (params.yaml)."""
    rs = config.RANDOM_STATE
    p = config.PARAMS["modelos"]
    return {
        "Baseline: persistência (agora)": (BaselineSazonal("req_1h_atras"), {}),
        "Baseline: sazonal diário (ontem)": (BaselineSazonal("req_24h_atras"), {}),
        "Baseline: sazonal semanal (semana passada)": (BaselineSazonal("req_7d_atras"), {}),
        # Série temporal clássica: ARIMA + sazonalidade diária e semanal por termos de Fourier
        "ARIMA + Fourier": (ArimaFourier(), _grade("arima_fourier", "")),
        "Regressão Linear": (_pipe(LinearRegression()), {}),
        "Random Forest": (
            _pipe(RandomForestRegressor(n_estimators=p["random_forest"]["n_estimators"], n_jobs=-1,
                                        random_state=rs)),
            _grade("random_forest", "modelo__")),
        "Gradient Boosting": (
            _pipe(HistGradientBoostingRegressor(max_iter=p["gradient_boosting"]["max_iter"], random_state=rs)),
            _grade("gradient_boosting", "modelo__")),
        # Rede Neural: perceptron multicamadas; o alvo também é padronizado.
        "Rede Neural (MLP)": (
            TransformedTargetRegressor(
                regressor=_pipe(MLPRegressor(max_iter=p["rede_neural"]["max_iter"], early_stopping=True,
                                             n_iter_no_change=20, learning_rate_init=1e-3, random_state=rs)),
                transformer=StandardScaler()),
            _grade("rede_neural", "regressor__modelo__")),
    }
