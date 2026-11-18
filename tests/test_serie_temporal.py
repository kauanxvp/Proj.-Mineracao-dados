"""Testes do modelo ARIMA + Fourier: previsão com 1 h de antecedência, sem usar dados do futuro."""
import numpy as np
import pandas as pd

from src import config
from src.features import FEATURES, criar_features
from src.pipeline import divisao_temporal
from src.serie_temporal import ArimaFourier, termos_fourier


def _tabela(dias=14, semente=0):
    idx = pd.date_range("1995-07-03", periods=dias * 96, freq="15min")
    rng = np.random.default_rng(semente)
    ciclo = 500 + 300 * np.sin(2 * np.pi * np.arange(len(idx)) / 96)
    g = pd.DataFrame({"requisicoes": ciclo + rng.normal(0, 20, len(idx)), "hosts_unicos": 50.0,
                      "taxa_erro": 1.0, "mb_transferidos": 10.0}, index=idx)
    g.index.name = "data_hora"
    return criar_features(g)


def test_termos_fourier_tem_periodo_de_um_dia():
    idx = pd.date_range("1995-07-03", periods=2 * 96, freq="15min")
    f = termos_fourier(idx, k_diario=2, k_semanal=1)
    assert np.allclose(f["sin_dia_1"].iloc[:96].to_numpy(), f["sin_dia_1"].iloc[96:].to_numpy())
    assert {"constante", "feriado", "cos_sem_1"} <= set(f.columns)


def test_arima_aprende_o_ciclo_diario():
    treino, teste = divisao_temporal(_tabela())
    m = ArimaFourier(p=1, q=0, k_diario=2, k_semanal=1).fit(treino[FEATURES], treino[config.TARGET])
    erro = np.abs(m.predict(teste[FEATURES]) - teste[config.TARGET]).mean()
    assert erro < 60   # o ruído tem desvio 20; sem aprender o ciclo o erro passaria de 150


def test_arima_nao_usa_dados_do_futuro():
    treino, teste = divisao_temporal(_tabela())
    m = ArimaFourier(p=1, q=0, k_diario=2, k_semanal=1).fit(treino[FEATURES], treino[config.TARGET])
    X = teste[FEATURES]
    corte = X.index[len(X) // 2]
    X2 = X.copy()
    X2.loc[X2.index > corte, "req_1h_atras"] *= 5      # altera só observações posteriores ao corte - 1h
    antes = X.index <= corte
    assert np.allclose(m.predict(X)[antes], m.predict(X2)[antes])
