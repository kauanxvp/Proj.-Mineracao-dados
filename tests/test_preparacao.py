"""Testes unitários: rodam sem a base real e sem modelo treinado."""
import gzip

import numpy as np
import pandas as pd

from src import capacity, clustering, config, data
from src.features import FEATURES, LAGS, criar_features, features_do_instante
from src.pipeline import divisao_temporal


def _bruto(inicio="1995-07-03 00:00", horas=24 * 9, por_intervalo=10):
    """Logs sintéticos: `por_intervalo` requisições em cada intervalo de 15 min."""
    intervalos = pd.date_range(inicio, periods=horas * 4, freq="15min")
    ts = np.repeat(intervalos, por_intervalo) + pd.to_timedelta(np.tile(np.arange(por_intervalo), len(intervalos)), "s")
    n = len(ts)
    return pd.DataFrame({"host": [f"h{i % 7}" for i in range(n)], "data_hora": ts, "metodo": "GET",
                         "url": "/", "status": np.where(np.arange(n) % 10 == 0, 404, 200).astype("int16"),
                         "bytes": 1000})


def _grade_sintetica(dias=9):
    idx = pd.date_range("1995-07-03", periods=dias * 96, freq="15min")
    g = pd.DataFrame({"requisicoes": np.arange(len(idx), dtype=float), "hosts_unicos": 5.0,
                      "taxa_erro": 0.0, "mb_transferidos": 1.0}, index=idx)
    g.index.name = "data_hora"
    return g


def test_agregacao_conta_requisicoes_clientes_e_erros():
    g = data.agregar(_bruto())
    validos = g.dropna()
    assert (validos["requisicoes"] == 10).all()
    assert (validos["hosts_unicos"] == 7).all()
    assert np.isclose(validos["taxa_erro"].mean(), 10.0)


def test_lacuna_longa_vira_nan_e_zero_isolado_continua_zero():
    b = _bruto()
    dia2 = b["data_hora"].dt.floor("15min")
    fora = (dia2 >= "1995-07-05 10:00") & (dia2 < "1995-07-05 13:00")     # 3h fora do ar
    isolado = dia2 == "1995-07-06 03:00"                                    # 15 min sem acesso
    g = data.agregar(b[~(fora | isolado)])
    assert g.loc["1995-07-05 10:00":"1995-07-05 12:45", "requisicoes"].isna().all()
    assert g.loc["1995-07-05 10:00", "fora_do_ar"] == 1
    assert g.loc["1995-07-06 03:00", "requisicoes"] == 0
    lac = data.encontrar_lacunas(data.agregar(b[~fora])["requisicoes"].fillna(0))
    assert len(lac) == 1 and lac.iloc[0]["horas"] == 3


def test_leitura_do_log_original(tmp_path):
    linhas = [
        'in24.inetnebr.com - - [01/Aug/1995:00:00:01 -0400] "GET /shuttle/missions/sts-68/news/sts-68-mcc-05.txt HTTP/1.0" 200 1839\n',
        'uplherc.upl.com - - [01/Aug/1995:00:00:07 -0400] "GET / HTTP/1.0" 304 0\n',
        'uplherc.upl.com - - [01/Aug/1995:00:00:08 -0400] "GET /images/ksclogo-medium.gif HTTP/1.0" 304 -\n',
    ]
    arq = tmp_path / "NASA_access_log_Aug95.gz"
    with gzip.open(arq, "wt") as f:
        f.writelines(linhas)
    df = data.carregar_bruto([arq])
    assert len(df) == 3
    assert df["data_hora"].iloc[0] == pd.Timestamp("1995-08-01 00:00:01")
    assert df["bytes"].tolist() == [1839, 0, 0]
    assert df["status"].tolist() == [200, 304, 304]


def test_lags_respeitam_o_horizonte_de_1h():
    tab = criar_features(_grade_sintetica())
    for nome, passos in LAGS.items():
        assert (tab["requisicoes"] - tab[nome] == passos).all(), nome
    # médias móveis terminam 1h antes do alvo
    assert (tab["media_movel_1h"] == tab["requisicoes"] - config.HORIZONTE - 1.5).all()


def test_sem_vazamento_do_futuro():
    g = _grade_sintetica()
    alvo = g.index[-1]
    antes = features_do_instante(g, alvo)
    g2 = g.copy()
    g2.iloc[-config.HORIZONTE:] = 99999          # altera a última hora (ainda não observada)
    depois = features_do_instante(g2, alvo)
    pd.testing.assert_frame_equal(antes, depois)


def test_inferencia_usa_os_mesmos_atributos_do_treino():
    g = _grade_sintetica()
    tab = criar_features(g)
    alvo = tab.index[-10]
    linha = features_do_instante(g, alvo)
    np.testing.assert_allclose(linha.values[0], tab.loc[alvo, FEATURES].values.astype(float))


def test_divisao_temporal_sem_vazamento():
    tab = criar_features(_grade_sintetica())
    treino, teste = divisao_temporal(tab)
    assert treino.index.max() < teste.index.min()


def test_classificacao_de_risco():
    assert capacity.classificar_risco(500, 1000) == "BAIXO"
    assert capacity.classificar_risco(700, 1000) == "MÉDIO"
    assert capacity.classificar_risco(900, 1000) == "ALTO"
    r = capacity.resumo_capacidade(800, 1000)
    assert r["capacidade_disponivel"] == 200 and r["utilizacao_percentual"] == 80.0


def test_escolha_de_k_por_parcimonia():
    aval = pd.DataFrame({"k": [2, 3, 4], "silhouette": [0.40, 0.520, 0.522], "davies_bouldin": [1, 1, 1]})
    assert clustering.melhor_k(aval) == 3
