"""Testes de integração: exigem que o pipeline já tenha sido executado com a base real."""
import json
import shutil

import pandas as pd
import pytest

from src import config

pytestmark = pytest.mark.skipif(not config.MODEL_PATH.exists(),
                                reason="Execute antes: python -m src.pipeline")

CORPO = {"data_hora_alvo": "1995-08-30T14:00:00", "requisicoes_agora": 900, "requisicoes_15min_antes": 870,
         "requisicoes_30min_antes": 850, "requisicoes_45min_antes": 820,
         "requisicoes_ontem_mesmo_horario": 980, "requisicoes_semana_passada_mesmo_horario": 950,
         "media_ultima_hora": 860, "media_ultimas_24h": 550, "desvio_ultimas_24h": 260,
         "clientes_distintos_agora": 130}


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    """API apontando para uma cópia do banco, para os testes não sujarem o histórico real."""
    from fastapi.testclient import TestClient
    from api.app import app

    copia = tmp_path / "trafego.db"
    shutil.copy(config.DB_PATH, copia)
    monkeypatch.setattr(config, "DB_PATH", copia)
    return TestClient(app)


def test_saidas_geradas():
    for arquivo in ["metrics.csv", "metrics.json", "cv_metrics.csv", "quality_report.json", "trafego.db"]:
        assert (config.OUTPUTS_DIR / arquivo).exists(), arquivo
    assert len(list(config.FIGURES_DIR.glob("*.png"))) >= 15
    assert config.MODEL_CARD_PATH.exists()


def test_modelos_superam_as_baselines():
    m = pd.read_csv(config.OUTPUTS_DIR / "metrics.csv").set_index("modelo")
    ml = m[~m.index.str.startswith("Baseline")]["RMSE"]
    baseline = m[m.index.str.startswith("Baseline")]["RMSE"].min()
    assert (ml < baseline).all()


def test_metricas_do_relatorio_batem_com_o_model_card():
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    m = json.loads((config.OUTPUTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert m["modelo"] == card["modelo"] and m["RMSE"] == card["metricas_teste"]["RMSE"]
    assert card["metricas_teste"]["R2"] > 0.80          # critério de sucesso da seção 1


def test_api(cliente):
    assert cliente.get("/health").json()["status"] == "ok"
    info = cliente.get("/model/info")
    assert info.status_code == 200 and "metricas_teste" in info.json()
    r = cliente.post("/prever", json=CORPO)
    assert r.status_code == 200
    assert r.json()["risco_saturacao"] in {"BAIXO", "MÉDIO", "ALTO"}
    r2 = cliente.get("/prever/historico", params={"data_hora_alvo": "1995-08-31T11:00:00"})
    assert r2.status_code == 200 and r2.json()["volume_real"] is not None
    r3 = cliente.get("/prever/historico", params={"data_hora_alvo": "1995-08-05T11:00:00"})
    assert r3.status_code == 422      # histórico cai na queda do furacão Erin


def test_api_rejeita_entrada_invalida(cliente):
    assert cliente.post("/prever", json={**CORPO, "requisicoes_agora": -5}).status_code == 422
    sem_campo = {k: v for k, v in CORPO.items() if k != "media_ultima_hora"}
    assert cliente.post("/prever", json=sem_campo).status_code == 422


def test_api_grava_cada_previsao_com_a_versao_do_modelo(cliente):
    versao = cliente.get("/model/info").json()["versao"]
    cliente.get("/prever/historico", params={"data_hora_alvo": "1995-08-31T11:00:00"})
    cliente.post("/prever", json=CORPO)
    r = cliente.get("/predicoes", params={"limite": 2})
    assert r.status_code == 200
    ultima, anterior = r.json()
    assert ultima["origem"] == "/prever" and ultima["versao_modelo"] == versao
    assert ultima["volume_real"] is None and anterior["volume_real"] is not None
