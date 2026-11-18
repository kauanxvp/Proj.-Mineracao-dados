"""API de previsão de carga do servidor e risco de saturação (FastAPI).

Executar na pasta do projeto:
    python -m uvicorn api.app:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import config, database, inference

app = FastAPI(
    title="Sistema Inteligente de Gerenciamento de Tráfego e Capacidade",
    description="Prevê as requisições ao servidor web com 1 hora de antecedência e classifica o risco de saturação.",
    version="2.1.0",
)


class EntradaPrevisao(BaseModel):
    """Situação observada AGORA; a previsão é para o intervalo de 15 min que começa daqui a 1 hora."""
    data_hora_alvo: datetime = Field(..., examples=["1995-08-30T14:00:00"],
                                     description="Início do intervalo de 15 min a prever (agora + 1h)")
    requisicoes_agora: float = Field(..., ge=0, examples=[900], description="Requisições nos últimos 15 min")
    requisicoes_15min_antes: float = Field(..., ge=0, examples=[870])
    requisicoes_30min_antes: float = Field(..., ge=0, examples=[850])
    requisicoes_45min_antes: float = Field(..., ge=0, examples=[820])
    requisicoes_ontem_mesmo_horario: float = Field(..., ge=0, examples=[980],
                                                   description="No mesmo intervalo do alvo, 24h antes")
    requisicoes_semana_passada_mesmo_horario: float = Field(..., ge=0, examples=[950])
    media_ultima_hora: float = Field(..., ge=0, examples=[860])
    media_ultimas_24h: float = Field(..., ge=0, examples=[550])
    desvio_ultimas_24h: float = Field(..., ge=0, examples=[260])
    clientes_distintos_agora: float = Field(..., ge=0, examples=[130], description="Hosts distintos nos últimos 15 min")
    taxa_erro_agora: float = Field(0.0, ge=0, le=100, description="% de respostas 4xx/5xx nos últimos 15 min")
    mb_transferidos_agora: float = Field(12.0, ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": config.MODEL_PATH.exists()}


@app.get("/model/info")
@app.get("/modelo", include_in_schema=False)
def modelo():
    """Model card: versão, hiperparâmetros, métricas, período de treino/teste, hash dos dados e run do MLflow."""
    if not config.MODEL_CARD_PATH.exists():
        raise HTTPException(404, "Modelo ainda não treinado. Execute python -m src.pipeline")
    return json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))


@app.get("/metricas")
def metricas():
    return database.ler_tabela("metricas").round(4).to_dict(orient="records")


@app.post("/prever")
def prever(e: EntradaPrevisao):
    valores = {
        "req_1h_atras": e.requisicoes_agora, "req_1h15_atras": e.requisicoes_15min_antes,
        "req_1h30_atras": e.requisicoes_30min_antes, "req_1h45_atras": e.requisicoes_45min_antes,
        "req_24h_atras": e.requisicoes_ontem_mesmo_horario,
        "req_7d_atras": e.requisicoes_semana_passada_mesmo_horario,
        "media_movel_1h": e.media_ultima_hora, "media_movel_24h": e.media_ultimas_24h,
        "desvio_movel_24h": e.desvio_ultimas_24h, "hosts_1h_atras": e.clientes_distintos_agora,
        "taxa_erro_1h_atras": e.taxa_erro_agora, "mb_1h_atras": e.mb_transferidos_agora,
    }
    try:
        resultado = inference.prever(e.data_hora_alvo, valores)
    except FileNotFoundError as erro:
        raise HTTPException(503, str(erro))
    resultado["id_registro"] = database.registrar_predicao("/prever", e.model_dump(mode="json"), resultado)
    return resultado


@app.get("/prever/historico")
def prever_historico(data_hora_alvo: datetime):
    """Ex.: /prever/historico?data_hora_alvo=1995-08-31T11:00:00 (usa a série salva no SQLite)."""
    try:
        resultado = inference.prever_pelo_historico(data_hora_alvo)
    except ValueError as erro:
        raise HTTPException(422, str(erro))
    except FileNotFoundError as erro:
        raise HTTPException(503, str(erro))
    resultado["id_registro"] = database.registrar_predicao(
        "/prever/historico", {"data_hora_alvo": str(data_hora_alvo)}, resultado)
    return resultado


@app.get("/predicoes")
def predicoes(limite: int = Query(20, ge=1, le=500)):
    """Últimas previsões servidas pela API (gravadas no SQLite com a versão do modelo)."""
    df = database.listar_predicoes(limite)
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")   # NaN vira null no JSON
