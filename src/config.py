"""Configurações centrais do projeto (caminhos, granularidade, horizonte e regras de capacidade)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
PARAMS = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))   # sementes, divisão e hiperparâmetros

DATA_RAW_DIR = ROOT / "data" / "raw"
# Arquivos aceitos, em ordem de preferência:
#   CSV do Kaggle (data.csv ou renomeado), o próprio .zip baixado do Kaggle,
#   ou os logs originais NASA_access_log_Jul95.gz / NASA_access_log_Aug95.gz.
CSV_CANDIDATOS = ["nasa_access_log_1995.csv", "data.csv"]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
DB_PATH = OUTPUTS_DIR / "trafego.db"
MODEL_PATH = MODELS_DIR / "modelo_trafego.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

URLS_OFICIAIS = [
    "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz",
    "ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz",
]
TOTAL_OFICIAL_REQUISICOES = 3_461_612   # informado na página oficial da base
FUSO_HORARIO_HORAS = -4                 # os logs são em horário local da Flórida (-0400)

RANDOM_STATE = PARAMS["semente"]
TEST_SIZE = PARAMS["divisao"]["teste"]
CV_SPLITS = PARAMS["divisao"]["dobras_cv"]

# Série temporal: requisições a cada 15 minutos
RESOLUCAO = PARAMS["dados"]["resolucao"]
PASSOS_HORA = 4
PASSOS_DIA = 96
PASSOS_SEMANA = 672
# Horizonte: prevemos a carga de DAQUI A 1 HORA (tempo para escalar a infraestrutura)
HORIZONTE = PARAMS["previsao"]["horizonte_passos"]     # passos de 15 min
LACUNA_MINIMA = PARAMS["dados"]["lacuna_minima_passos"]  # >= 1h seguida sem requisição = servidor fora do ar

TARGET = "requisicoes"

# Capacidade operacional de referência = percentil 95 do TREINO (req/15 min).
# Não é a capacidade física medida do servidor: é uma regra operacional do protótipo.
CAPACIDADE_PERCENTIL = PARAMS["capacidade"]["percentil"]
LIMIAR_MEDIO = PARAMS["capacidade"]["limiar_medio"]
LIMIAR_ALTO = PARAMS["capacidade"]["limiar_alto"]

FERIADOS = PARAMS["dados"]["feriados"]   # Independence Day (EUA)

# Rastreamento de experimentos (opcional): MLflow com banco SQLite na pasta do projeto
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTEFATOS = ROOT / "mlruns"
MLFLOW_EXPERIMENTO = "trafego-capacidade"
MLFLOW_MODELO_REGISTRADO = "previsao-trafego-1h"


def garantir_pastas() -> None:
    for pasta in (DATA_RAW_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
        pasta.mkdir(parents=True, exist_ok=True)
