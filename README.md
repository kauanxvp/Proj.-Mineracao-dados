# Sistema Inteligente de Gerenciamento de Tráfego e Capacidade

Projeto de Mineração de Dados · Engenharia de Software (6º período) · **Tema 5** · Kauan de Oliveira Machado e
Carlos Leopoldo Moscardi · Metodologia **CRISP-DM** + MLOps

A partir dos **logs HTTP reais do servidor web da NASA** (jul-ago/1995, cerca de 3 milhões de requisições), o sistema
prevê quantas requisições chegarão **daqui a 1 hora** (em janelas de 15 minutos) e classifica o **risco de saturação**
(BAIXO / MÉDIO / ALTO). O resultado é entregue por um pipeline reprodutível (params.yaml, DVC e MLflow), uma API
(FastAPI) que grava cada previsão no banco e um dashboard operacional (Streamlit).

## Como rodar

### Com Docker (tudo de uma vez)

```bash
docker compose up --build
```

A build treina o modelo e roda os testes. Depois: API em http://localhost:8000/docs, dashboard em
http://localhost:8501 e MLflow em http://localhost:5000.

### Sem Docker (Windows + PowerShell), sempre na pasta do projeto

```powershell
python -m pip install -r requirements.txt     # inclui statsmodels, pyyaml e mlflow
python -m src.pipeline                        # refaz dados -> treino -> avaliação (~2 min); ou: dvc repro
python -m pytest -q                           # 18 testes
```

| Ação | Comando |
|---|---|
| API | `python -m uvicorn api.app:app --reload` → http://127.0.0.1:8000/docs |
| Dashboard | `python -m streamlit run dashboard/streamlit_app.py` → http://localhost:8501 |
| MLflow | `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000 |
| Notebook | `python -m jupyter lab` → `notebooks/projeto_trafego_crispdm.ipynb` → *Restart Kernel and Run All Cells* |
| Métricas pelo DVC | `dvc metrics show` |

A API e o dashboard usam o modelo salvo em `models/`. O pacote já vem com o modelo treinado; se rodar o pipeline ou o
notebook de novo, reinicie a API.

## Base de dados

*NASA-HTTP* (Internet Traffic Archive; Arlitt e Williamson, ACM SIGMETRICS 1996), versão do Kaggle *NASA access log
dataset 1995*, em `data/raw/nasa_access_log_1995.zip` (lido direto do zip). Os traços podem ser redistribuídos
livremente; a fonte pede análise só de padrões gerais de tráfego.

| Pedido no Tema 5 | Nesta base |
|---|---|
| logs HTTP | ✅ logs reais do servidor |
| número de requisições | ✅ requisições por 15 min (alvo) |
| usuários simultâneos | ✅ aproximado por clientes (hosts) distintos por intervalo |
| códigos HTTP | ✅ taxa de erro 4xx/5xx |
| horários e histórico | ✅ 2 meses em série de 15 min |
| latência, CPU, memória | ❌ não existem na base (limitação declarada) |

## Estrutura

```
projeto_trafego_capacidade/
├── data/raw/                 nasa_access_log_1995.zip + LEIA-ME.txt
├── notebooks/                projeto_trafego_crispdm.ipynb (narrativa CRISP-DM, executado)
├── src/
│   ├── config.py             caminhos; lê o params.yaml
│   ├── data.py               leitura (zip/CSV do Kaggle ou log original), qualidade, ETL para série de 15 min
│   ├── features.py           atributos disponíveis 1 h antes (mesma função no treino, na API e no dashboard)
│   ├── models.py             3 baselines, Regressão Linear, Random Forest, Gradient Boosting, Rede Neural
│   ├── serie_temporal.py     ARIMA + Fourier (sazonalidade diária e semanal)
│   ├── clustering.py         K-Means de perfis diários e de perfis de clientes
│   ├── capacity.py           capacidade de referência e risco
│   ├── database.py           SQLite: série, previsões, métricas, execuções e previsões da API
│   ├── viz.py                gráficos
│   ├── inference.py          previsão com o modelo salvo
│   ├── pipeline.py           orquestra as etapas e registra no MLflow
│   └── download_data.py      tenta baixar os logs originais
├── api/app.py                /health, /model/info, /metricas, /prever, /prever/historico, /predicoes
├── dashboard/streamlit_app.py
├── tests/                    18 testes (ETL, lacunas, sem vazamento, ARIMA, risco, modelo, API)
├── models/ · outputs/        gerados pelo pipeline (modelo, model card, métricas, figuras, trafego.db)
├── params.yaml · dvc.yaml    parâmetros do experimento e pipeline declarado
├── mlflow.db · mlruns/       runs do MLflow (gerados pelo pipeline)
└── Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Resultados (teste: 23/08 07h45 a 31/08 23h45, previsão com 1 hora de antecedência)

| Modelo | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| **Random Forest** (escolhido pela validação cruzada) | 81,1 | 111,8 | 19,1% | 0,877 |
| ARIMA + Fourier | 89,8 | 122,5 | 21,0% | 0,852 |
| Regressão Linear | 94,3 | 128,2 | 21,0% | 0,838 |
| Gradient Boosting | 90,9 | 135,1 | 19,8% | 0,820 |
| Rede Neural (MLP) | 92,9 | 135,5 | 19,6% | 0,819 |
| Baseline persistência | 116,6 | 158,8 | 25,4% | 0,751 |
| Baseline sazonal semanal | 139,0 | 204,0 | 27,3% | 0,589 |
| Baseline sazonal diária | 156,7 | 218,6 | 35,9% | 0,529 |

- Unidade: requisições por 15 minutos. Capacidade de referência: **1.107 req/15 min** (P95 do treino, ≈ 1,23 req/s).
- O modelo escolhido reduz o RMSE em 30% em relação à melhor baseline (persistência).
- Risco ALTO: precision 0,87 · recall 0,63 · F1 0,73.
- K-Means: 3 perfis de dia (útil, fim de semana, dia de evento: 13/07, lançamento do STS-70), Silhouette 0,522;
  3 perfis de clientes (recorrentes geram 76% do tráfego sendo 33% dos clientes), Silhouette 0,300.
- Validação cruzada (RMSE): Random Forest 109,8 ± 15,6 × Gradient Boosting 110,0 ± 12,8. A escolha é automática
  (menor RMSE na validação) e fica registrada em `models/model_card.json` e no MLflow. Valores obtidos com Python 3.13,
  scikit-learn 1.9.1 e pandas 3.0.5; com outras versões o desempate pode mudar.

## MLOps

- `params.yaml`: semente (42), divisão, horizonte, regras de capacidade e grades de hiperparâmetros.
- `dvc.yaml`: o pipeline como estágio do DVC (`dvc repro` só refaz se a base, o código ou os parâmetros mudarem).
- MLflow: um run por execução e um run filho por modelo (parâmetros e métricas); o modelo escolhido é registrado como
  `previsao-trafego-1h`. Se o MLflow não estiver instalado, o pipeline continua e só pula esse registro.
- Model card: versão, hiperparâmetros, métricas, períodos, hash MD5 da base, versões das bibliotecas e run do MLflow.
- A API grava cada previsão na tabela `predicoes_api` com a versão do modelo (`GET /predicoes`).
- CI (GitHub Actions): testes unitários, pipeline completo, todos os testes e build da imagem Docker.

## Limitações

Sem latência/CPU/memória na base; capacidade definida por regra (P95), não medida; dois meses com lacunas; tráfego de
1995; versões das bibliotecas com mínimo, não fixadas; sem ingestão em tempo real nem monitoramento automático de drift.
