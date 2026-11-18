# Prompt para o Gamma (10 slides)

Copie o bloco abaixo e cole no Gamma (**Gerar → Colar texto**, formato **Apresentação**). O Gamma tem limite de 10 slides.
Depois de gerar, troque as imagens sugeridas pelos gráficos de `outputs/figures` indicados no fim deste arquivo.

Os números são os da execução com **Random Forest** como modelo escolhido (Python 3.13/3.14 e scikit-learn 1.9).
A apresentação `.pptx` do pacote já está atualizada com estes números; este prompt só é necessário para gerar outra versão.
Se rodar com outras versões e algum valor mudar, confira em `outputs/metrics.csv` e `models/model_card.json`.

```text
Crie uma apresentação acadêmica em português do Brasil com EXATAMENTE 10 slides, para a disciplina Projeto de Mineração de Dados (Engenharia de Software, 6º período). Estilo técnico e limpo, destaque azul petróleo, vermelho só para indicar risco. No máximo 5 tópicos curtos por slide, números grandes nos resultados. Coloque as falas apenas nas notas do apresentador, nunca no corpo do slide. Use apenas os números abaixo, sem inventar valores.

Slide 1 — Capa
Sistema Inteligente de Gerenciamento de Tráfego e Capacidade. Subtítulo: previsão de carga de servidor web e risco de saturação com Mineração de Dados. Tema 5 · Kauan e Carlos · CRISP-DM.

Slide 2 — Problema e objetivo
Picos de requisições aumentam a latência e derrubam serviços; escalar leva tempo, então é preciso saber antes. Objetivo: prever as requisições de cada janela de 15 min com 1 hora de antecedência e classificar o risco em BAIXO, MÉDIO ou ALTO, com API e dashboard. Critérios de sucesso: superar as baselines de série temporal, R² acima de 0,80 e precision acima de 0,80 na classe ALTO.

Slide 3 — Base de dados e qualidade
Logs HTTP reais do servidor web do Kennedy Space Center da NASA, julho e agosto de 1995: 2.965.561 requisições de 135.268 clientes, com instante, cliente, URL, código HTTP e bytes. Tratamentos: fuso convertido de UTC para -4h; queda real do servidor de 01/08 15h a 03/08 04h30 pelo furacão Erin e lacuna de 28 a 31/07 (cerca de 120 h no total), tratadas como "fora do ar" e não como demanda zero. Latência e CPU não existem na base.

Slide 4 — O que os dados mostram
Tráfego segue o horário comercial dos EUA, com pico entre 12h e 16h. Fim de semana tem 57% do volume de um dia útil. Clientes distintos e requisições têm correlação de 0,97. Em 13/07, lançamento do ônibus espacial STS-70, foram 118.606 requisições no dia, o dobro de um dia útil.

Slide 5 — Pipeline CRISP-DM
Fluxo em 6 etapas: Negócio → Dados → Preparação → Modelagem → Avaliação → Implantação. ETL: 3 milhões de linhas de log viram intervalos de 15 min com requisições, clientes distintos, taxa de erro e MB transferidos. Atributos disponíveis 1 hora antes: requisições de 1h, 1h15, 1h30 e 1h45 atrás, mesmo horário de ontem e da semana passada, médias móveis, clientes agora, hora e dia da semana. Divisão temporal sem embaralhar: treino de 08/07 a 23/08 e teste de 23/08 a 31/08. Hiperparâmetros escolhidos por validação cruzada temporal (TimeSeriesSplit).

Slide 6 — Modelos e resultados
Tabela no teste (requisições por 15 min, MAE / RMSE / MAPE / R²): Random Forest 81,1 / 111,8 / 19,1% / 0,877 (escolhido); ARIMA + Fourier 89,8 / 122,5 / 21,0% / 0,852; Regressão Linear 94,3 / 128,2 / 21,0% / 0,838; Gradient Boosting 90,9 / 135,1 / 19,8% / 0,820; Rede Neural MLP 92,9 / 135,5 / 19,6% / 0,819; Baseline persistência 116,6 / 158,8 / 25,4% / 0,751; Baseline semanal 139,0 / 204,0 / 27,3% / 0,589; Baseline diária 156,7 / 218,6 / 35,9% / 0,529. Mensagem: todos os modelos superam as baselines; o Random Forest teve o menor erro na validação cruzada e no teste; o ARIMA + Fourier, modelo clássico de série temporal, ficou em 2º.

Slide 7 — Previsão e risco de saturação
Previsão feita 1 hora antes acompanha a curva real; os atributos mais importantes são clientes distintos agora e hora do dia. Capacidade de referência: 1.107 requisições por 15 min (percentil 95 do treino). Regra: BAIXO abaixo de 70%, MÉDIO de 70% a 90%, ALTO a partir de 90%. Classe ALTO: precision 0,87, recall 0,63, F1 0,73. Quando alerta, acerta 87% das vezes; baixar o limiar pega mais picos em troca de mais alarmes falsos.

Slide 8 — Perfis de tráfego (K-Means)
Dias, 3 perfis (Silhouette 0,522): dia útil (38 dias, cerca de 58 mil requisições), fim de semana (16 dias, cerca de 34 mil) e dia de evento (13/07, cerca de 119 mil). Clientes, 3 perfis (Silhouette 0,300): recorrentes são 33% dos clientes e geram 76% do tráfego; ocasionais são 65% e geram 24%; 1% concentra erros 404.

Slide 9 — Arquitetura do sistema
Diagrama: logs HTTP → ETL em Python → treino e avaliação → modelo versionado (model card com hash dos dados e registro no MLflow) → banco SQLite, API FastAPI (/prever, /prever/historico, /model/info, /predicoes) e dashboard Streamlit (Tráfego atual → Previsão → Capacidade disponível → Risco, com simulação). Qualidade: params.yaml, DVC, 18 testes automatizados, Docker e integração contínua no GitHub Actions.

Slide 10 — Conclusões, limitações e próximos passos
R² de 0,88 prevendo 1 hora à frente, 87% de acerto nos alertas de risco ALTO e 30% menos erro que a melhor baseline. Limitações: sem latência e CPU; capacidade definida por regra (P95); dois meses de dados com lacunas; recall de 0,63; tráfego de 1995. Próximos passos: coletar latência e CPU (ex.: Prometheus), previsão em vários horizontes, LSTM, detecção de eventos e retreino automático com monitoramento de drift.
```

| Slide | Gráfico sugerido (`outputs/figures`) |
|---|---|
| 3 | `01_serie_temporal.png` |
| 4 | `02_perfil_horario.png` |
| 6 | `06_comparacao_modelos.png` |
| 7 | `07_real_vs_previsto.png` ou `10_risco_saturacao.png` |
| 8 | `13_perfis_diarios.png` e `14_perfis_clientes.png` |
