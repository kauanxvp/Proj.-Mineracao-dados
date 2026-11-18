"""Dashboard operacional: Tráfego atual -> Previsão -> Capacidade disponível -> Risco de saturação.

Executar na pasta do projeto:
    python -m streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import capacity, config, database  # noqa: E402

st.set_page_config(page_title="Tráfego e Capacidade — Servidor Web", page_icon="🛰️", layout="wide")
CORES_RISCO = {"BAIXO": "#3a9a5b", "MÉDIO": "#e07b39", "ALTO": "#c0392b"}
FIG = config.FIGURES_DIR
DIAS_SEMANA = ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"]
# Eixos dos gráficos em português: datas só com números e milhar com ponto (1.600, e não 1,600)
EIXO_DATA = alt.Axis(format="%d/%m %Hh", labelAngle=0, labelOverlap=True)
EIXO_NUM = {"labelExpr": "replace(datum.label, ',', '.')"}


def fmt(v: float, casas: int = 0) -> str:
    return f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tabela(df: pd.DataFrame, nomes: dict, datas: tuple = ()) -> None:
    """Mostra a tabela com títulos em português, números no formato do navegador (1.107,5) e datas dd/mm/aaaa."""
    df = df.copy()
    for c in datas:
        df[c] = pd.to_datetime(df[c])
    cfg = {}
    for c in df.columns:
        rotulo = nomes.get(c, c)
        if c in datas:
            cfg[c] = st.column_config.DatetimeColumn(rotulo, format="DD/MM/YYYY HH:mm")
        elif pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any():
            # valor ausente aparece como "—" (e não "None"); os demais no formato brasileiro
            inteiro = df[c].dropna().mod(1).eq(0).all()
            df[c] = df[c].map(lambda v: "—" if pd.isna(v) else fmt(v, 0 if inteiro else 1))
            cfg[c] = st.column_config.TextColumn(rotulo)
        elif pd.api.types.is_numeric_dtype(df[c]) and c != "id":
            cfg[c] = st.column_config.NumberColumn(rotulo, format="localized")
        else:
            df[c] = df[c].replace({"{}": "—", None: "—"})
            cfg[c] = st.column_config.Column(rotulo)
    st.dataframe(df, column_config=cfg, width="stretch", hide_index=True)


@st.cache_data
def carregar():
    prev = database.ler_tabela("previsoes_teste")
    prev["data_hora"] = pd.to_datetime(prev["data_hora"])
    card = json.loads(config.MODEL_CARD_PATH.read_text(encoding="utf-8"))
    qualidade = json.loads((config.OUTPUTS_DIR / "quality_report.json").read_text(encoding="utf-8"))
    return (prev, database.ler_tabela("metricas"), database.ler_tabela("validacao_cruzada"),
            database.ler_tabela("clusters_diarios"), database.ler_tabela("clusters_clientes"), card, qualidade)


if not config.MODEL_CARD_PATH.exists() or not config.DB_PATH.exists():
    st.error("Modelo não encontrado. Execute primeiro `python -m src.pipeline` ou o notebook.")
    st.stop()

prev, metricas, cv, dias_cl, clientes_cl, card, qualidade = carregar()
cap_base = card["capacidade_referencia_req_15min"]

# ---------------- Barra lateral ----------------
st.sidebar.header("Momento da decisão")
datas = sorted(prev["data_hora"].dt.date.unique())
dia = st.sidebar.selectbox("Dia (período de teste)", datas, index=len(datas) - 1,
                           format_func=lambda d: f"{d:%d/%m/%Y} ({DIAS_SEMANA[d.weekday()]})")
horarios = prev.loc[prev["data_hora"].dt.date == dia, "data_hora"].dt.strftime("%H:%M").tolist()
alvo_txt = st.sidebar.select_slider("Horário a prever (daqui a 1h)", options=horarios,
                                    value="11:00" if "11:00" in horarios else horarios[-1])
st.sidebar.divider()
st.sidebar.subheader("Planejamento de capacidade")
crescimento = st.sidebar.slider("Crescimento da demanda (%)", -20, 100, 0, 5,
                                help="Ex.: campanha, lançamento ou evento que aumenta o acesso.")
expansao = st.sidebar.slider("Expansão da capacidade (%)", 0, 100, 0, 10,
                             help="Ex.: adicionar servidores ou instâncias.")
capacidade = cap_base * (1 + expansao / 100)

cen = prev.copy()
cen["previsto_cenario"] = cen["previsto"] * (1 + crescimento / 100)
cen["risco_cenario"] = capacity.classificar_risco(cen["previsto_cenario"], capacidade)
alvo = pd.Timestamp(f"{dia} {alvo_txt}")
linha = cen[cen["data_hora"] == alvo].iloc[0]
agora = alvo - pd.Timedelta(hours=1)

# ---------------- Fluxo principal ----------------
st.title("🛰️ Gerenciamento de Tráfego e Capacidade")
st.caption(f"Servidor web da NASA (jul-ago/1995) · modelo **{card['modelo']}** · versão {card['versao']} · "
           f"capacidade de referência {fmt(capacidade)} req/15 min ({fmt(capacidade / 900, 2)} req/s)")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"1. Tráfego atual ({agora:%H:%M})", f"{fmt(linha['agora'])} req",
          help="Requisições nos últimos 15 minutos")
c1.caption(f"{fmt(linha['hosts_agora'])} clientes distintos")
c2.metric(f"2. Previsão para {alvo:%H:%M}", f"{fmt(linha['previsto_cenario'])} req",
          delta=f"{linha['previsto_cenario'] - linha['agora']:+.0f} em 1h")
c3.metric("3. Capacidade disponível", f"{fmt(capacidade - linha['previsto_cenario'])} req",
          delta=f"utilização {100 * linha['previsto_cenario'] / capacidade:.0f}%", delta_color="off")
risco = linha["risco_cenario"]
c4.markdown(f"**4. Risco de saturação**<br><span style='font-size:2.1rem;font-weight:700;"
            f"color:{CORES_RISCO[risco]}'>{risco}</span>", unsafe_allow_html=True)
if risco == "ALTO":
    st.warning(f"Ação sugerida: escalar a infraestrutura antes das {alvo:%H:%M}. Seria necessário cerca de "
               f"{max(0, 100 * (linha['previsto_cenario'] / (0.9 * cap_base) - 1)):.0f}% a mais de capacidade "
               "para ficar abaixo de 90% de utilização.")
st.caption(f"Valor real observado às {alvo:%H:%M}: {fmt(linha['real'])} requisições")

ini, fim = alvo.normalize() - pd.Timedelta(days=2), alvo.normalize() + pd.Timedelta(days=1)
jan = cen[(cen["data_hora"] >= ini) & (cen["data_hora"] < fim)]
longo = jan.melt("data_hora", ["real", "previsto_cenario"], var_name="serie", value_name="req")
longo["serie"] = longo["serie"].map({"real": "Real", "previsto_cenario": "Previsto (1h antes)"})
graf = alt.Chart(longo).mark_line().encode(
    x=alt.X("data_hora:T", title=None, axis=EIXO_DATA),
    y=alt.Y("req:Q", title="requisições por 15 min", axis=alt.Axis(**EIXO_NUM)),
    color=alt.Color("serie:N", scale=alt.Scale(range=["#e07b39", "#222222"]), title=None),
    tooltip=[alt.Tooltip("data_hora:T", title="Data e hora", format="%d/%m/%Y %H:%M"),
             alt.Tooltip("serie:N", title="Série"), alt.Tooltip("req:Q", title="Requisições", format=".0f")])
regras = alt.Chart(pd.DataFrame({"y": [capacidade, 0.9 * capacidade], "n": ["Capacidade", "Limiar ALTO"]})) \
    .mark_rule(color="#c0392b", strokeDash=[4, 3]).encode(y="y:Q", tooltip=alt.Tooltip("n:N", title="Linha"))
marca = alt.Chart(pd.DataFrame({"x": [alvo]})).mark_rule(color="#1f5f8b", size=2).encode(x="x:T")
st.altair_chart((graf + regras + marca).properties(height=320), width="stretch")

aba1, aba2, aba3, aba4, aba5 = st.tabs(["Planejamento de capacidade", "Modelos", "Perfis de tráfego (K-Means)",
                                        "Qualidade dos dados", "Previsões da API"])
with aba1:
    tot = cen["risco_cenario"].value_counts().reindex(capacity.NIVEIS, fill_value=0)
    a, b, c = st.columns(3)
    a.metric("Horas em risco ALTO no teste", fmt(tot["ALTO"] / 4, 1))
    b.metric("Horas em risco MÉDIO", fmt(tot["MÉDIO"] / 4, 1))
    c.metric("Horas em risco BAIXO", fmt(tot["BAIXO"] / 4, 1))
    por_hora = (cen.assign(hora=cen["data_hora"].dt.hour, alto=cen["risco_cenario"].eq("ALTO"))
                   .groupby("hora")["alto"].mean().mul(100).reset_index())
    st.altair_chart(alt.Chart(por_hora).mark_bar(color="#c0392b").encode(
        x=alt.X("hora:O", title="hora do dia", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("alto:Q", title="% dos intervalos em risco ALTO", axis=alt.Axis(**EIXO_NUM)),
        tooltip=[alt.Tooltip("hora:O", title="Hora"), alt.Tooltip("alto:Q", title="% em risco ALTO", format=".1f")])
        .properties(height=260, title="Horários em que a saturação se concentra"), width="stretch")
    st.info("Teste nos controles da barra lateral: se a demanda crescer 30%, quanto é preciso expandir "
            "a capacidade para zerar as horas de risco ALTO?")

with aba2:
    tabela(metricas.round(3), {"modelo": "Modelo", "MAPE": "MAPE (%)", "R2": "R²"})
    st.markdown("Escolha feita pela **validação cruzada temporal** no treino (tabela abaixo); o teste é usado "
                "uma única vez. As baselines de série temporal são a referência mínima a superar.")
    colunas_cv = [c for c in ["modelo", "RMSE_cv_media", "RMSE_cv_desvio", "MAE_cv_media", "MAE_cv_desvio",
                              "combinacoes_testadas", "melhores_parametros"] if c in cv.columns]
    tabela(cv[colunas_cv].round(1), {
        "modelo": "Modelo", "RMSE_cv_media": "RMSE (média)", "RMSE_cv_desvio": "RMSE (desvio)",
        "MAE_cv_media": "MAE (média)", "MAE_cv_desvio": "MAE (desvio)",
        "combinacoes_testadas": "Combinações testadas", "melhores_parametros": "Melhores parâmetros"})
    col1, col2 = st.columns(2)
    col1.image(str(FIG / "06_comparacao_modelos.png")); col2.image(str(FIG / "08_importancia_atributos.png"))
    col1.image(str(FIG / "09_erro_por_hora.png")); col2.image(str(FIG / "11_matriz_confusao_risco.png"))

with aba3:
    st.image(str(FIG / "13_perfis_diarios.png"))
    tabela(dias_cl.groupby("perfil").agg(dias=("cluster", "size"), req_por_dia=("req_dia", "mean"))
           .round(0).reset_index(), {"perfil": "Perfil", "dias": "Dias", "req_por_dia": "Requisições por dia (média)"})
    st.image(str(FIG / "14_perfis_clientes.png"))
    tabela(clientes_cl.round(1), {
        "perfil": "Perfil", "hosts": "Clientes", "req_mediana": "Requisições (mediana)",
        "horas_ativas_mediana": "Horas ativas (mediana)", "kb_por_req": "KB por requisição",
        "taxa_erro_media": "Taxa de erro média (%)", "pct_cache_304": "Respostas 304 - cache (%)",
        "req_total": "Requisições (total)", "pct_do_trafego": "% do tráfego", "pct_dos_hosts": "% dos clientes"})
    st.image(str(FIG / "12_escolha_k.png"))

with aba4:
    q = qualidade
    a, b, c, d = st.columns(4)
    a.metric("Requisições no arquivo", fmt(q["registros"]))
    b.metric("Clientes distintos", fmt(q["hosts_distintos"]))
    c.metric("Horas sem registros", fmt(q["horas_em_lacunas"], 1))
    d.metric("Respostas com erro", f"{fmt(q['percentual_erros_4xx_5xx'], 2)}%")
    st.markdown("**Períodos sem registros** (servidor fora do ar ou log perdido)")
    tabela(pd.DataFrame(q["lacunas"]), {"inicio": "Início", "fim": "Fim", "horas": "Horas"}, datas=("inicio", "fim"))
    st.image(str(FIG / "01_serie_temporal.png")); st.image(str(FIG / "04_status_http.png"))

with aba5:
    st.markdown("Cada consulta feita à API (`POST /prever` e `GET /prever/historico`) fica gravada no SQLite com a "
                "**versão do modelo** que respondeu. Atualize a página depois de consultar a API.")
    hist = database.listar_predicoes(50)
    if hist.empty:
        st.info("Nenhuma previsão registrada ainda. Ligue a API e faça uma consulta em http://127.0.0.1:8000/docs")
    else:
        tabela(hist.drop(columns=["entrada"]), {
            "id": "Nº", "criado_em": "Consultado em", "origem": "Endpoint", "previsao_para": "Previsão para",
            "volume_previsto": "Volume previsto", "utilizacao_percentual": "Utilização (%)",
            "risco_saturacao": "Risco", "volume_real": "Volume real", "modelo": "Modelo",
            "versao_modelo": "Versão do modelo"}, datas=("criado_em", "previsao_para"))
