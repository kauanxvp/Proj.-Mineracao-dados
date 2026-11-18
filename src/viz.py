"""Gráficos gerados pelo pipeline (salvos em outputs/figures)."""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

AZUL, LARANJA, VERDE, VERMELHO, CINZA = "#1f5f8b", "#e07b39", "#3a9a5b", "#c0392b", "#8a9499"
DIAS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
UNIDADE = "requisições por 15 min"


def _salvar(fig, nome: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = config.FIGURES_DIR / nome
    fig.tight_layout()
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return str(caminho)


def _datas(ax, diario: bool = False):
    if diario:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.SA))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))


# ---------------- Entendimento dos dados ----------------
def serie_temporal(grade: pd.DataFrame, lacunas: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(grade.index, grade[config.TARGET], color=AZUL, lw=0.5)
    for i, lac in enumerate(lacunas):
        ax.axvspan(pd.Timestamp(lac["inicio"]), pd.Timestamp(lac["fim"]), color=CINZA, alpha=0.35,
                   label="Sem registros (fora do ar / perdido)" if i == 0 else None)
    pico = grade[config.TARGET].idxmax()
    ax.annotate("13/07: lançamento\ndo ônibus espacial", xy=(pico, grade[config.TARGET].max()),
                xytext=(pico + pd.Timedelta(days=4), grade[config.TARGET].max() * 0.9),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax.set_title("Requisições ao servidor web da NASA a cada 15 minutos (jul-ago/1995)")
    ax.set_ylabel(UNIDADE); _datas(ax); ax.legend(loc="upper right")
    return _salvar(fig, "01_serie_temporal.png")


def perfil_horario(grade: pd.DataFrame) -> str:
    g = grade.dropna(subset=[config.TARGET])
    tipo = np.where(g.index.dayofweek >= 5, "Fim de semana", "Dia útil")
    hora = g.index.hour + g.index.minute / 60
    tab = g.groupby([hora, tipo])[config.TARGET].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(tab.index, tab["Dia útil"], color=AZUL, lw=2, label="Dia útil")
    ax.plot(tab.index, tab["Fim de semana"], color=LARANJA, lw=2, label="Fim de semana")
    ax.set_title("Perfil médio ao longo do dia: pico no horário comercial dos EUA")
    ax.set_xlabel("hora do dia (horário da Flórida)"); ax.set_ylabel(UNIDADE)
    ax.set_xticks(range(0, 25, 2)); ax.legend()
    return _salvar(fig, "02_perfil_horario.png")


def heatmap_hora_dia(grade: pd.DataFrame) -> str:
    g = grade.dropna(subset=[config.TARGET])
    tab = g.groupby([g.index.dayofweek, g.index.hour])[config.TARGET].mean().unstack()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(tab.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7)); ax.set_yticklabels(DIAS)
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hora do dia")
    ax.set_title("Mapa de calor: requisições médias por dia da semana e hora")
    fig.colorbar(im, ax=ax, label=UNIDADE)
    return _salvar(fig, "03_mapa_calor_hora_dia.png")


def status_http(qualidade: dict) -> str:
    st = pd.Series(qualidade["status_http"])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cores = [VERDE if c < 300 else AZUL if c < 400 else VERMELHO for c in st.index]
    ax.bar(st.index.astype(str), st.values, color=cores)
    ax.set_yscale("log")
    for i, v in enumerate(st.values):
        ax.text(i, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
    ax.set_title("Códigos de resposta HTTP (escala log)")
    ax.set_ylabel("requisições")
    return _salvar(fig, "04_status_http.png")


def usuarios_vs_requisicoes(grade: pd.DataFrame) -> str:
    g = grade.dropna(subset=[config.TARGET])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(g["hosts_unicos"], g[config.TARGET], s=4, alpha=0.3, color=AZUL)
    r = np.corrcoef(g["hosts_unicos"], g[config.TARGET])[0, 1]
    ax.set_title(f"Clientes distintos x requisições (correlação {r:.2f})")
    ax.set_xlabel("hosts distintos no intervalo"); ax.set_ylabel(UNIDADE)
    return _salvar(fig, "05_usuarios_vs_requisicoes.png")


# ---------------- Modelagem e avaliação ----------------
def comparacao_modelos(metricas: pd.DataFrame, melhor: str) -> str:
    m = metricas.sort_values("RMSE", ascending=False)
    cores = [VERDE if n == melhor else CINZA if n.startswith("Baseline") else AZUL for n in m["modelo"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(m["modelo"], m["RMSE"], color=cores)
    for i, v in enumerate(m["RMSE"]):
        ax.text(v, i, f" {v:.0f}", va="center")
    ax.set_title("RMSE no conjunto de teste (menor é melhor; verde = modelo escolhido)")
    ax.set_xlabel("RMSE (requisições por 15 min)")
    return _salvar(fig, "06_comparacao_modelos.png")


def real_vs_previsto(prev: pd.DataFrame, nome_modelo: str, capacidade: float, dias: int = 4) -> str:
    p = prev[prev["data_hora"] > prev["data_hora"].max() - pd.Timedelta(days=dias)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(p["data_hora"], p["real"], color="black", lw=1.1, label="Real")
    ax.plot(p["data_hora"], p["previsto"], color=LARANJA, lw=1.3, ls="--",
            label=f"Previsto com 1h de antecedência ({nome_modelo})")
    ax.axhline(capacidade, color=VERMELHO, lw=1, label=f"Capacidade de referência ({capacidade:.0f})")
    ax.axhline(capacidade * config.LIMIAR_ALTO, color=VERMELHO, lw=1, ls=":", label="Limiar ALTO (90%)")
    ax.set_title(f"Real x previsto — últimos {dias} dias do teste")
    ax.set_ylabel(UNIDADE); _datas(ax, diario=True); ax.legend(loc="upper left", fontsize=8, ncol=2)
    return _salvar(fig, "07_real_vs_previsto.png")


def importancia(imp: pd.DataFrame, nome_modelo: str) -> str:
    imp = imp.sort_values("importancia").tail(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp["atributo"], imp["importancia"], color=AZUL)
    ax.set_title(f"Importância por permutação — {nome_modelo}")
    ax.set_xlabel("aumento do RMSE ao embaralhar o atributo")
    return _salvar(fig, "08_importancia_atributos.png")


def erro_por_hora(prev: pd.DataFrame) -> str:
    e = prev.assign(hora=prev["data_hora"].dt.hour, erro=(prev["real"] - prev["previsto"]).abs())
    tab = e.groupby("hora")["erro"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(tab.index, tab.values, color=LARANJA)
    ax.set_title("Erro absoluto médio por hora do dia")
    ax.set_xlabel("hora"); ax.set_ylabel("MAE"); ax.set_xticks(range(0, 24, 2))
    return _salvar(fig, "09_erro_por_hora.png")


def risco_saturacao(prev: pd.DataFrame, capacidade: float, dias: int = 3) -> str:
    p = prev[prev["data_hora"] > prev["data_hora"].max() - pd.Timedelta(days=dias)]
    cores = p["risco_previsto"].map({"BAIXO": VERDE, "MÉDIO": LARANJA, "ALTO": VERMELHO})
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(p["data_hora"], p["previsto"], width=1 / 96, color=cores)
    ax.plot(p["data_hora"], p["real"], color="black", lw=1, label="Real")
    ax.axhline(capacidade, color=VERMELHO, lw=1, label="Capacidade")
    ax.axhline(capacidade * config.LIMIAR_ALTO, color=VERMELHO, ls=":", lw=1, label="90%")
    ax.axhline(capacidade * config.LIMIAR_MEDIO, color=LARANJA, ls=":", lw=1, label="70%")
    ax.set_title("Risco de saturação previsto (verde = BAIXO, laranja = MÉDIO, vermelho = ALTO)")
    ax.set_ylabel(UNIDADE); _datas(ax, diario=True); ax.legend(loc="upper left", fontsize=8, ncol=4)
    return _salvar(fig, "10_risco_saturacao.png")


def matriz_confusao(cm: np.ndarray, rotulos: list[str]) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(rotulos))); ax.set_xticklabels(rotulos)
    ax.set_yticks(range(len(rotulos))); ax.set_yticklabels(rotulos)
    ax.set_xlabel("risco previsto"); ax.set_ylabel("risco real")
    ax.set_title("Matriz de confusão do risco")
    return _salvar(fig, "11_matriz_confusao_risco.png")


def escolha_k(aval_dias: pd.DataFrame, aval_hosts: pd.DataFrame) -> str:
    fig, eixos = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, aval, titulo in [(eixos[0], aval_dias, "Perfis diários"), (eixos[1], aval_hosts, "Perfis de clientes")]:
        ax.plot(aval["k"], aval["silhouette"], "o-", color=AZUL, label="Silhouette (maior melhor)")
        ax2 = ax.twinx()
        ax2.plot(aval["k"], aval["davies_bouldin"], "s--", color=LARANJA, label="Davies-Bouldin (menor melhor)")
        ax.set_xlabel("k"); ax.set_ylabel("Silhouette", color=AZUL); ax2.set_ylabel("Davies-Bouldin", color=LARANJA)
        ax.set_title(f"Escolha de k — {titulo}")
    return _salvar(fig, "12_escolha_k.png")


def perfis_clusters(perfis: pd.DataFrame, rotulos: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cores = {"Dia de evento (pico excepcional)": VERMELHO, "Dia útil": AZUL, "Fim de semana": LARANJA}
    for nome, grupo in rotulos.groupby("perfil"):
        media = perfis.loc[grupo.index].mean()
        ax.plot(media.index, media.values, lw=2.2, color=cores.get(nome), label=f"{nome} ({len(grupo)} dias)")
    ax.set_title("Perfis diários encontrados pelo K-Means (média de cada grupo)")
    ax.set_xlabel("hora do dia"); ax.set_ylabel("requisições por hora"); ax.set_xticks(range(0, 24, 2))
    ax.legend()
    return _salvar(fig, "13_perfis_diarios.png")


def clusters_hosts(resumo: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    y = np.arange(len(resumo))
    ax.barh(y - 0.2, resumo["pct_dos_hosts"], height=0.4, color=CINZA, label="% dos clientes")
    ax.barh(y + 0.2, resumo["pct_do_trafego"], height=0.4, color=AZUL, label="% do tráfego")
    ax.set_yticks(y); ax.set_yticklabels(resumo["perfil"]); ax.invert_yaxis()
    for i, (h, t) in enumerate(zip(resumo["pct_dos_hosts"], resumo["pct_do_trafego"])):
        ax.text(h, i - 0.2, f" {h:.1f}%", va="center", fontsize=9)
        ax.text(t, i + 0.2, f" {t:.1f}%", va="center", fontsize=9)
    ax.set_title("Perfis de clientes (K-Means): quem gera o tráfego")
    ax.set_xlabel("%"); ax.legend(loc="lower right")
    return _salvar(fig, "14_perfis_clientes.png")


def residuos(prev: pd.DataFrame, nome_modelo: str) -> str:
    """Resíduos (real - previsto) ao longo do período de teste."""
    r = prev["real"] - prev["previsto"]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(prev["data_hora"], r, color=AZUL, lw=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.fill_between(prev["data_hora"], -r.abs().mean(), r.abs().mean(), color=CINZA, alpha=0.25,
                    label=f"± MAE ({r.abs().mean():.0f})")
    ax.set_ylabel("real − previsto")
    ax.set_title(f"Resíduos no teste ao longo do tempo — {nome_modelo}")
    ax.legend(loc="upper left")
    _datas(ax, diario=True)
    return _salvar(fig, "15_residuos_tempo.png")
