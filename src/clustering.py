"""Clustering (K-Means): perfis diários de tráfego e perfis de clientes (hosts)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config


def avaliar_k(X, ks=range(2, 7), amostra: int = 10_000) -> pd.DataFrame:
    """Silhouette (maior = melhor) e Davies-Bouldin (menor = melhor) para cada k."""
    X = np.asarray(X)
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X), size=min(amostra, len(X)), replace=False)
    linhas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        linhas.append({"k": k, "silhouette": silhouette_score(X[idx], km.labels_[idx]),
                       "davies_bouldin": davies_bouldin_score(X, km.labels_), "inercia": km.inertia_})
    return pd.DataFrame(linhas)


def melhor_k(avaliacao: pd.DataFrame, tolerancia: float = 0.01) -> int:
    """Regra da parcimônia: entre os k com Silhouette a menos de 1% do máximo, fica o menor k."""
    limite = avaliacao["silhouette"].max() * (1 - tolerancia)
    return int(avaliacao.loc[avaliacao["silhouette"] >= limite, "k"].min())


# ---------------------------------------------------------------- perfis diários
def perfis_diarios(grade: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por dia completo, 24 colunas (requisições em cada hora do dia)."""
    por_hora = grade[config.TARGET].resample("h").sum(min_count=config.PASSOS_HORA)
    t = por_hora.to_frame("req")
    t["data"], t["hora"] = t.index.normalize(), t.index.hour
    return t.pivot_table(index="data", columns="hora", values="req").dropna()


def agrupar_dias(perfis: pd.DataFrame, k: int) -> pd.DataFrame:
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(perfis)
    r = pd.DataFrame(index=perfis.index)
    r["cluster"] = km.labels_
    r["req_dia"] = perfis.sum(axis=1)
    r["fim_de_semana"] = (perfis.index.dayofweek >= 5).astype(int)
    ordem = r.groupby("cluster")["req_dia"].mean().sort_values(ascending=False).index
    nomes = {2: ["Dia útil", "Fim de semana"],
             3: ["Dia de evento (pico excepcional)", "Dia útil", "Fim de semana"]}.get(
        k, [f"Perfil {i + 1}" for i in range(k)])
    r["perfil"] = r["cluster"].map({c: nomes[i] for i, c in enumerate(ordem)})
    return r


# ---------------------------------------------------------------- perfis de clientes
ATRIBUTOS_HOST = ["requisicoes", "kb_por_requisicao", "horas_ativas", "req_por_hora_ativa",
                  "taxa_erro", "pct_cache_304"]


def matriz_hosts(perfil: pd.DataFrame) -> np.ndarray:
    """Log nas contagens (distribuições muito assimétricas) + padronização."""
    X = perfil[ATRIBUTOS_HOST].copy()
    for c in ["requisicoes", "kb_por_requisicao", "horas_ativas", "req_por_hora_ativa"]:
        X[c] = np.log1p(X[c])
    return StandardScaler().fit_transform(X)


def agrupar_hosts(perfil: pd.DataFrame, X: np.ndarray, k: int) -> tuple[pd.Series, pd.DataFrame]:
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    rotulo = pd.Series(km.labels_, index=perfil.index, name="cluster")
    total = perfil["requisicoes"].sum()
    resumo = perfil.assign(cluster=rotulo).groupby("cluster").agg(
        hosts=("requisicoes", "size"), req_mediana=("requisicoes", "median"),
        horas_ativas_mediana=("horas_ativas", "median"), kb_por_req=("kb_por_requisicao", "median"),
        taxa_erro_media=("taxa_erro", "mean"), pct_cache_304=("pct_cache_304", "mean"),
        req_total=("requisicoes", "sum"))
    resumo["pct_do_trafego"] = 100 * resumo["req_total"] / total
    resumo["pct_dos_hosts"] = 100 * resumo["hosts"] / resumo["hosts"].sum()
    resumo = resumo.sort_values("req_mediana", ascending=False)
    resumo.insert(0, "perfil", _nomear_hosts(resumo))
    return rotulo.map(resumo["perfil"]), resumo.reset_index(drop=True)


def _nomear_hosts(resumo: pd.DataFrame) -> list[str]:
    """Nomeia os grupos pelas características (resumo já ordenado por requisições, decrescente)."""
    nomes, normais = {}, []
    for c, linha in resumo.iterrows():
        if linha["taxa_erro_media"] > 5:
            nomes[c] = "Clientes com muitos erros"
        else:
            normais.append(c)
    for i, c in enumerate(normais):
        if i == 0:
            nomes[c] = "Recorrentes (uso intenso)"
        elif i == len(normais) - 1:
            nomes[c] = "Ocasionais (visita rápida)"
        else:
            nomes[c] = f"Intermediários {i}"
    return [nomes[c] for c in resumo.index]
