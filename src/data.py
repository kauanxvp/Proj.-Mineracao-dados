"""Aquisição, diagnóstico de qualidade e agregação dos logs HTTP da NASA (jul-ago/1995)."""
from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

COLUNAS = ["host", "data_hora", "metodo", "url", "status", "bytes"]
_REGEX_LOG = re.compile(r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+)?\s*(\S+)?[^"]*" (\d{3}) (\S+)')


# ---------------------------------------------------------------- aquisição
def localizar_fontes() -> list[Path]:
    """Procura a base em data/raw. Aceita o CSV do Kaggle, o .zip do Kaggle ou os logs originais .gz."""
    pasta = config.DATA_RAW_DIR
    for nome in config.CSV_CANDIDATOS:
        if (pasta / nome).exists():
            return [pasta / nome]
    zips = sorted(pasta.glob("*.zip"))
    if zips:
        return [zips[0]]
    originais = sorted(pasta.glob("NASA_access_log_*"))
    if originais:
        return originais
    raise FileNotFoundError(
        f"Base não encontrada em {pasta}. Copie para essa pasta o data.csv (ou o .zip) baixado do Kaggle "
        "'NASA access log dataset 1995', ou os arquivos originais NASA_access_log_Jul95.gz e _Aug95.gz."
    )


def hash_arquivos(caminhos: list[Path]) -> str:
    """MD5 dos arquivos brutos: identifica exatamente qual versão dos dados treinou o modelo."""
    md5 = hashlib.md5()
    for caminho in caminhos:
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(1 << 20), b""):
                md5.update(bloco)
    return md5.hexdigest()


def _ler_csv_kaggle(caminho: Path) -> pd.DataFrame:
    """CSV do Kaggle: colunas ,host,time,method,url,response,bytes. 'time' é Unix (UTC)."""
    df = pd.read_csv(caminho, usecols=["host", "time", "method", "url", "response", "bytes"],
                     dtype={"host": "string", "method": "category", "url": "string"})
    utc = pd.to_datetime(df["time"], unit="s")
    return pd.DataFrame({
        "host": df["host"],
        # Os logs originais estão no horário local da Flórida (-0400). Convertemos de UTC para ele.
        "data_hora": utc + pd.Timedelta(hours=config.FUSO_HORARIO_HORAS),
        "metodo": df["method"],
        "url": df["url"],
        "status": df["response"].astype("int16"),
        "bytes": pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype("int64"),
    })


def _ler_log_original(caminho: Path) -> pd.DataFrame:
    """Formato Common Log Format dos arquivos oficiais (NASA_access_log_Jul95.gz)."""
    abrir = gzip.open if caminho.suffix == ".gz" else open
    linhas = []
    with abrir(caminho, "rt", encoding="latin-1", errors="ignore") as f:
        for linha in f:
            m = _REGEX_LOG.match(linha)
            if m:
                linhas.append(m.groups())
    df = pd.DataFrame(linhas, columns=["host", "ts", "metodo", "url", "status", "bytes"])
    df["data_hora"] = pd.to_datetime(df["ts"].str[:20], format="%d/%b/%Y:%H:%M:%S")
    df["status"] = df["status"].astype("int16")
    df["bytes"] = pd.to_numeric(df["bytes"].replace("-", "0"), errors="coerce").fillna(0).astype("int64")
    return df[COLUNAS]


def carregar_bruto(caminhos: list[Path] | None = None) -> pd.DataFrame:
    caminhos = caminhos or localizar_fontes()
    if caminhos[0].suffix in (".csv", ".zip"):
        df = _ler_csv_kaggle(caminhos[0])
    else:
        df = pd.concat([_ler_log_original(c) for c in caminhos], ignore_index=True)
    return df.sort_values("data_hora", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- qualidade
def encontrar_lacunas(contagem: pd.Series, minimo: int = config.LACUNA_MINIMA) -> pd.DataFrame:
    """Sequências de intervalos consecutivos sem nenhuma requisição (servidor fora do ar ou dado perdido)."""
    zero = contagem.eq(0)
    grupos = (zero != zero.shift()).cumsum()
    linhas = []
    for _, bloco in contagem[zero].groupby(grupos[zero]):
        if len(bloco) >= minimo:
            linhas.append({"inicio": bloco.index[0], "fim": bloco.index[-1] + pd.Timedelta(config.RESOLUCAO),
                           "horas": len(bloco) / config.PASSOS_HORA})
    return pd.DataFrame(linhas, columns=["inicio", "fim", "horas"])


def relatorio_qualidade(bruto: pd.DataFrame) -> dict:
    contagem = bruto.set_index("data_hora").resample(config.RESOLUCAO).size()
    lacunas = encontrar_lacunas(contagem)
    status = bruto["status"].value_counts().sort_index()
    return {
        "registros": int(len(bruto)),
        "total_oficial": config.TOTAL_OFICIAL_REQUISICOES,
        "percentual_do_oficial": round(100 * len(bruto) / config.TOTAL_OFICIAL_REQUISICOES, 1),
        "periodo_inicio": str(bruto["data_hora"].min()),
        "periodo_fim": str(bruto["data_hora"].max()),
        "hosts_distintos": int(bruto["host"].nunique()),
        "urls_distintas": int(bruto["url"].nunique()),
        "valores_nulos": {c: int(v) for c, v in bruto.isna().sum().items()},
        "linhas_identicas": int(bruto.duplicated().sum()),
        "status_http": {int(k): int(v) for k, v in status.items()},
        "percentual_erros_4xx_5xx": round(100 * bruto["status"].ge(400).mean(), 3),
        "metodos_http": {str(k): int(v) for k, v in bruto["metodo"].value_counts().items()},
        "respostas_sem_corpo_bytes_0": int(bruto["bytes"].eq(0).sum()),
        "intervalos_15min_esperados": int(len(contagem)),
        "intervalos_sem_requisicao": int(contagem.eq(0).sum()),
        "lacunas": [{"inicio": str(r.inicio), "fim": str(r.fim), "horas": r.horas}
                    for r in lacunas.itertuples()],
        "horas_em_lacunas": float(lacunas["horas"].sum()) if len(lacunas) else 0.0,
    }


# ---------------------------------------------------------------- agregação (ETL)
def agregar(bruto: pd.DataFrame) -> pd.DataFrame:
    """Transforma 3 milhões de linhas de log em uma série de 15 em 15 minutos.

    Métricas por intervalo:
      requisicoes    -> volume de tráfego (alvo)
      hosts_unicos   -> clientes distintos, aproximação de usuários simultâneos
      taxa_erro      -> % de respostas 4xx/5xx
      mb_transferidos-> volume de dados enviados (banda)

    Intervalos dentro de uma lacuna (>= 1h sem nenhuma requisição) viram NaN: o servidor estava
    fora do ar, então "zero" ali não é demanda real e não pode ser usado para treinar.
    Intervalos isolados com zero requisições são mantidos como zero (demanda baixa real).
    """
    b = bruto.assign(intervalo=bruto["data_hora"].dt.floor(config.RESOLUCAO),
                     erro=bruto["status"].ge(400))
    g = b.groupby("intervalo")
    serie = pd.DataFrame({
        "requisicoes": g.size(),
        "hosts_unicos": g["host"].nunique(),
        "taxa_erro": 100 * g["erro"].mean(),
        "mb_transferidos": g["bytes"].sum() / 1e6,
    })
    inicio = bruto["data_hora"].min().floor("D")
    fim = bruto["data_hora"].max().ceil("D") - pd.Timedelta(config.RESOLUCAO)
    grade = serie.reindex(pd.date_range(inicio, fim, freq=config.RESOLUCAO))
    grade[["requisicoes", "hosts_unicos", "mb_transferidos"]] = \
        grade[["requisicoes", "hosts_unicos", "mb_transferidos"]].fillna(0)
    grade["taxa_erro"] = grade["taxa_erro"].fillna(0)

    for r in encontrar_lacunas(grade["requisicoes"]).itertuples():
        grade.loc[(grade.index >= r.inicio) & (grade.index < r.fim), :] = np.nan
    grade["fora_do_ar"] = grade["requisicoes"].isna().astype(int)
    grade.index.name = "data_hora"
    return grade


def perfil_hosts(bruto: pd.DataFrame, minimo_requisicoes: int = 5) -> pd.DataFrame:
    """Uma linha por cliente (host) com o seu comportamento agregado, para o clustering de clientes."""
    b = bruto.assign(hora=bruto["data_hora"].dt.floor("h"), erro=bruto["status"].ge(400),
                     nao_modificado=bruto["status"].eq(304))
    g = b.groupby("host")
    p = pd.DataFrame({
        "requisicoes": g.size(),
        "kb_por_requisicao": g["bytes"].mean() / 1e3,
        "horas_ativas": g["hora"].nunique(),
        "taxa_erro": 100 * g["erro"].mean(),
        "pct_cache_304": 100 * g["nao_modificado"].mean(),
    })
    p["req_por_hora_ativa"] = p["requisicoes"] / p["horas_ativas"]
    return p[p["requisicoes"] >= minimo_requisicoes]
