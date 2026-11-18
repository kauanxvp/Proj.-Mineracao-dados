"""Pipeline completo CRISP-DM: logs HTTP -> série de 15 min -> modelos -> avaliação -> implantação.

Execução (na pasta do projeto):
    python -m src.pipeline
ou pelo notebook notebooks/projeto_trafego_crispdm.ipynb no Jupyter Lab.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

if __package__ in (None, ""):  # permite também "python src/pipeline.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

from . import capacity, clustering, config, data, database, viz  # noqa: E402
from .features import FEATURES, criar_features  # noqa: E402
from .models import obter_modelos  # noqa: E402


def _metricas(y_true, y_pred) -> dict:
    """MAE e RMSE em requisições por 15 min; MAPE em %; R² sem unidade."""
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": 100 * mean_absolute_percentage_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


# 2. Entendimento dos dados --------------------------------------------------
def etapa_entendimento():
    fontes = data.localizar_fontes()
    bruto = data.carregar_bruto(fontes)
    qualidade = data.relatorio_qualidade(bruto)
    qualidade["arquivos"] = [f.name for f in fontes]
    qualidade["hash_md5"] = data.hash_arquivos(fontes)
    return bruto, qualidade


# 3. Preparação dos dados ----------------------------------------------------
def etapa_preparacao(bruto: pd.DataFrame):
    grade = data.agregar(bruto)
    tabela = criar_features(grade)
    return grade, tabela


def divisao_temporal(tabela: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Primeiros 80% do tempo = treino; últimos 20% = teste. Sem embaralhar."""
    corte = int(len(tabela) * (1 - test_size))
    return tabela.iloc[:corte], tabela.iloc[corte:]


# 4. Modelagem ---------------------------------------------------------------
def etapa_modelagem(treino: pd.DataFrame, verbose: bool = True):
    """Busca de hiperparâmetros com validação cruzada TEMPORAL, usando somente o treino."""
    X, y = treino[FEATURES], treino[config.TARGET]
    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    linhas, ajustados = [], {}
    for nome, (modelo, grade_params) in obter_modelos().items():
        t0 = time.time()
        busca = GridSearchCV(modelo, grade_params or {}, cv=tscv, n_jobs=1, refit="rmse",
                             scoring={"rmse": "neg_root_mean_squared_error",
                                      "mae": "neg_mean_absolute_error"}).fit(X, y)
        i = busca.best_index_
        r = busca.cv_results_
        ajustados[nome] = busca.best_estimator_
        linhas.append({
            "modelo": nome,
            "RMSE_cv_media": -r["mean_test_rmse"][i],
            "RMSE_cv_desvio": r["std_test_rmse"][i],
            "MAE_cv_media": -r["mean_test_mae"][i],
            "MAE_cv_desvio": r["std_test_mae"][i],
            "combinacoes_testadas": len(busca.cv_results_["params"]),
            "melhores_parametros": json.dumps({k.split("__")[-1]: str(v) for k, v in busca.best_params_.items()},
                                              ensure_ascii=False),
            "tempo_s": time.time() - t0,
        })
        if verbose:
            l = linhas[-1]
            print(f"  {nome:<44} RMSE CV = {l['RMSE_cv_media']:6.1f} ± {l['RMSE_cv_desvio']:5.1f}"
                  f"  ({l['combinacoes_testadas']} config., {l['tempo_s']:4.1f}s)")
    cv = pd.DataFrame(linhas)
    candidatos = cv[~cv["modelo"].str.startswith("Baseline")]
    melhor = candidatos.sort_values("RMSE_cv_media").iloc[0]["modelo"]
    return ajustados, cv, melhor


# 5. Avaliação ---------------------------------------------------------------
def etapa_avaliacao(ajustados: dict, teste: pd.DataFrame, melhor: str, capacidade_ref: float):
    X, y = teste[FEATURES], teste[config.TARGET]
    metricas = pd.DataFrame([{"modelo": n, **_metricas(y, m.predict(X))} for n, m in ajustados.items()])
    metricas = metricas.sort_values("RMSE").reset_index(drop=True)

    prev = pd.DataFrame({
        "data_hora": teste.index, "real": y.values,
        "previsto": np.clip(ajustados[melhor].predict(X), 0, None),
        "agora": teste["req_1h_atras"].values, "hosts_agora": teste["hosts_1h_atras"].values,
    })
    prev["risco_real"] = capacity.classificar_risco(prev["real"], capacidade_ref)
    prev["risco_previsto"] = capacity.classificar_risco(prev["previsto"], capacidade_ref)

    rotulos = capacity.NIVEIS
    cm = confusion_matrix(prev["risco_real"], prev["risco_previsto"], labels=rotulos)
    rel = classification_report(prev["risco_real"], prev["risco_previsto"], labels=rotulos,
                                output_dict=True, zero_division=0)
    pi = permutation_importance(ajustados[melhor], X, y, scoring="neg_root_mean_squared_error",
                                n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=1)
    imp = pd.DataFrame({"atributo": FEATURES, "importancia": pi.importances_mean})
    return metricas, prev, cm, rel, imp.sort_values("importancia", ascending=False)


# Clustering -----------------------------------------------------------------
def etapa_clustering(grade: pd.DataFrame, bruto: pd.DataFrame):
    perfis = clustering.perfis_diarios(grade)
    aval_dias = clustering.avaliar_k(perfis)
    k_dias = clustering.melhor_k(aval_dias)
    dias = clustering.agrupar_dias(perfis, k_dias)

    hosts = data.perfil_hosts(bruto)
    X = clustering.matriz_hosts(hosts)
    aval_hosts = clustering.avaliar_k(X)
    k_hosts = clustering.melhor_k(aval_hosts)
    rotulo_hosts, resumo_hosts = clustering.agrupar_hosts(hosts, X, k_hosts)
    return {"perfis": perfis, "aval_dias": aval_dias, "k_dias": k_dias, "dias": dias,
            "aval_hosts": aval_hosts, "k_hosts": k_hosts, "resumo_hosts": resumo_hosts,
            "n_hosts": len(hosts)}


# Rastreamento de experimentos (MLflow) ---------------------------------------
def registrar_mlflow(ajustados, cv, metricas, melhor, card, treino, verbose: bool = True) -> dict:
    """Chama o registro no MLflow sem deixar uma falha dele interromper o pipeline."""
    try:
        return _registrar_mlflow(ajustados, cv, metricas, melhor, card, treino, verbose)
    except Exception as erro:  # noqa: BLE001
        print(f"  Aviso: registro no MLflow falhou ({type(erro).__name__}: {erro}). O pipeline continua.")
        return {}


def _registrar_mlflow(ajustados, cv, metricas, melhor, card, treino, verbose: bool = True) -> dict:
    """Registra cada modelo como um run no MLflow e o escolhido no Model Registry.

    Opcional: se o MLflow não estiver instalado, o pipeline segue normalmente (o model card e a tabela
    `execucoes` do SQLite continuam registrando a execução). Interface: python -m mlflow ui
    --backend-store-uri sqlite:///mlflow.db (na pasta do projeto).
    """
    try:
        import logging
        import os
        import warnings

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

        import mlflow
        import mlflow.sklearn
    except ImportError:
        if verbose:
            print("  MLflow não instalado: registro de experimentos ignorado (pip install mlflow)")
        return {}
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", module="mlflow")
    mlflow.set_tracking_uri("sqlite:///" + config.MLFLOW_DB.as_posix())
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENTO)
    if exp is not None and exp.artifact_location != config.MLFLOW_ARTEFATOS.as_uri():
        # Banco copiado de outra pasta/computador: guarda o histórico antigo com outro nome e começa um novo
        mlflow.MlflowClient().rename_experiment(
            exp.experiment_id, f"{config.MLFLOW_EXPERIMENTO} (outra pasta {exp.experiment_id})")
        exp = None
    if exp is None:
        mlflow.create_experiment(config.MLFLOW_EXPERIMENTO, artifact_location=config.MLFLOW_ARTEFATOS.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENTO)
    teste_m = metricas.set_index("modelo")
    cv_m = cv.set_index("modelo")
    tags = {"versao_pipeline": card["versao"], "hash_dados": card["dados"]["hash_md5"],
            "horizonte": "1h", "resolucao": config.RESOLUCAO}
    info = {}
    with mlflow.start_run(run_name=f"pipeline {card['versao']}", tags=tags) as pai:
        mlflow.log_params({"semente": config.RANDOM_STATE, "dobras_cv": config.CV_SPLITS,
                           "teste": config.TEST_SIZE, "horizonte_passos": config.HORIZONTE,
                           "capacidade_percentil": config.CAPACIDADE_PERCENTIL, "modelo_escolhido": melhor})
        mlflow.log_artifact(str(config.PARAMS_PATH))
        for nome in ajustados:
            with mlflow.start_run(run_name=nome, nested=True, tags={**tags, "escolhido": str(nome == melhor)}):
                mlflow.log_params({"modelo": nome, **json.loads(cv_m.loc[nome, "melhores_parametros"])})
                mlflow.log_metrics({"rmse_cv_media": cv_m.loc[nome, "RMSE_cv_media"],
                                    "rmse_cv_desvio": cv_m.loc[nome, "RMSE_cv_desvio"],
                                    "mae_cv_media": cv_m.loc[nome, "MAE_cv_media"],
                                    **{f"{k.lower()}_teste": float(v) for k, v in teste_m.loc[nome].items()}})
                if nome == melhor:
                    modelo_info = mlflow.sklearn.log_model(
                        ajustados[nome], name="modelo", input_example=treino[FEATURES].head(3),
                        serialization_format="cloudpickle",   # modelo criado por nós mesmos
                        registered_model_name=config.MLFLOW_MODELO_REGISTRADO)
                    info = {"run_id": mlflow.active_run().info.run_id,
                            "modelo_registrado": config.MLFLOW_MODELO_REGISTRADO,
                            "versao_registro": int(modelo_info.registered_model_version)}
        info["run_pipeline"] = pai.info.run_id
    if verbose:
        print(f"  MLflow: {len(ajustados)} runs registrados; modelo '{config.MLFLOW_MODELO_REGISTRADO}' "
              f"versão {info.get('versao_registro')}")
    return info


# 6. Implantação -------------------------------------------------------------
def etapa_implantacao(ajustados, melhor, capacidade_ref, metricas, cv, prev, grade, qualidade,
                      clusters, rel_risco, imp, treino, teste, verbose: bool = True):
    config.garantir_pastas()
    versao = datetime.now().strftime("%Y%m%d-%H%M%S")
    linha = metricas.set_index("modelo").loc[melhor]
    card = {
        "versao": versao,
        "modelo": melhor,
        "melhores_parametros": json.loads(cv.set_index("modelo").loc[melhor, "melhores_parametros"]),
        "tarefa": "prever requisições por 15 min com 1 hora de antecedência",
        "dados": {"arquivos": qualidade["arquivos"], "hash_md5": qualidade["hash_md5"],
                  "registros": qualidade["registros"]},
        "treino": {"inicio": str(treino.index.min()), "fim": str(treino.index.max()), "intervalos": len(treino)},
        "teste": {"inicio": str(teste.index.min()), "fim": str(teste.index.max()), "intervalos": len(teste)},
        "metricas_teste": {k: round(float(v), 4) for k, v in linha.items()},
        "capacidade_referencia_req_15min": round(capacidade_ref, 1),
        "capacidade_referencia_req_s": round(capacidade_ref / 900, 2),
        "regra_risco": "BAIXO < 70% <= MÉDIO < 90% <= ALTO (utilização = previsão / capacidade)",
        "risco_alto": {k: round(v, 3) for k, v in rel_risco["ALTO"].items()},
        "features": FEATURES,
        "versoes": {"python": sys.version.split()[0], "pandas": pd.__version__,
                    "scikit-learn": sklearn.__version__},
    }
    card["mlflow"] = registrar_mlflow(ajustados, cv, metricas, melhor, card, treino, verbose)
    joblib.dump({"modelo": ajustados[melhor], "nome_modelo": melhor, "versao": versao, "features": FEATURES,
                 "capacidade_referencia": capacidade_ref, "horizonte_min": 15 * config.HORIZONTE,
                 "limiares": {"medio": config.LIMIAR_MEDIO, "alto": config.LIMIAR_ALTO}},
                config.MODEL_PATH)
    config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")

    metricas.to_csv(config.OUTPUTS_DIR / "metrics.csv", index=False)
    # Métricas do modelo implantado em JSON (lidas pelo dvc metrics show e citadas no relatório)
    (config.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(
        {"modelo": melhor, **card["metricas_teste"],
         "risco_alto": card["risco_alto"], "rmse_cv_media": round(float(cv.set_index("modelo")
                                                                          .loc[melhor, "RMSE_cv_media"]), 4)},
        indent=2, ensure_ascii=False), encoding="utf-8")
    cv.to_csv(config.OUTPUTS_DIR / "cv_metrics.csv", index=False)
    imp.to_csv(config.OUTPUTS_DIR / "importancia_atributos.csv", index=False)
    clusters["aval_dias"].to_csv(config.OUTPUTS_DIR / "clustering_dias_k.csv", index=False)
    clusters["aval_hosts"].to_csv(config.OUTPUTS_DIR / "clustering_clientes_k.csv", index=False)
    clusters["resumo_hosts"].to_csv(config.OUTPUTS_DIR / "clustering_clientes_resumo.csv", index=False)
    (config.OUTPUTS_DIR / "quality_report.json").write_text(
        json.dumps(qualidade, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    serie = grade.reset_index()
    database.salvar_tabela(serie.assign(data_hora=serie["data_hora"].astype(str)), "requisicoes_15min")
    database.salvar_tabela(prev.assign(data_hora=prev["data_hora"].astype(str)), "previsoes_teste")
    database.salvar_tabela(metricas, "metricas")
    database.salvar_tabela(cv, "validacao_cruzada")
    dias = clusters["dias"].reset_index()
    database.salvar_tabela(dias.assign(data=dias["data"].astype(str)), "clusters_diarios")
    database.salvar_tabela(clusters["resumo_hosts"], "clusters_clientes")
    database.registrar_execucao({"melhor_modelo": melhor, "hash_dados": qualidade["hash_md5"],
                                 **card["metricas_teste"]})
    return card


def gerar_figuras(grade, qualidade, metricas, melhor, prev, capacidade_ref, imp, cm, clusters) -> list[str]:
    return [
        viz.serie_temporal(grade, qualidade["lacunas"]), viz.perfil_horario(grade),
        viz.heatmap_hora_dia(grade), viz.status_http(qualidade), viz.usuarios_vs_requisicoes(grade),
        viz.comparacao_modelos(metricas, melhor), viz.real_vs_previsto(prev, melhor, capacidade_ref),
        viz.importancia(imp, melhor), viz.erro_por_hora(prev), viz.risco_saturacao(prev, capacidade_ref),
        viz.matriz_confusao(cm, capacity.NIVEIS), viz.escolha_k(clusters["aval_dias"], clusters["aval_hosts"]),
        viz.perfis_clusters(clusters["perfis"], clusters["dias"]), viz.clusters_hosts(clusters["resumo_hosts"]),
        viz.residuos(prev, melhor),
    ]


# Orquestração ---------------------------------------------------------------
def executar_pipeline(verbose: bool = True) -> dict:
    t0 = time.time()
    config.garantir_pastas()
    log = print if verbose else (lambda *a, **k: None)

    log("[1/6] Negócio: prever a carga do servidor com 1h de antecedência e o risco de saturação")
    log("[2/6] Entendimento dos dados (lendo ~3 milhões de linhas de log)")
    bruto, qualidade = etapa_entendimento()
    log(f"  {qualidade['registros']:,} requisições | {qualidade['hosts_distintos']:,} clientes | "
        f"{qualidade['horas_em_lacunas']:.0f} h sem registros".replace(",", "."))

    log("[3/6] Preparação: agregação em intervalos de 15 min e engenharia de atributos")
    grade, tabela = etapa_preparacao(bruto)
    treino, teste = divisao_temporal(tabela)
    capacidade_ref = capacity.calcular_capacidade(treino[config.TARGET])
    log(f"  tabela de modelagem: {len(tabela)} intervalos x {len(FEATURES)} atributos")
    log(f"  treino: {treino.index.min():%d/%m %H:%M} a {treino.index.max():%d/%m %H:%M} ({len(treino)})"
        f" | teste: {teste.index.min():%d/%m %H:%M} a {teste.index.max():%d/%m %H:%M} ({len(teste)})")
    log(f"  capacidade de referência (P95 do treino): {capacidade_ref:.0f} req/15 min "
        f"(~{capacidade_ref / 900:.2f} req/s)")

    log("[4/6] Modelagem: busca de hiperparâmetros com validação cruzada temporal")
    ajustados, cv, melhor = etapa_modelagem(treino, verbose)
    log(f"  modelo escolhido pela validação: {melhor}")

    log("[5/6] Avaliação no conjunto de teste + clustering")
    metricas, prev, cm, rel_risco, imp = etapa_avaliacao(ajustados, teste, melhor, capacidade_ref)
    clusters = etapa_clustering(grade, bruto)
    if verbose:
        print(metricas.round(3).to_string(index=False))
        a = rel_risco["ALTO"]
        print(f"  risco ALTO -> precision {a['precision']:.2f} | recall {a['recall']:.2f} | F1 {a['f1-score']:.2f}")
        for chave, nome in [("dias", "perfis diários"), ("hosts", "perfis de clientes")]:
            k = clusters[f"k_{chave}"]
            lk = clusters[f"aval_{chave}"].set_index("k").loc[k]
            print(f"  K-Means {nome}: k={k} | Silhouette {lk['silhouette']:.3f}"
                  f" | Davies-Bouldin {lk['davies_bouldin']:.3f}")

    figuras = gerar_figuras(grade, qualidade, metricas, melhor, prev, capacidade_ref, imp, cm, clusters)

    log("[6/6] Implantação: modelo versionado, model card e banco SQLite")
    card = etapa_implantacao(ajustados, melhor, capacidade_ref, metricas, cv, prev, grade, qualidade,
                             clusters, rel_risco, imp, treino, teste, verbose)
    log(f"  salvo: {config.MODEL_PATH.relative_to(config.ROOT)} | "
        f"{config.DB_PATH.relative_to(config.ROOT)} | {len(figuras)} gráficos")
    log(f"Concluído em {time.time() - t0:.0f}s")

    return {"bruto": bruto, "qualidade": qualidade, "grade": grade, "tabela": tabela, "treino": treino,
            "teste": teste, "capacidade": capacidade_ref, "modelos": ajustados, "cv": cv, "melhor": melhor,
            "metricas": metricas, "previsoes": prev, "matriz_confusao": cm, "relatorio_risco": rel_risco,
            "importancia": imp, "clusters": clusters, "model_card": card, "figuras": figuras}


if __name__ == "__main__":
    executar_pipeline()
