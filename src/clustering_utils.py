"""
Clustering algorithms and evaluation utilities for the smart K-means research project.
"""

import math
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    euclidean_distances,
    silhouette_score,
)

from src.settings import load_config


def gap_statistic_method(data, nrefs=3, min_clusters=2, max_clusters=15):
    """
    Calcular a quantidade ótima de clusters utilizando o método Gap Statistic
    Params:
        data: Conjunto de dados (DataFrame ou ndarray)
        nrefs: Quantidade de conjuntos de referência que serão criados
        maxClusters: Número máximo de clusters que serão testados
    Return: (melhor_k, resultsDf)
    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    gaps = np.zeros((len(range(min_clusters, max_clusters)),))
    df_result = pd.DataFrame({"qty_clusters": [], "gap": []})

    for gap_index, k in enumerate(range(min_clusters, max_clusters)):
        ref_disps = np.zeros(nrefs)

        # Para n referências, gere uma amostra aleatória e execute o algoritmo K-means,
        # obtendo a dispersão resultante de cada iteração.
        for i in range(nrefs):
            # Criar conjunto aleatório de referência
            random_reference = np.random.random_sample(size=data.shape)

            # Aplicar o K-means no conjunto de referência
            kmeans = KMeans(k)
            kmeans.fit(random_reference)
            ref_disp = kmeans.inertia_
            ref_disps[i] = ref_disp

        # Aplicar o K-means no conjunto de dados original e calculaar a dispersão
        kmeans = KMeans(k)
        kmeans.fit(data)
        orig_disp = kmeans.inertia_

        # Calcular o GAP Statistic
        gap = np.log(np.mean(ref_disps)) - np.log(orig_disp)

        # Armazenar o resultado do GAP Statistic
        gaps[gap_index] = gap
        df_result.loc[gap_index] = [k, gap]

    best_k_index = gaps.argmax()

    best_k = df_result.iloc[best_k_index]["qty_clusters"]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=df_result["qty_clusters"], y=df_result["gap"], marker="o", ax=ax)

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Gap Statistic")
    ax.set_title("Best Cluster w/ Gap Statistic")
    ax.axvline(
        x=best_k, color="red", linestyle="--", label=f"Ideal no. clusters: {best_k}"
    )
    # ax.legend()

    return best_k, best_k_index, df_result, fig


def elbow_method(data, min_clusters=2, max_clusters=10):
    """
    Implementação do método de cotovelo para encontrar o número ideal de clusters.

    Params:
        data: ndarray de forma (n_samples, n_features)
            O conjunto de dados a ser agrupado.
        max_clusters: int, opcional (padrão=10)
            O número máximo de clusters a serem testados.

    Return:
        (indice_maior_elbow, resultsDf)
    """
    distortions = []  # Armazena a soma dos quadrados das distâncias intra-cluster
    df_result = pd.DataFrame({"qty_clusters": [], "distortions": []})

    for i, k in enumerate(range(min_clusters, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)

        # Armazenar o resultado do soma dos quadrados das distâncias intra-cluster
        distortions.append(kmeans.inertia_)
        df_result.loc[i] = [k, kmeans.inertia_]

    # Encontrar o ponto de cotovelo (método mais simples - procurando a maior inclinação)
    differences = np.diff(distortions)
    best_k_index = np.argmax(differences)
    best_k = df_result.iloc[best_k_index]["qty_clusters"]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=df_result["qty_clusters"], y=df_result["distortions"], marker="o", ax=ax
    )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Sum of squares of intra-cluster distances")
    ax.set_title("Elbow method")
    ax.axvline(
        x=best_k,
        color="red",
        linestyle="--",
        label=f"Ideal number of clusters: {best_k}",
    )
    # ax.legend()

    return best_k, best_k_index, df_result, fig, differences


def silhouette_score_method(data, min_clusters=2, max_clusters=15):
    """
    Implementação do Silhouette Score para determinar o número ideal de clusters usando K-means.

    Params:
        data: ndarray de forma (n_samples, n_features)
            O conjunto de dados a ser agrupado.
        min_clusters: int, opcional (padrão=3)
            O número mínimo de clusters a serem testados.
        max_clusters: int, opcional (padrão=10)
            O número máximo de clusters a serem testados.

    Return:
        indice_maior_silhueta: int
            O número ideal de clusters com base no Silhouette Score.
        resultsDf: DataFrame
            Dataframe contendo os Silhouette Scores para cada número de clusters testado.
    """

    silhouette_scores = []
    df_results = pd.DataFrame({"qty_clusters": [], "silhouette_score": []})

    for i, k in enumerate(range(min_clusters, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        silhouette_scores.append(silhouette_avg)
        df_results.loc[i] = [k, silhouette_avg]

    best_k_index = np.argmax(silhouette_scores)
    best_k = df_results.iloc[best_k_index]["qty_clusters"]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=df_results["qty_clusters"],
        y=df_results["silhouette_score"],
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score for K-means")
    ax.axvline(
        x=best_k,
        color="red",
        linestyle="--",
        label=f"Ideal number of clusters: {best_k}",
    )
    # ax.legend()

    return best_k, best_k_index, df_results, fig


def calinski_harabasz_index_method(data, min_clusters=3, max_clusters=10):
    """
    Implementação do índice de Calinski-Harabasz para determinar o número ideal de clusters usando K-means.

    Params:
        data: ndarray de forma (n_samples, n_features)
            O conjunto de dados a ser agrupado.
        min_clusters: int, opcional (padrão=3)
            O número mínimo de clusters a serem testados.
        max_clusters: int, opcional (padrão=10)
            O número máximo de clusters a serem testados.

    Return:
        best_k: int
            O número ideal de clusters com base no índice de Calinski-Harabasz.
        calinski_scores: list
            Lista contendo os índices de Calinski-Harabasz para cada número de clusters testado.
    """
    calinski_scores = []
    df_results = pd.DataFrame({"qty_clusters": [], "calinski_score": []})

    for i, k in enumerate(range(min_clusters, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        calinski_score = calinski_harabasz_score(data, cluster_labels)

        # Armazenar o resultado dos scores
        calinski_scores.append(calinski_score)
        df_results.loc[i] = [k, calinski_score]

    best_k_index = np.argmax(calinski_scores)
    best_k = df_results.iloc[best_k_index]["qty_clusters"]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=df_results["qty_clusters"], y=df_results["calinski_score"], marker="o", ax=ax
    )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Calinski-Harabasz index for K-means")
    ax.axvline(
        x=best_k,
        color="red",
        linestyle="--",
        label=f"Ideal number of clusters: {best_k}",
    )
    # ax.legend()

    return best_k, best_k_index, df_results, fig


def davies_bouldin_index_method(data, min_clusters=3, max_clusters=10):
    """
    Implementação do Índice Davies-Bouldin para determinar o número ideal de clusters usando K-means.

    Params:
        data: ndarray de forma (n_samples, n_features)
            O conjunto de dados a ser agrupado.
        max_clusters: int, opcional (padrão=10)
            O número máximo de clusters a serem testados.

    Return:
        best_k: int
            O número ideal de clusters com base no Índice Davies-Bouldin.
        davies_bouldin_scores: list
            Lista contendo os Índices Davies-Bouldin para cada número de clusters testado.
    """
    davies_bouldin_scores = []
    df_results = pd.DataFrame({"qty_clusters": [], "davies_bouldin_score": []})
    for i, k in enumerate(range(min_clusters, max_clusters + 1)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        davies_bouldin_score_value = davies_bouldin_score(data, cluster_labels)

        # Armazenar o resultado dos scores
        davies_bouldin_scores.append(davies_bouldin_score_value)
        df_results.loc[i] = [k, davies_bouldin_score_value]

    best_k_index = np.argmin(davies_bouldin_scores)
    best_k = df_results.iloc[best_k_index]["qty_clusters"]

    # Generate graph
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=df_results["qty_clusters"],
        y=df_results["davies_bouldin_score"],
        marker="o",
        ax=ax,
    )

    ax.set_xlabel("Número de clusters")
    ax.set_ylabel("Índice Davies-Bouldin")
    ax.set_title("Índice Davies-Bouldin para K-means")
    ax.axvline(
        x=best_k,
        color="red",
        linestyle="--",
        label=f"Número ideal de clusters: {best_k}",
    )
    # ax.legend()

    return best_k, best_k_index, df_results, fig


def get_grouping_suggestions(df, min_clusters=2, max_clusters=15):
    df_suggestions = pd.DataFrame({"method": [], "qty_clusters": []})

    res_gap_stats = gap_statistic_method(
        df, min_clusters=min_clusters, max_clusters=max_clusters
    )

    res_elbow = elbow_method(df, min_clusters, max_clusters)
    res_calinski = calinski_harabasz_index_method(df, min_clusters, max_clusters)
    res_davies = davies_bouldin_index_method(df, min_clusters, max_clusters)
    res_silhouett = silhouette_score_method(df, min_clusters, max_clusters)

    df_suggestions.loc[len(df_suggestions.index)] = ["Gap Statistic", res_gap_stats[0]]
    df_suggestions.loc[len(df_suggestions.index)] = ["Elbow", res_elbow[0]]
    df_suggestions.loc[len(df_suggestions.index)] = ["Calinski", res_calinski[0]]
    df_suggestions.loc[len(df_suggestions.index)] = ["Davies", res_davies[0]]
    df_suggestions.loc[len(df_suggestions.index)] = ["Silhouette", res_silhouett[0]]

    charts = {
        "gap_statistic": res_gap_stats[3],
        "elbow": res_elbow[3],
        "calinski": res_calinski[3],
        "davies": res_davies[3],
        "silhouette": res_silhouett[3],
    }

    # Generate comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_suggestions,
        x="method",
        y="qty_clusters",
        palette="tab10",
        hue="method",
        ax=ax,
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Qty Clusters")
    ax.set_title("Comparision methods of suggestions clustering")
    ##ax.legend()

    return df_suggestions, charts, fig


def calculate_dunn_index(X, labels, centroids):
    """
    Calcula o Índice de Dunn.

    O Índice de Dunn é uma métrica de validação de cluster que mede a
    razão entre a menor distância inter-cluster e a maior distância intra-cluster.
    Um valor maior indica melhor agrupamento.

    Args:
        X (np.array): O conjunto de dados original.
        labels (np.array): Os rótulos de cluster atribuídos a cada ponto.
        centroids (np.array): Os centroides dos clusters.

    Returns:
        float: O valor do Índice de Dunn.
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    if num_clusters < 2:
        return 0.0

    # Calcular a menor distância inter-cluster
    min_inter_cluster_dist = float("inf")
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < min_inter_cluster_dist:
                min_inter_cluster_dist = dist

    # Calcular a maior distância intra-cluster
    max_intra_cluster_dist = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            distances = euclidean_distances(cluster_points)
            max_current_intra_dist = np.max(distances)
            if max_current_intra_dist > max_intra_cluster_dist:
                max_intra_cluster_dist = max_current_intra_dist

    if max_intra_cluster_dist == 0:
        return float("inf")

    dunn_index = min_inter_cluster_dist / max_intra_cluster_dist
    return dunn_index


def calculate_shannon_entropy(dados):
    """
    Calcula a entropia de Shannon de uma série de dados.

    Args:
        dados (pd.Series or list): Uma série ou lista de dados.

    Returns:
        float: O valor da entropia de Shannon.
    """
    if not isinstance(dados, pd.Series):
        dados = pd.Series(dados)

    if dados.empty:
        return 0.0

    counts = Counter(dados)
    total_elements = len(dados)
    probabilities = [count / total_elements for count in counts.values()]
    entropy_value = 0.0
    for p in probabilities:
        if p > 0:
            entropy_value -= p * math.log2(p)
    return entropy_value


def calculate_linear_entropy(scores):
    """
    Calcula uma medida de entropia linear.

    Args:
        scores (np.array): Uma matriz de pontuações.

    Returns:
        float: O valor da entropia linear arredondado para 3 casas decimais.
    """
    entropia_ideal = np.linspace(np.min(scores), np.max(scores), len(scores))
    absolute_difference = np.abs(scores - entropia_ideal)
    system_entropy = np.sum(absolute_difference)
    max_entropy_system = np.full(len(scores), np.min(scores))
    max_entropy_system[-1] = np.max(scores)
    max_entropy_system = np.abs(max_entropy_system - entropia_ideal)
    max_entropy = np.sum(max_entropy_system)
    result = 1 - (system_entropy / max_entropy)
    return round(result, 3)


def calculate_pca(df):
    """
    Calcula a Análise de Componentes Principais (PCA) e retorna os loadings.

    Args:
        df (pd.DataFrame): O DataFrame de dados.

    Returns:
        pd.DataFrame: Um DataFrame com os loadings PCA e o ranking
                      das variáveis por influência na PC1.
    """
    pca = PCA(n_components=len(df.columns))
    pca.fit(df)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(len(df.columns))],
        index=df.columns,
    )
    loadings["PC1_abs"] = loadings["PC1"].abs()
    ranking = loadings.sort_values(by="PC1_abs")

    return ranking


def save_model(model, filename, models_path=None):
    """
    Save trained clustering model.

    Parameters:
    -----------
    model : object
        Trained clustering model
    filename : str
        Output filename
    models_path : str, optional
        Path to models directory
    """
    if models_path is None:
        config = load_config()
        models_path = config["output"]["models_path"]

    os.makedirs(models_path, exist_ok=True)
    filepath = os.path.join(models_path, filename)
    joblib.dump(model, filepath)


def load_model(filename, models_path=None):
    """
    Load trained clustering model.

    Parameters:
    -----------
    filename : str
        Model filename
    models_path : str, optional
        Path to models directory

    Returns:
    --------
    object
        Loaded clustering model
    """
    if models_path is None:
        config = load_config()
        models_path = config["output"]["models_path"]

    filepath = os.path.join(models_path, filename)
    return joblib.load(filepath)
