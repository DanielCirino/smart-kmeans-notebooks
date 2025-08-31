import math

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm, pyplot as plt
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, silhouette_samples

from src.clustering_utils import calculate_dunn_index, calculate_shannon_entropy

pio.templates.default = "plotly"


def evaluate_cluster(df, n_clusters):
    """
    Avalia a clusterização K-Means para um número específico de clusters.

    Args:
        df (pd.DataFrame): O DataFrame de dados.
        n_clusters (int): O número de clusters a ser testado.

    Returns:
        dict: Um dicionário com os resultados da avaliação, incluindo
              pontuações, rótulos e detalhes do cluster.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    centroids = kmeans.cluster_centers_
    davies_bouldin_score_value = davies_bouldin_score(df, labels)
    dunn_score_value = calculate_dunn_index(df, labels, centroids)

    df_result = pd.DataFrame(
        {
            "label": labels,
            "score": silhouette_samples(df, labels),
        }
    )

    grouping_summary = df_result.groupby("label")["score"].count()

    grouping_details = []
    for label in np.unique(labels):
        grouping_details.append(
            {
                "group": f"C{label + 1}",
                "count": grouping_summary[label],
                "score": df_result[df_result["label"] == label]["score"].mean(),
                "centroids": centroids,
            }
        )

    return {
        "cluster": f"{n_clusters} groups",
        "num_clusters": n_clusters,
        "labels": labels,
        "silhouette_score": score,
        "davies_bouldin_score": davies_bouldin_score_value,
        "dunn_score": dunn_score_value,
        "silhouettes": silhouette_samples(df, labels),
        "details": grouping_details,
        "k_means": kmeans,
    }


def evaluate_grouping_options(dataset, qtd_min_clusters, qtd_max_clusters):
    """
    Avalia a clusterização K-Means para um intervalo de clusters.

    Args:
        qtd_min_clusters (int): O número mínimo de clusters a ser testado.
        qtd_max_clusters (int): O número máximo de clusters a ser testado.
        dataset (pd.DataFrame): O DataFrame de dados.

    Returns:
        list: Uma lista de dicionários contendo os results para cada arranjo.
    """
    range_clusters = range(qtd_min_clusters, qtd_max_clusters + 1)
    results = []

    for i, k in enumerate(range_clusters):
        resultado = {"grouping": f"{k} Groups", "qty_groups": k}
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(dataset)
        labels = kmeans.labels_
        centroides = kmeans.cluster_centers_
        grouping_summary = []
        qty_per_groups = pd.DataFrame(labels).groupby(0)[0].count()

        for i, qty in enumerate(qty_per_groups):
            grouping_summary.append({"group": i + 1, "qty": qty, "silhouette": 0})

        silhouettes = silhouette_samples(dataset, labels)
        mean_silhouette = round(silhouette_score(dataset, labels), 2)
        resultado["mean_silhouette"] = mean_silhouette

        for i in np.unique(labels):
            grouping_summary[i]["silhouette"] = round(
                np.mean(silhouettes[labels == i]), 2
            )
            grouping_summary[i]["centroides"] = centroides[i]

        resultado["grouping_summary"] = grouping_summary
        resultado["labels"] = labels
        resultado["silhouettes"] = silhouettes
        resultado["centroides"] = centroides
        results.append(resultado)
    return results


def calculate_dataset_entropy(df):
    entropies = []
    for col in df.columns.to_list():
        entropies.append(
            {
                "subindicator": col,
                "entropy": calculate_shannon_entropy(df[col].round(4)),
            }
        )

    df_entropies = pd.DataFrame(entropies)
    df_entropies.sort_values(by=["subindicator"], ascending=False, inplace=True)

    return df_entropies


def calculate_best_k_with_entropy(df, min_clusters=3, max_clusters=7):
    evaluation_result = []
    iterations_summary = []
    df_entropies = calculate_dataset_entropy(df)
    df_entropies.sort_values(by=["entropy"], ascending=False, inplace=True)

    for k in range(min_clusters, max_clusters + 1):
        subindicators_list = list(df_entropies["subindicator"])
        iteration = 1

        for subindicador in list(df_entropies["subindicator"]):
            entropy = df_entropies[df_entropies["subindicator"] == subindicador]
            cluster_assesment = evaluate_cluster(df[subindicators_list], k)

            iterations_summary.append(
                {
                    "cluster": cluster_assesment["cluster"],
                    "iteration": iteration,
                    "excluded_indicator": subindicador,
                    "entropy": entropy["entropy"].values[0],
                    "silhouette_score": cluster_assesment["silhouette_score"],
                    "davies_bouldin_score": cluster_assesment["davies_bouldin_score"],
                    "dunn_score": cluster_assesment["dunn_score"],
                }
            )

            if cluster_assesment["silhouette_score"] > 0.5:
                evaluation_result.append(
                    {
                        "cluster": cluster_assesment["cluster"],
                        "qty_subindicators": len(subindicators_list),
                        "silhouette_score": cluster_assesment["silhouette_score"],
                        "davies_bouldin_score": cluster_assesment[
                            "davies_bouldin_score"
                        ],
                        "dunn_score": cluster_assesment["dunn_score"],
                        "subindicators": subindicators_list,
                        "details": cluster_assesment,
                    }
                )
                break

            subindicators_list.remove(subindicador)
            iteration += 1

    df_iterations = pd.DataFrame(iterations_summary)
    df_evaluation_result = pd.DataFrame(evaluation_result)
    df_evaluation_result.sort_values(
        by=["qty_subindicators", "silhouette_score", "cluster"],
        ascending=(False, False, True),
        inplace=True,
        ignore_index=True,
    )

    return df_evaluation_result, df_entropies, df_iterations, df.columns, df


def get_valid_results(results, cut_silhouette=0.50):
    """
    Filtra os resultados de clusterização para obter apenas os arranjos válidos.

    Um arranjo é considerado válido se a silhueta média e a silhueta de cada
    subgrupo forem maiores que um valor de corte.

    Args:
        results (list): Lista de resultados de clusterização.
        cut_silhouette (float): O valor de corte para a silhueta.

    Returns:
        list: Uma lista de arranjos de cluster válidos.
    """
    valid_results = []
    for res in results:
        if res["mean_silhouette"] > cut_silhouette:
            is_valid = True
            for group in res["grouping_summary"]:
                if group["silhouette"] < cut_silhouette:
                    is_valid = False
                    break

            if is_valid:
                res["is_valid"] = True
                valid_results.append(res)
    return valid_results


def get_best_grouping_option(valid_results):
    """
    Seleciona o melhor arranjo de cluster entre os válidos.

    Prioriza o arranjo com a maior pontuação de silhueta.

    Args:
        valid_results (list): Uma lista de arranjos de cluster válidos.

    Returns:
        dict or None: O melhor arranjo de cluster ou None se a lista estiver vazia.
    """
    if not valid_results:
        return None

    best_grouping = max(valid_results, key=lambda x: x["mean_silhouette"])
    return best_grouping


def print_cluster_details(cluster):
    """
    Imprime os detalhes de um arranjo de cluster em formato de tabela HTML.

    Args:
        cluster (dict): Um dicionário contendo os detalhes do cluster.
    """
    html_content = f"""
    <h4> Detalhes do Arranjo</h4>
    <hr>
    <ul>
        <li><b>Arranjo</b>: {cluster["cluster"]}</li>
        <li><b>Qtd. Grupos</b>: {cluster["num_clusters"]}</li>
        <li><b>Silhueta média</b>: {cluster["score"]:.4f}</li>
        <li><b>Davies-Bouldin Index</b>: {cluster["davies_bouldin_scoree"]:.4f}</li>
        <li><b>Dunn Index</b>: {cluster["dunn_score"]:.4f}</li>
    </ul>
    <br/>
    <table>
        <tr><th>No.</th><th>Qtd. Registros</th><th>Silhueta</th></tr>
        {'</tr><tr>'.join(
    f'<td>{grupo["group"]}</td><td>{grupo["count"]}</td><td>{round(grupo["score"], 3)}</td>'
    for grupo in cluster["details"]
  )}
    </table>
    """
    return html_content


def get_comparision_clusters_graph(df, title=""):
    fig = px.bar(
        df,
        x="cluster",
        y="silhouette_score",
        color="cluster",
        title=title,
        text_auto=True,
    )

    fig.add_hline(
        y=0.5,
        line_width=2,
        line_color="red",
        line_dash="dot",
        annotation={"text": "threshold"},
        annotation_position="bottom right",
    )

    mean_silhouette = round(df["silhouette_score"].mean(), 2)

    fig.add_hline(
        y=mean_silhouette,
        line_dash="dot",
        annotation_text=f"mean: {mean_silhouette}",
        annotation_position="top right",
        line_color="orange",
    )

    fig.update(layout_yaxis_range=[-1, 1])
    fig.update_layout(
        height=360,
        width=720,
        showlegend=False,
    )

    return fig


def get_evaluate_cluster_graph(data):

    df_silhouette_label = pd.DataFrame(
        {"label": data["labels"], "silhouette": data["silhouettes"]}
    )

    df_silhouette_label.sort_values(by=["label", "silhouette"], inplace=True)

    df_silhouette_label["group"] = df_silhouette_label["label"].apply(
        lambda x: f"G{x+1} [{data['score']:.2f}]"
    )

    number_of_rows = math.ceil(data["num_clusters"] / 2)
    fig = make_subplots(
        rows=number_of_rows, cols=2, shared_yaxes=False, shared_xaxes=True
    )

    for group in range(data["num_clusters"]):
        group_data = df_silhouette_label[df_silhouette_label["label"] == group][
            "silhouette"
        ]
        fig.add_trace(
            go.Bar(
                x=group_data,
                name=f"G{group+1}",
            ),
            int(group / 2) + 1,
            group % 2 + 1,
        )

    fig.add_vline(
        x=0.5,
        line_width=1,
        line_dash="dash",
        line_color="red",
        annotation={"text": "threshold"},
        annotation_position="top left",
    )

    fig.add_vline(
        x=df_silhouette_label["silhouette"].mean(),
        line_width=2,
        line_dash="dot",
        annotation={"text": "mean"},
        annotation_position="bottom right",
    )

    fig.update_layout(
        title=f"Silhouette score - {data['num_clusters']} Groups",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def get_grouping_analysis_graph(arranjo, df):
    cluster = arranjo

    print(
        "For n_clusters =",
        cluster["qty_groups"],
        "The average silhouette_score is :",
        cluster["mean_silhouette"],
    )

    # Criar um gráfico com 1 linha e 2 colunas
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # Gráfico para exibir a silhueta
    # Definir os limites
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (cluster["qty_groups"] + 1) * 10])

    y_lower = 10

    for i in range(cluster["qty_groups"]):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = cluster["silhouettes"][cluster["labels"] == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / cluster["qty_groups"])
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=cluster["mean_silhouette"], color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster["labels"].astype(float) / cluster["qty_groups"])
    ax2.scatter(
        df.iloc[:, 1],
        df.iloc[:, 2],
        marker=".",
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    # Labeling the clusters
    centers = cluster["centroides"]
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % cluster["qty_groups"],
        fontsize=14,
        fontweight="bold",
    )

    plt.show()
