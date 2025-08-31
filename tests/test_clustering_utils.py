import pandas as pd

from src.clustering_utils import gap_statistic_method, get_grouping_suggestions
from src.smart_k_means import (
    calculate_best_k_with_entropy,
    evaluate_cluster,
    evaluate_grouping_options,
    get_grouping_analysis_graph,
)


def test_gap_statistics_method(datasets):
    res = gap_statistic_method(datasets[1], min_clusters=2, max_clusters=12)
    assert res is not None


def test_grouping_suggestion(datasets):
    res = get_grouping_suggestions(datasets[1])
    cluster_suggestions = set(res[0]["qty_clusters"].astype(int).values.tolist())
    cluster_evaluations = [
        evaluate_grouping_options(datasets, k, k) for k in cluster_suggestions
    ]

    for e in cluster_evaluations:
        get_grouping_analysis_graph(e[0], datasets)

    assert res is not None


def test_save_clustering_result(datasets):
    df_original, df_processed = datasets
    results = calculate_best_k_with_entropy(df_processed, 3, 7)

    best_k = results[0].iloc[0]
    best_k_labels = best_k["details"]["labels"]
    df_clustered = df_original
    df_clustered["CLUSTER"] = pd.Series(best_k_labels)
    df_clustered["CLUSTER"] = df_clustered["CLUSTER"].apply(lambda x: f"C{x+1}")
    df_clustered.to_csv("clusters.csv")
