# Smart K-Means Algorithm Documentation

## Overview

This document describes the Smart K-Means algorithm with entropy-based feature selection, implemented in this research project. The algorithm systematically determines the optimal number of clusters while simultaneously selecting the most relevant features based on Shannon entropy analysis.

## Algorithm Version

**Version:** 1.0.0  
**Date:** August 31, 2025  
**Authors:** 
- Daniel Cirino Martins ([ORCID: 0009-0002-5304-9185](https://orcid.org/0009-0002-5304-9185)) - UNIMONTES
- Marcos Flávio Silveira Vasconcelos D'Angelo - UNIMONTES
- Matheus Pereira Libório - UNIMONTES
- Petr Ekel - PUC Minas
- Honovan Paz Rocha - UNIMONTES
- Douglas Alexandre Gomes Vieira - CEFET-MG
- Laura Cozzi Ribeiro - PUC Minas
- Maria Fernanda Oliveira Guimarães - PUC Minas  
**Implementation:** `src/smart_k_means.py::calculate_best_k_with_entropy()`
**Published:** SBPO 2024 - LVI Simpósio Brasileiro de Pesquisa Operacional

## Algorithm Description

The Smart K-Means algorithm combines traditional K-means clustering with an intelligent feature selection mechanism based on Shannon entropy. The algorithm iteratively removes features with the highest entropy (most variability/noise) until it finds a clustering configuration that meets the quality threshold.

## Pseudocode

```
// SMART K-MEANS ALGORITHM WITH ENTROPY-BASED FEATURE SELECTION
// Version 1.0.0

// PRE-CONDITIONS:
// - Input DataFrame must contain only numeric columns for clustering
// - ID columns and text/categorical variables must be pre-excluded  
// - Missing values must be handled (removed or imputed) beforehand
// - Feature scaling/normalization recommended but not implemented in current version

// Input: Clean numeric DataFrame, range for number of clusters (k_min, k_max)

// Step 1: Data Validation and Entropy Calculation
1. VALIDATE that all columns in DataFrame are numeric
2. CALCULATE the Shannon entropy for each sub-indicator in the DataFrame
3. SORT the sub-indicators by entropy, from highest to lowest

// Step 2: Iterative Evaluation Process  
4. FOR each value of 'k' in the range from k_min to k_max:
   5. INITIALIZE a list of sub-indicators with all variables
   6. SET iteration counter = 1
   7. FOR each sub-indicator to be removed in the entropy-ordered list:
      8. EXECUTE the K-Means algorithm on the data using the current set of sub-indicators
      9. CALCULATE clustering metrics:
         - Silhouette score
         - Davies-Bouldin score
         - Dunn score
      10. STORE the iteration details (k, iteration, excluded_indicator, entropy, all_scores)
      11. IF the Silhouette score is greater than 0.5:
          12. STORE the valid evaluation result (k, num_features, scores, feature_list, details)
          13. BREAK this inner loop and proceed to the next value of 'k'
      14. REMOVE the current sub-indicator from the list for the next iteration
      15. INCREMENT iteration counter

// Step 3: Select Optimal Result
16. FILTER all evaluation results that met the quality threshold
17. SORT the valid results by priority:
    - Quantity of sub-indicators used (descending - prefer more features)
    - Silhouette score (descending - prefer better clustering quality)  
    - Number of clusters 'k' (ascending - prefer simpler solutions)
18. SELECT the first result from this sorted list as the optimal solution

// Output: Comprehensive results package
RETURN (
  evaluation_results_dataframe,    // All valid clustering arrangements
  entropy_ranking_dataframe,       // Feature entropy rankings  
  iterations_summary_dataframe,    // Detailed iteration log
  original_column_names,           // Feature names used
  original_dataframe              // Input data reference
)

// POST-CONDITIONS:
// - Returns optimal clustering configuration with maximum feature retention
// - Provides full audit trail of the selection process
// - Enables reproducible results with documented parameters
```

## Algorithm Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | Required | Input dataset with numeric features only |
| `min_clusters` | `int` | 3 | Minimum number of clusters to evaluate |
| `max_clusters` | `int` | 7 | Maximum number of clusters to evaluate |
| `silhouette_threshold` | `float` | 0.5 | Quality threshold for accepting clustering results |

## Key Features

### 1. Entropy-Based Feature Selection
- **Shannon Entropy Calculation**: Measures the information content of each feature
- **Descending Order Removal**: Features with highest entropy (most noise) are removed first
- **Systematic Evaluation**: Each removal step is evaluated for clustering quality

### 2. Multi-Metric Evaluation
- **Silhouette Score**: Measures cluster separation and cohesion
- **Davies-Bouldin Index**: Evaluates cluster compactness and separation
- **Dunn Index**: Ratio of minimum inter-cluster to maximum intra-cluster distance

### 3. Intelligent Selection Criteria
- **Feature Retention Priority**: Prefers solutions that use more features
- **Quality Assurance**: Only accepts solutions above the silhouette threshold
- **Simplicity Preference**: Among equal solutions, prefers fewer clusters

## Implementation Details

### Core Functions

1. **`calculate_dataset_entropy(df)`**
   - Computes Shannon entropy for all DataFrame columns
   - Returns sorted DataFrame with entropy values

2. **`evaluate_cluster(df, n_clusters)`**
   - Performs K-means clustering with specified parameters
   - Calculates all validation metrics
   - Returns comprehensive clustering assessment

3. **`calculate_best_k_with_entropy(df, min_clusters, max_clusters)`**
   - Main algorithm implementation
   - Orchestrates the entropy-based selection process
   - Returns complete results package

### Data Flow

```
Input DataFrame
    ↓
Entropy Calculation & Sorting
    ↓
For each k in range:
    ↓
Initialize full feature set
    ↓
For each feature (by entropy order):
    ↓
Execute K-means → Calculate metrics
    ↓
Quality check (Silhouette > 0.5)
    ↓
[YES] Store result & break → Next k
[NO] Remove feature & continue
    ↓
Rank all valid results
    ↓
Return optimal solution
```

## Mathematical Foundation

### Shannon Entropy Formula
```
H(X) = -Σ P(xi) * log2(P(xi))
```
Where P(xi) is the probability of value xi in the feature distribution.

### Silhouette Score Formula
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```
Where:
- a(i) = average distance to points in same cluster
- b(i) = average distance to points in nearest other cluster

## Validation and Quality Assurance

### Quality Thresholds
- **Silhouette Score > 0.5**: Ensures reasonable cluster separation
- **Minimum Features**: Algorithm preserves maximum possible features
- **Reproducibility**: Fixed random seeds ensure consistent results

### Performance Characteristics
- **Time Complexity**: O(k × f × n × i) where k=clusters, f=features, n=samples, i=iterations
- **Space Complexity**: O(n × f) for data storage plus O(k × r) for results where r=valid results
- **Scalability**: Suitable for datasets with moderate feature counts (< 50 features recommended)

## Limitations and Future Improvements

### Current Limitations
1. **No Automatic Scaling**: Requires pre-normalized data for optimal results
2. **Fixed Threshold**: Silhouette threshold is hard-coded at 0.5
3. **Single Metric Primary**: Uses silhouette as primary quality gate
4. **No Categorical Handling**: Requires numeric-only input data

### Proposed Improvements
1. **Adaptive Thresholds**: Dynamic quality thresholds based on data characteristics
2. **Multi-Objective Optimization**: Balance multiple metrics simultaneously  
3. **Automatic Preprocessing**: Integrated scaling and categorical encoding
4. **Parallel Processing**: Distribute k-value evaluations across cores
5. **Feature Interaction Analysis**: Consider feature correlations in selection

## Usage Example

```python
from src.smart_k_means import calculate_best_k_with_entropy

# Prepare clean numeric DataFrame
df_clean = df.select_dtypes(include=[np.number])

# Execute Smart K-Means
results, entropy_data, iterations, columns, original_df = \
    calculate_best_k_with_entropy(df_clean, min_clusters=3, max_clusters=7)

# Access optimal solution
optimal_solution = results.iloc[0] if len(results) > 0 else None
```

## References and Citations

This algorithm builds upon established clustering validation methods:

1. **Silhouette Analysis**: Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.
2. **Shannon Entropy**: Shannon, C. E. (1948). A mathematical theory of communication.
3. **K-means Clustering**: Lloyd, S. (1982). Least squares quantization in PCM.
4. **Davies-Bouldin Index**: Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-08-31 | Initial algorithm documentation |

---

For implementation details and code examples, see the corresponding Python modules in the `src/` directory.
