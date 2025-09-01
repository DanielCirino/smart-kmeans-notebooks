# Smart K-Means Analysis - Research Project

## Overview
This repository contains computational notebooks and analysis code for research on smart K-means clustering algorithms applied to social exclusion indicators. The project implements and compares various methods for determining the optimal number of clusters in K-means analysis.

## Project Structure
```
├── README.md                     # Project documentation
├── ALGORITHM.md                  # Detailed algorithm documentation and pseudocode
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration settings
├── LICENSE                       # MIT License
├── .gitignore                   # Git ignore rules
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb    # Exploratory Data Analysis
│   ├── 02_processing.ipynb          # Data cleaning and preprocessing
│   ├── 03_kmeans_analysis.ipynb     # K-means clustering analysis
│   └── 04_results_visualization.ipynb # Results visualization
├── src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── clustering_utils.py          # Clustering utilities and methods
│   ├── data_processing.py           # Data processing functions
│   ├── settings.py                  # Configuration management
│   ├── smart_k_means.py            # Smart K-means implementation
│   ├── utils.py                     # General utility functions
│   └── visualization.py            # Visualization functions
├── data/                        # Data directory
│   ├── raw/                        # Original, immutable datasets
│   ├── processed/                  # Cleaned and processed data
│   └── results/                    # Analysis outputs
│       ├── figures/                   # Generated plots and charts
│       ├── models/                    # Saved model parameters
│       └── tables/                    # Result tables and summaries
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   └── test_clustering_utils.py    # Clustering utilities tests
├── models/                      # Additional model storage
└── docs/                        # Documentation and paper materials
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/DanielCirino/smart-kmeans-notebooks.git
   cd smart-kmeans-notebooks
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Start Jupyter:
   ```bash
   jupyter lab
   ```

2. Run notebooks in sequential order:
   - **01_data_exploration.ipynb**: Perform exploratory data analysis on social exclusion indicators
   - **02_processing.ipynb**: Clean and preprocess the datasets
   - **03_kmeans_analysis.ipynb**: Execute K-means clustering analysis with multiple optimization methods
   - **04_results_visualization.ipynb**: Generate visualizations and result summaries

### Configuration
The project uses a `config.yaml` file to manage settings such as:
- Dataset paths and names
- Model parameters (random seeds, cluster ranges)
- Visualization preferences
- Output directories

## Methodology

This project implements and compares several methods for determining the optimal number of clusters in K-means analysis:

### Cluster Optimization Methods
- **Elbow Method**: Identifies the "elbow" point in the within-cluster sum of squares (WCSS) curve
- **Silhouette Analysis**: Measures how similar objects are to their own cluster compared to other clusters
- **Gap Statistic**: Compares total within-cluster variation with that expected under null reference distribution
- **Calinski-Harabasz Index**: Ratio of between-cluster dispersion to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity measure of each cluster with its most similar cluster

### Smart K-Means Features
- **Entropy-based Selection**: Uses linear entropy calculation to select optimal clustering arrangements
- **Multi-metric Evaluation**: Combines multiple validation indices for robust cluster assessment
- **Automated Parameter Selection**: Systematic evaluation across different cluster numbers
- **Iterative Feature Reduction**: Removes sub-indicators by entropy order (highest to lowest) until optimal clustering is achieved

📖 **For detailed algorithm documentation, mathematical foundation, and pseudocode, see [ALGORITHM.md](ALGORITHM.md)**

### Algorithm Prerequisites
The Smart K-Means algorithm assumes:
1. **Clean Numeric Data**: All input features are numeric and properly preprocessed
2. **Feature Selection**: Non-relevant columns (IDs, text) are excluded beforehand
3. **Data Quality**: No missing values in the feature set
4. **Appropriate Scale**: While the current implementation works with raw values, normalized/standardized features typically yield better clustering results

## Data

### Dataset Information
- **Source**: Social exclusion indicators from Brazilian municipalities (2010 Census)
- **Cities Included**: Apucarana, Cascavel, Foz do Iguaçu, Guarapuava, Londrina, Maringá, Ponta Grossa, Toledo
- **Variables**: Multiple socioeconomic indicators for census sectors

### Data Structure
- **Raw data**: Original Excel files in `data/raw/`
- **Processed data**: Cleaned datasets in Parquet format in `data/processed/`
- **Results**: Analysis outputs saved in `data/results/`

### Data Requirements and Preprocessing

⚠️ **Important**: The clustering algorithms in this project have specific data requirements:

#### **Data Format Requirements:**
1. **Numeric Data Only**: All columns used for clustering must contain only numeric values
2. **No Missing Values**: Missing values must be handled before clustering (current implementation removes rows with NaN)
3. **Pre-normalized Data**: While not explicitly implemented in the current version, K-means clustering typically requires feature scaling/normalization for optimal results

#### **Column Management:**
- **ID Columns**: Must be specified in `cols_to_ignore` and will be dropped before analysis
- **Text/Categorical Columns**: Must be listed in `cols_to_ignore` (e.g., `["RendaMedia", "Cod_Setor"]`)
- **Target Variables**: Any non-feature columns should be excluded from the clustering analysis

#### **Example Configuration:**
```python
# In your notebook or script
cols_to_ignore = ["RendaMedia"]  # Text or ID columns to exclude
id_column = "Cod_Setor"          # Primary identifier column

# Data is automatically filtered to numeric columns only
df_processed = df_original.drop(columns=[id_column, *cols_to_ignore])
```

⚠️ **Note**: Future versions should implement automatic feature scaling for improved clustering performance.

## Testing

The project includes unit tests for the clustering utilities. To run tests:

```bash
# Install testing dependencies if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_clustering_utils.py
```

## Reproducibility

All analyses are designed to be fully reproducible:
- **Fixed Random Seeds**: Consistent results across runs using `random_state=42`
- **Deterministic Processing**: All data processing steps are deterministic
- **Configuration Management**: All parameters controlled via `config.yaml`
- **Version Control**: Complete project history tracked in Git

## Key Features

- **Comprehensive Clustering Analysis**: Implementation of multiple cluster validation methods
- **Interactive Visualizations**: Plotly-based charts for exploration and presentation
- **Modular Architecture**: Well-structured, reusable code components
- **International Standards**: Code and documentation following international research practices
- **Extensive Documentation**: Detailed docstrings and comments throughout

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{smartkmeans2025,
  title={Smart K-Means Analysis for Social Exclusion Indicators},
  author={Daniel Cirino Martins},
  year={2025},
  publisher={GitHub},
  url={https://github.com/DanielCirino/smart-kmeans-notebooks}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## Contact
[Add your contact information]
