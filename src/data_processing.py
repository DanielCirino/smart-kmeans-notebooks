"""
Data processing utilities for the smart K-means research project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import yaml
import os

from src.settings import load_config
from src.utils import get_file_path, get_root_dir


def load_raw_data(filename, data_path=None):
    """
    Load raw data from the specified path.

    Parameters:
    -----------
    filename : str
        Name of the data file
    data_path : str, optional
        Path to data directory. If None, uses config default.

    Returns:
    --------
    pandas.DataFrame
        Loaded dataset
    """
    if data_path is None:
        config = load_config()
        data_path = config['data']['raw_path']


    filepath = get_file_path([get_root_dir(),data_path, filename])

    if filename.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def prepare_data(df, cols_to_drop=list[str]):
    """
    Clean, prepare and select features for modeling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols_to_drop (list): List of columns to drop.
        id_column (str): The name of the ID column.

    Returns:
        pd.DataFrame: A new DataFrame with the prepared data.
    """

    df_prepared = df.copy()

    # 1. Drop unnecessary columns
    df_prepared = df_prepared.drop(columns=cols_to_drop)

    # 2. Handle missing values (in your research, we chose to remove them)
    if df_prepared.isnull().any().any():
      print("    -> Removing rows with null values.")
      df_prepared.dropna(inplace=True)

    return df_prepared



def save_processed_data(df, filename, data_path=None):
  """
  Save processed data to the processed data directory.

  Parameters:
  -----------
  df : pandas.DataFrame
      Dataset to save
  filename : str
      Output filename
  data_path : str, optional
      Path to processed data directory
  """
  if data_path is None:
    config = load_config()
    data_path = config['data']['processed_path']

  os.makedirs(data_path, exist_ok=True)


  filepath = get_file_path([get_root_dir(),data_path,filename])

  # Salvar em formato Parquet
  df.to_parquet(filepath, index=False, compression='gzip',)
  return filepath
