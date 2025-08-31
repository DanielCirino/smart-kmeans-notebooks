import pandas as pd
import pytest

from src.settings import load_config
from src.utils import get_file_path, get_root_dir


@pytest.fixture(scope="session")
def datasets():
    config = load_config()
    full_path = config["datasets"][5]

    # Nomes das colunas a serem ignoradas, conforme o notebook original
    cols_to_ignore = ["RendaMedia"]
    id_column = "Cod_Setor"

    df_original = pd.read_excel(full_path, dtype={id_column: "object"})
    return df_original, df_original.drop(columns=[id_column, *cols_to_ignore])
