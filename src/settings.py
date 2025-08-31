import os
import sys
import yaml

from src.utils import get_root_dir, get_file_path

src_path = os.path.abspath(os.path.join(os.getcwd(), "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


def load_config() -> dict:
    """
    Loads the configuration file
    :param
    :return:
    """
    root_dir = get_root_dir()

    config_path = get_file_path([root_dir, "config.yaml"])

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        raw_dir = get_file_path([root_dir, "data", config["data"]["raw_path"]])
        processed_dir = get_file_path(
            [root_dir, "data", config["data"]["processed_path"]]
        )

        config["datasets"] = [get_file_path([raw_dir, d]) for d in config["datasets"]]

        return config
