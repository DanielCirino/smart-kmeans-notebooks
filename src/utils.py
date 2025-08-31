import os


def get_root_dir() -> str:
  """

  :return:
  """
  file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  root_dir = os.sep.join(file_dir.split(os.sep))
  return root_dir


def get_file_path(paths: list[str]) -> str:
  """

  :param paths:
  :return:
  """
  return os.sep.join(paths)
