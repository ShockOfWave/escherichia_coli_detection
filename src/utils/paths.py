import os
from pathlib import Path

def get_project_path() -> str:
    return Path(__file__).parent.parent.parent

PATH_TO_RAW_DATA = os.path.join(get_project_path(), 'data', 'raw', 'database.csv')
PATH_TO_PROCESS = os.path.join(get_project_path(), 'models', 'process')