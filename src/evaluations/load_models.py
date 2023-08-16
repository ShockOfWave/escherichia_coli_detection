import os
import joblib

from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from src.utils.paths import get_project_path

def load_catboost_model(path_to_model: str = os.path.join(get_project_path(), 'models', 'catboost', 'version_2', 'min_max_data', 'model')) -> CatBoostClassifier:
    model = CatBoostClassifier().load_model(path_to_model)
    return model

def load_mlp_model(path_to_model: str = os.path.join(get_project_path(), 'models', 'MLP_v2', 'min_max_data', 'model.pkl')) -> MLPClassifier:
    model = joblib.load(open(path_to_model, 'rb'))
    return model

def load_random_forest_model(path_to_model: str = os.path.join(get_project_path(), 'models', 'random_forest', 'min_max_data', 'model.pkl')) -> RandomForestClassifier:
    model = joblib.load(open(path_to_model, 'rb'))
    return model