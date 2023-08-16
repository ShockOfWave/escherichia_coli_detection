import os
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, LabelEncoder
from src.utils.paths import PATH_TO_PROCESS

def process_train_dataset(dataset: pd.DataFrame, save_process: bool = False, version: str = 'version_0') -> tuple:
    """Process dataset with StandardScaler, Normalizer, MinMaxScaler and LabelEncoder. Returns tuple with:
    1. List with datasets
    2. Encoded y_data
    Save pkl files

    Args:
        dataset (pd.DataFrame): train dataset
        save_process (bool, optional): True or False to save pkl files. Defaults to False.

    Returns:
        list: list with 4 processed X data
        np.ndarray: y data
    """
    sc = StandardScaler()
    norm = Normalizer()
    min_max = MinMaxScaler()
    le = LabelEncoder()
    
    X_data = dataset.drop('Compound', axis=1)
    y_data = dataset.Compound.values
    
    X_data_sc = sc.fit_transform(X_data)
    X_data_norm = norm.fit_transform(X_data)
    X_data_min_max = min_max.fit_transform(X_data)
    
    y_data = le.fit_transform(y_data)
    
    if save_process:
        if not os.path.exists(os.path.join(PATH_TO_PROCESS, version)):
            os.makedirs(os.path.join(PATH_TO_PROCESS, version))
        
        pickle.dump(sc, open(os.path.join(PATH_TO_PROCESS, version, 'StandardScaler.pkl'), 'wb'))
        pickle.dump(norm, open(os.path.join(PATH_TO_PROCESS, version, 'Normalizer.pkl'), 'wb'))
        pickle.dump(min_max, open(os.path.join(PATH_TO_PROCESS, version, 'MinMaxScaler.pkl'), 'wb'))
        pickle.dump(le, open(os.path.join(PATH_TO_PROCESS, version, 'LabelEncoder.pkl'), 'wb'))
        
    return ([X_data, X_data_sc, X_data_norm, X_data_min_max], y_data)