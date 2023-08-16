import os

import warnings

import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.data.make_dataset import process_train_dataset
from src.utils.paths import get_project_path, PATH_TO_PROCESS, PATH_TO_RAW_DATA

warnings.simplefilter("ignore")

version = 'version_2'

if __name__ == '__main__':
    
    data = pd.read_csv(PATH_TO_RAW_DATA)

    processed_dataset = process_train_dataset(data, save_process=True, version=version)

    # load processing pkl files
    sc = pickle.load(open(os.path.join(PATH_TO_PROCESS, version, 'StandardScaler.pkl'), 'rb'))
    norm = pickle.load(open(os.path.join(PATH_TO_PROCESS, version, 'Normalizer.pkl'), 'rb'))
    min_max = pickle.load(open(os.path.join(PATH_TO_PROCESS, version, 'MinMaxScaler.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(PATH_TO_PROCESS, version, 'LabelEncoder.pkl'), 'rb'))

    grid = {
        'depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'learning_rate': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'l2_leaf_reg': [1, 20, 50, 100],
        'iterations': [100, 300, 600, 1000, 5000],
        'early_stopping_rounds': [50],
        'verbose': [250]
    }

    names = ['pure_data', 'sc_data', 'norm_data', 'min_max_data']

    for i, name in enumerate(names):
        
        X_train, X_test, y_train, y_test = train_test_split(processed_dataset[0][i], processed_dataset[1], train_size=0.7, random_state=42, shuffle=True)
        
        if not os.path.exists(os.path.join(get_project_path(), 'models', 'catboost', version, name)):
            os.makedirs(os.path.join(get_project_path(), 'models', 'catboost', version, name))
        
        model = CatBoostClassifier(task_type='GPU', devices=[0, 1], train_dir=os.path.join(get_project_path(), 'models', 'catboost', version, name), bootstrap_type='Poisson', eval_metric='Accuracy', loss_function='MultiClass')
        grid_search_results = model.grid_search(grid, X_train, y_train, plot=False, cv=3, shuffle=True, stratified=True, train_size=0.3, refit=True)
        
        json.dump(grid_search_results, open(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'grid_search_results.json'), 'w'), indent=4)
        model.save_model(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'model'))
        
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        
        probs_train = model.predict_proba(X_train)
        probs_test = model.predict_proba(X_test)
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'conf_matrix_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'conf_matrix_train_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'roc_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'prec_recall_train.svg'), format='svg', dpi=1000)
        plt.close()
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'conf_matrix_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'conf_matrix_test_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'roc_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'catboost', version, name, 'prec_recall_test.svg'), format='svg', dpi=1000)
        plt.close()
