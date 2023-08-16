import os

import warnings
warnings.simplefilter("ignore")

import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.data.make_dataset import process_train_dataset
from src.utils.paths import get_project_path, PATH_TO_PROCESS, PATH_TO_RAW_DATA

version = 'version_2'

if __name__ == '__main__':

    data = pd.read_csv(PATH_TO_RAW_DATA)

    processed_dataset = process_train_dataset(data, save_process=True, version=version)

    # load processing pkl files
    sc = pickle.load(open(os.path.join(PATH_TO_PROCESS, 'StandardScaler.pkl'), 'rb'))
    norm = pickle.load(open(os.path.join(PATH_TO_PROCESS, 'Normalizer.pkl'), 'rb'))
    min_max = pickle.load(open(os.path.join(PATH_TO_PROCESS, 'MinMaxScaler.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(PATH_TO_PROCESS, 'LabelEncoder.pkl'), 'rb'))

    grid = {
        'n_estimators': [10, 50, 100, 500, 1000, 5000],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }

    names = ['pure_data', 'sc_data', 'norm_data', 'min_max_data']

    for i, name in enumerate(names):
        
        X_train, X_test, y_train, y_test = train_test_split(processed_dataset[0][i], processed_dataset[1], train_size=0.7, random_state=42, shuffle=True)
        
        if not os.path.exists(os.path.join(get_project_path(), 'models', 'random_forest', version, name)):
            os.makedirs(os.path.join(get_project_path(), 'models', 'random_forest', version, name))
        
        model = RandomForestClassifier()
        
        clf = GridSearchCV(model, grid, n_jobs=23, cv=3, refit=True, verbose=2)
        
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'grid_search_results.pkl'))
        joblib.dump(clf.best_estimator_, os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'model.pkl'))
        
        pred_train = clf.best_estimator_.predict(X_train)
        pred_test = clf.best_estimator_.predict(X_test)
        
        probs_train = clf.best_estimator_.predict_proba(X_train)
        probs_test = clf.best_estimator_.predict_proba(X_test)
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'conf_matrix_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'conf_matrix_train_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'roc_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'prec_recall_train.svg'), format='svg', dpi=1000)
        plt.close()
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'conf_matrix_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'conf_matrix_test_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'roc_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'random_forest', version, name, 'prec_recall_test.svg'), format='svg', dpi=1000)
        plt.close()