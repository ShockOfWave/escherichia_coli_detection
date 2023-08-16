import os

import warnings
warnings.simplefilter("ignore")

import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from src.data.make_dataset import process_train_dataset
from src.utils.paths import get_project_path, PATH_TO_PROCESS, PATH_TO_RAW_DATA

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
        'hidden_layer_sizes': [(10, 10), (10, 50), (10, 100), (10, 250), (10, 500), (10, 10, 10), (10, 50, 25), (10, 100, 50), (10, 250, 125), (10, 500, 250),
                               (50, 10), (50, 50), (50, 100), (50, 250), (50, 500), (50, 10, 10), (50, 50, 25), (50, 100, 50), (50, 250, 125), (50, 500, 250),
                               (100, 10), (100, 50), (100, 100), (100, 250), (100, 500), (100, 10, 10), (100, 50, 25), (100, 100, 50), (100, 250, 125), (100, 500, 250)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive']
    }

    names = ['pure_data', 'sc_data', 'norm_data', 'min_max_data']

    for i, name in enumerate(names):
        
        X_train, X_test, y_train, y_test = train_test_split(processed_dataset[0][i], processed_dataset[1], train_size=0.7, random_state=42, shuffle=True)
        
        if not os.path.exists(os.path.join(get_project_path(), 'models', 'MLP', version, name)):
            os.makedirs(os.path.join(get_project_path(), 'models', 'MLP', version, name))
        
        model = MLPClassifier(max_iter=1000)
        
        clf = GridSearchCV(model, grid, n_jobs=40, cv=3, refit=True, verbose=2)
        
        clf.fit(X_train, y_train)
        
        joblib.dump(clf, os.path.join(get_project_path(), 'models', 'MLP', version, name, 'grid_search_results.pkl'))
        joblib.dump(clf.best_estimator_, os.path.join(get_project_path(), 'models', 'MLP', version, name, 'model.pkl'))
        
        pred_train = clf.best_estimator_.predict(X_train)
        pred_test = clf.best_estimator_.predict(X_test)
        
        probs_train = clf.best_estimator_.predict_proba(X_train)
        probs_test = clf.best_estimator_.predict_proba(X_test)
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'conf_matrix_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_train), le.inverse_transform(pred_train), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'conf_matrix_train_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'roc_train.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_train), probs_train)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'prec_recall_train.svg'), format='svg', dpi=1000)
        plt.close()
        
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=False, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'conf_matrix_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(pred_test), normalize=True, x_tick_rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'conf_matrix_test_norm.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_roc(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'roc_test.svg'), format='svg', dpi=1000)
        plt.close()
        skplt.metrics.plot_precision_recall(le.inverse_transform(y_test), probs_test)
        plt.tight_layout()
        plt.savefig(os.path.join(get_project_path(), 'models', 'MLP', version, name, 'prec_recall_test.svg'), format='svg', dpi=1000)
        plt.close()