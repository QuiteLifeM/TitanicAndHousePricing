import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'TitanicPassengersClassificationDataset'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'seed': 42,
    'paths': {
        'train': DATA_DIR / 'train.csv',
        'test': DATA_DIR / 'test.csv',
        'checkpoint_dir': CHECKPOINT_DIR,
        'train_preprocessed': CHECKPOINT_DIR / 'train_preprocessed.csv',
        'train_with_folds': CHECKPOINT_DIR / 'train_with_folds.csv',
        'train_with_folds_fe': CHECKPOINT_DIR / 'train_with_folds_fe.csv',  # после feature engineering
        'metadata_pickle': CHECKPOINT_DIR / 'preprocessing_metadata.pkl',
        'metadata_json': CHECKPOINT_DIR / 'preprocessing_metadata.json',
        'metrics_results': CHECKPOINT_DIR / 'metrics_results.csv',  # 04: классические модели
        'dl_results': CHECKPOINT_DIR / 'dl_results.csv',              # 05: DNN
        'ensemble_results': CHECKPOINT_DIR / 'ensemble_results.csv',  # 07: ансамбли
        'all_results': CHECKPOINT_DIR / 'all_results.csv',            # 08: итоговая сводка
    },
    'preprocessing': {
        'fill_age_with': 'median',       # 'median' | 'mean'
        'fill_embarked_with': 'mode',    # 'mode'
        'drop_cabin': True,              # True | False
        'log_fare': True,                # True | False
        'encode_method': 'label',        # 'label' | 'onehot'
        'categorical_cols': ['Sex', 'Embarked', 'Pclass']
    },
    'validation': {
        'n_splits': 5,                  # 1 | 5 | 10 | ...
        'strategy': 'stratified',       # 'stratified'
        'random_state': 42,
        'target_column': 'Survived',
        'shuffle': True,                # True | False
    },
    'models': {
        'logistic_regression': {
            'max_iter': 1000,
            'solver': 'lbfgs',          # 'lbfgs' | 'saga'
            'n_jobs': -1,
        },
        'knn': {
            'n_neighbors': 7,
            'weights': 'distance',       # 'uniform' | 'distance'
            'metric': 'minkowski',      # 'minkowski' | 'euclidean' | 'manhattan'
        },
        'decision_tree': {
            'max_depth': 5,
            'random_state': 42,
        },
        'random_forest': {
            'n_estimators': 300,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
        },
        'xgboost': {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
        },
        'lightgbm': {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,   # убрать [Info] и [Warning] "No further splits with positive gain"
        },
        'catboost': {
            'iterations': 400,
            'learning_rate': 0.05,
            'depth': 4,
            'loss_function': 'Logloss',
            'verbose': 100,
        },
        'dl': {
            'lr': 1e-3,
            'batch_size': 32,
            'n_epochs': 50,
            'dropout': 0.2,
            'optimizer': 'adam',             # 'adam' | 'sgd' | 'adamw'
            'scheduler': 'cosine',           # 'cosine' | 'step' | None
        },
    },
}