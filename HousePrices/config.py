from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "HousePricesRegressionDataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

CONFIG = {
    "seed": 42,
    "paths": {
        "train": DATA_DIR / "train.csv",
        "test": DATA_DIR / "test.csv",
        "checkpoint_dir": CHECKPOINT_DIR,
        "train_preprocessed": CHECKPOINT_DIR / "train_preprocessed.csv",
        "test_preprocessed": CHECKPOINT_DIR / "test_preprocessed.csv",
        "train_with_folds": CHECKPOINT_DIR / "train_with_folds.csv",
        "train_with_folds_fe": CHECKPOINT_DIR / "train_with_folds_fe.csv",  # после feature engineering
        "metadata_pickle": CHECKPOINT_DIR / "preprocessing_metadata.pkl",
        "metadata_json": CHECKPOINT_DIR / "preprocessing_metadata.json",
        "metrics_results": CHECKPOINT_DIR / "metrics_results.csv",  # 04: классические модели
        "dl_results": CHECKPOINT_DIR / "dl_results.csv",  # 05: DNN
        "ensemble_results": CHECKPOINT_DIR / "ensemble_results.csv",  # 07: ансамбли
        "all_results": CHECKPOINT_DIR / "all_results.csv",  # 08: итоговая сводка
        "submission": CHECKPOINT_DIR / "submission.csv",
    },
    "preprocessing": {
        "id_column": "Id",
        "target_column": "SalePrice",
        "target_transform": "log1p",  # None | 'log1p'
        "numeric_imputer": "median",  # 'median' | 'mean'
        "categorical_imputer": "most_frequent",  # 'most_frequent'
        "scale_numeric": True,  # True | False
        "encode_method": "onehot",  # 'label' | 'onehot'
        "max_onehot_levels": None,  # None = без ограничения
    },
    "validation": {
        "n_splits": 5,  # 1 | 5 | 10 | ...
        "strategy": "kfold",  # 'kfold'
        "random_state": 42,
        "shuffle": True,  # True | False
        "target_column": "SalePrice",  # для совместимости с ноутбуками
    },
    "models": {
        "linear_regression": {},
        "ridge": {"alpha": 10.0, "random_state": 42},
        "lasso": {"alpha": 0.0005, "random_state": 42, "max_iter": 20000},
        "elasticnet": {
            "alpha": 0.0005,
            "l1_ratio": 0.5,
            "random_state": 42,
            "max_iter": 20000,
        },
        "knn": {"n_neighbors": 7, "weights": "distance", "metric": "minkowski"},
        "decision_tree": {"max_depth": 6, "random_state": 42},
        "random_forest": {"n_estimators": 800, "max_depth": None, "random_state": 42, "n_jobs": -1},
        "xgboost": {
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "max_depth": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
        },
        "lightgbm": {
            "n_estimators": 5000,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "regression",
            "metric": "rmse",
            "verbose": -1,
        },
        "catboost": {
            "iterations": 5000,
            "learning_rate": 0.02,
            "depth": 6,
            "loss_function": "RMSE",
            "verbose": 200,
        },
        "dl": {
            "lr": 1e-3,
            "batch_size": 64,
            "n_epochs": 80,
            "dropout": 0.2,
            "optimizer": "adam",  # 'adam' | 'sgd' | 'adamw'
            "scheduler": "cosine",  # 'cosine' | 'step' | None
        },
    },
}

