import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validate(model, X, y, X_test, config):
    """
    Оценивает модель на K-Fold (или Stratified K-Fold по корзинам таргета).
    Возвращает RMSE по фолдам и усреднённые предсказания на test.
    """
    n_bins = config.get("stratify_bins", 0)
    if n_bins > 0:
        splitter = StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        groups = pd.qcut(y, q=n_bins, labels=False)
    else:
        splitter = KFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.seed
        )
        groups = y
    scorer = get_scorer(config.metric)

    test_pred = np.zeros(len(X_test))
    scores = []
    for train_idx, val_idx in splitter.split(X, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        scores.append(abs(scorer(fold_model, X_val, y_val)))
        test_pred += fold_model.predict(X_test)
    test_pred /= config.n_splits

    return np.array(scores), test_pred


def log_result(config, model_name, params, features, scores):
    """
    Функция дописывает строку с результатом эксперимента в файл из config.results_path
    """
    path = config.results_path
    os.makedirs(os.path.dirname(path), exist_ok=True)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": model_name,
        "params": str(params),
        "features": ",".join(features),
        "cv_mean": round(scores.mean(), 4),
        "cv_std": round(scores.std(), 4),
        "kaggle": None,
    }
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=not os.path.exists(path), index=False
    )
