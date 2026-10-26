import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold


def cross_validate(model, X, y, X_test, config):
    """
    Оценивает модель на Stratified K-Fold.
    Возвращает массив значений метрики, по одному на фолд, а также усреднённые вероятности на test.
    """
    skf = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed
    )
    scorer = get_scorer(config.metric)

    test_pred = np.zeros(len(X_test))
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        scores.append(scorer(fold_model, X_val, y_val))
        test_pred += fold_model.predict_proba(X_test)[:, 1]
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
