import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.features import add_features


def load_data(config):
    """
    Загружает train и test, отделяет целевую переменную и id.
    Возвращает X, y, X_test, test_ids.
    """
    train = pd.read_csv(config.data.train_path)
    test = pd.read_csv(config.data.test_path)

    target = config.data.target
    id_col = config.data.id_column

    y = train[target]
    X = train.drop(columns=[target, id_col])
    X_test = test.drop(columns=[id_col])
    test_ids = test[id_col]

    X = add_features(X)
    X_test = add_features(X_test)

    # TicketGroup: сколько пассажиров едут по одному билету (считаем по train и test вместе).
    counts = pd.concat([X["Ticket"], X_test["Ticket"]]).value_counts()
    X["TicketGroup"] = X["Ticket"].map(counts)
    X_test["TicketGroup"] = X_test["Ticket"].map(counts)

    return X, y, X_test, test_ids


def save_submission(test_ids, test_pred, config, name):
    """
    Функция сохраняет результаты в файл submission для будущих сабмитов на Kaggle.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, f"submission_{name}.csv")

    submission = pd.DataFrame(
        {
            config.data.id_column: test_ids,
            config.data.target: (test_pred > 0.5).astype(int),
        }
    )
    submission.to_csv(path, index=False)
    return path


def build_preprocessor(config):
    """
    Собирает ColumnTransformer по спискам столбцов из конфига:
    numeric — медиана (+ StandardScaler),
    log_numeric — медиана + log1p (+ StandardScaler),
    categorical — самое частое + энкодер из config.preprocessing.encoder (onehot / ordinal / none).
    """
    encoder_name = config.preprocessing.encoder
    if encoder_name == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="if_binary"
        )
    elif encoder_name == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        encoder = None

    log_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p)),
    ]
    if config.preprocessing.scale:
        log_steps.append(("scaler", StandardScaler()))
    log_pipe = Pipeline(log_steps)

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if config.preprocessing.scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
    if encoder is not None:
        cat_steps.append(("encoder", encoder))
    cat_pipe = Pipeline(cat_steps)

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, list(config.features.numeric)),
            ("cat", cat_pipe, list(config.features.categorical)),
            ("log", log_pipe, list(config.features.log_numeric)),
        ],
        verbose_feature_names_out=False,
    )
    if encoder_name == "none":
        preprocessor.set_output(transform="pandas")
    return preprocessor
