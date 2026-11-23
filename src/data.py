import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src.features import add_features


def fill_absent(df, config):
    """
    Заполняет пропуски, которые значат «объекта нет»:
    категории из fill_none → "None", числа из fill_zero → 0.
    """
    df = df.copy()
    none_cols = list(config.preprocessing.get("fill_none", []))
    zero_cols = list(config.preprocessing.get("fill_zero", []))
    df[none_cols] = df[none_cols].fillna("None")
    df[zero_cols] = df[zero_cols].fillna(0)
    return df


def encode_ordinal(df, config):
    """
    Порядковые категории → числа по шкалам из config.preprocessing.ordinal.
    Значения, которых нет в шкале, становятся NaN.
    """
    df = df.copy()
    ordinal = config.preprocessing.get("ordinal", None)
    if ordinal is None:
        return df
    for col, scale_name in ordinal.columns.items():
        mapping = dict(ordinal.scales[scale_name])
        df[col] = df[col].map(mapping)
    return df


def load_data(config):
    """
    Загружает train и test, убирает выбросы из data.drop_ids,
    отделяет цель (log1p, если log_target) и id.
    Затем заполняет пропуски «объекта нет», кодирует порядковые категории
    и добавляет новые признаки. Возвращает X, y, X_test, test_ids.
    """
    train = pd.read_csv(config.data.train_path)
    test = pd.read_csv(config.data.test_path)

    target = config.data.target
    id_col = config.data.id_column

    drop_ids = list(config.data.get("drop_ids", []))
    train = train[~train[id_col].isin(drop_ids)].reset_index(drop=True)

    y = train[target]
    if config.data.get("log_target", False):
        y = np.log1p(y)
    X = train.drop(columns=[target, id_col])
    X_test = test.drop(columns=[id_col])
    test_ids = test[id_col]

    X = fill_absent(X, config)
    X_test = fill_absent(X_test, config)

    X = encode_ordinal(X, config)
    X_test = encode_ordinal(X_test, config)

    X = add_features(X)
    X_test = add_features(X_test)

    return X, y, X_test, test_ids


def save_submission(test_ids, test_pred, config, name):
    """
    Функция сохраняет результаты в файл submission для будущих сабмитов на Kaggle.
    Переводит предсказания обратно из логарифма (expm1), если log_target.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, f"submission_{name}.csv")

    if config.data.get("log_target", False):
        test_pred = np.expm1(test_pred)

    submission = pd.DataFrame(
        {
            config.data.id_column: test_ids,
            config.data.target: test_pred,
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
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
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
