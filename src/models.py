from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.data import build_preprocessor

MODELS = {
    "linreg": LinearRegression,
    "lasso": Lasso,
    "ridge": Ridge,
    "elasticnet": ElasticNet,
    "knn": KNeighborsRegressor,
    "tree": DecisionTreeRegressor,
    "rf": RandomForestRegressor,
    "catboost": CatBoostRegressor,
    "lgbm": LGBMRegressor,
    "xgb": XGBRegressor,
}


def get_model(name, params):
    """Возвращает модель по имени из MODELS с параметрами params."""
    model_class = MODELS[name]
    # CatBoost: cat_features нужен кортежем, со списком ломается sklearn.clone
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return model_class(**params)


def build_model(config):
    """
    Собирает пайплайн: предобработка из конфига, затем модель из config.model
    """
    params = OmegaConf.to_container(config.model.params, resolve=True)
    model = get_model(config.model.name, params)
    return Pipeline(
        [
            ("prep", build_preprocessor(config)),
            ("model", model),
        ]
    )
