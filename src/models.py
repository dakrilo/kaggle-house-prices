from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
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
from src.nn import MLPRegressorTorch

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
    "mlp": MLPRegressorTorch,
}


def get_model(name, params):
    """Возвращает модель по имени из MODELS с параметрами params."""
    model_class = MODELS[name]
    # CatBoost: cat_features нужен кортежем, со списком ломается sklearn.clone
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in params.items()}
    return model_class(**params)


def build_single(config, name, params):
    """Пайплайн: предобработка из конфига, затем модель name с параметрами params."""
    params = OmegaConf.to_container(params, resolve=True)
    return Pipeline(
        [
            ("prep", build_preprocessor(config)),
            ("model", get_model(name, params)),
        ]
    )


def build_model(config):
    """
    Собирает модель из config.model.
    name: voting — VotingRegressor из участников members с параметрами из config.tuned.
    Иначе одиночная модель с config.model.params.
    """
    if config.model.name == "voting":
        estimators = [
            (name, build_single(config, name, config.tuned[name]))
            for name in config.model.members
        ]
        return VotingRegressor(estimators=estimators)
    return build_single(config, config.model.name, config.model.params)
