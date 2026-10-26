from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.data import build_preprocessor
from src.nn import MLPClassifierTorch

MODELS = {
    "logreg": LogisticRegression,
    "knn": KNeighborsClassifier,
    "tree": DecisionTreeClassifier,
    "rf": RandomForestClassifier,
    "catboost": CatBoostClassifier,
    "lgbm": LGBMClassifier,
    "xgb": XGBClassifier,
    "mlp": MLPClassifierTorch,
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
