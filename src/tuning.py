import optuna
from omegaconf import OmegaConf

from src.models import build_model
from src.validation import cross_validate


def suggest_params(trial, space):
    """
    Берёт у trial значения параметров по описанию пространства из конфига.
    space: {имя: {type: int|float, low, high, step?, log?}}.
    """
    params = {}
    for name, spec in space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(
                name, spec.low, spec.high, step=spec.get("step", 1)
            )
        elif spec.type == "float":
            params[name] = trial.suggest_float(
                name, spec.low, spec.high, log=spec.get("log", False)
            )
        else:
            raise ValueError(f"Неизвестный тип параметра: {spec.type}")
    return params


def tune(config, X, y, X_test, model_name, n_trials=None):
    """
    Подбирает параметры модели model_name через Optuna.
    Пространство: config.tuning.space, фиксированные: config.tuning.fixed.
    Возвращает study (лучшее: study.best_value, study.best_params).
    """
    space = config.tuning.space[model_name]
    fixed = OmegaConf.to_container(
        config.tuning.fixed.get(model_name, {}), resolve=True
    )
    n_trials = n_trials or config.tuning.n_trials

    def objective(trial):
        params = suggest_params(trial, space)
        cfg = OmegaConf.merge(config, {"model": {"name": model_name}})
        cfg.model.params = {**fixed, **params}
        scores, _ = cross_validate(build_model(cfg), X, y, X_test, cfg)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
