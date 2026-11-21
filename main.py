"""Воспроизводит лучшую модель из config.yaml: кросс-валидация и сабмит для Kaggle.
Запуск: python main.py
"""

from omegaconf import OmegaConf

from src.data import load_data, save_submission
from src.models import build_model
from src.validation import cross_validate


def main():
    cli = OmegaConf.from_cli()
    config = OmegaConf.merge(OmegaConf.load("config.yaml"), cli)
    X, y, X_test, test_ids = load_data(config)
    model = build_model(config)
    scores, test_pred = cross_validate(model, X, y, X_test, config)
    print(f"CV RMSE: {scores.mean():.4f} ± {scores.std():.4f}")
    path = save_submission(test_ids, test_pred, config, config.model.name)
    print(f"Сабмит сохранён: {path}")


if __name__ == "__main__":
    main()
