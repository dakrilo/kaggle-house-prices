import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn
from tqdm import tqdm

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


class MLPRegressorTorch(RegressorMixin, BaseEstimator):
    """Полносвязная сеть-регрессор на PyTorch в обёртке sklearn: работает в Pipeline, CV и ансамблях."""

    def __init__(
        self,
        hidden_size=32,
        n_layers=1,
        batchnorm=False,
        dropout=0.0,
        activation="relu",
        optimizer="adam",
        weight_decay=0.0,
        scheduler=None,
        lr=1e-3,
        epochs=100,
        batch_size=64,
        random_state=42,
        verbose=False,
    ):
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.activation = activation
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _build_net(self, n_features):
        """Собирает сеть: n_layers × (Linear → [BatchNorm] → активация → [Dropout]) и выход Linear на 1 выход."""
        layers = []
        in_size = n_features
        for _ in range(self.n_layers):
            layers.append(nn.Linear(in_size, self.hidden_size))
            if self.batchnorm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(ACTIVATIONS[self.activation]())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_size = self.hidden_size
        layers.append(nn.Linear(in_size, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        """Обучает сеть мини-батчами на MSELoss. Цель нормируется внутри."""
        torch.manual_seed(self.random_state)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_ = y.mean()
        self.y_std_ = y.std()
        y = torch.tensor((y - self.y_mean_) / self.y_std_).reshape(-1, 1)

        self.net_ = self._build_net(X.shape[1])

        params = self.net_.parameters()
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                params, lr=self.lr, weight_decay=self.weight_decay
            )

        scheduler = None
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )

        loss_fn = nn.MSELoss()

        self.net_.train()
        epochs = range(self.epochs)
        if self.verbose:
            epochs = tqdm(epochs, desc="Обучение")
        for epoch in epochs:
            perm = torch.randperm(len(X))
            epoch_loss = 0.0
            for start in range(0, len(X), self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net_(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            if scheduler is not None:
                scheduler.step()
            if self.verbose:
                epochs.set_postfix(loss=epoch_loss / len(X))
        return self

    def predict(self, X):
        """Возвращает предсказания в исходной шкале цели (обратная нормировка)."""
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(X).numpy().ravel()
        return out * self.y_std_ + self.y_mean_
