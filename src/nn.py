import numpy as np

import torch
from torch import nn

from sklearn.base import BaseEstimator, ClassifierMixin

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


class MLPClassifierTorch(ClassifierMixin, BaseEstimator):
    """Полносвязная сеть на PyTorch в обёртке sklearn: работает в Pipeline, CV и ансамблях."""

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

    def _build_net(self, n_features):
        """Собирает сеть: n_layers × (Linear → [BatchNorm] → активация → [Dropout]) и выход Linear на 1 логит."""
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
        """Обучает сеть мини-батчами на BCEWithLogitsLoss."""
        torch.manual_seed(self.random_state)
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        y = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)

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

        loss_fn = nn.BCEWithLogitsLoss()

        self.net_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.net_(X[idx]), y[idx])
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Возвращает вероятности классов: столбцы [P(0), P(1)]."""
        X = torch.tensor(np.asarray(X, dtype=np.float32))
        self.net_.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net_(X)).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Возвращает метки 0/1 по порогу 0.5."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
