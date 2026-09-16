"""Non-private training baseline used as a utility ceiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class NonPrivateConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 0.05
    seed: int = 0
    class_weights: list[float] | None = None

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass(frozen=True)
class NonPrivateResult:
    train_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    test_auroc: float | None
    test_auprc: float | None
    selected_windows: int
    config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return logits.argmax(dim=1).detach().cpu().numpy()


def _predict_proba(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=1).detach().cpu().numpy()


def _rank_metrics(y_true: np.ndarray, proba: np.ndarray) -> tuple[float | None, float | None]:
    classes = np.unique(y_true)
    if len(classes) < 2:
        return None, None
    try:
        if proba.shape[1] == 2:
            auroc = roc_auc_score(y_true, proba[:, 1])
            auprc = average_precision_score(y_true, proba[:, 1])
        else:
            labels = np.arange(proba.shape[1])
            one_hot = np.eye(proba.shape[1], dtype=int)[y_true]
            auroc = roc_auc_score(
                y_true,
                proba,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
            auprc = average_precision_score(one_hot, proba, average="macro")
    except ValueError:
        return None, None
    return float(auroc), float(auprc)


def train_nonprivate(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    config: NonPrivateConfig,
) -> NonPrivateResult:
    """Train a standard non-private classifier."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    class_weight_tensor = None
    if config.class_weights is not None:
        class_weight_tensor = torch.tensor(config.class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    model.train()
    for _epoch in range(config.epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    train_pred = _predict(model, train_x)
    test_pred = _predict(model, test_x)
    test_proba = _predict_proba(model, test_x)
    test_auroc, test_auprc = _rank_metrics(test_y, test_proba)
    return NonPrivateResult(
        train_accuracy=float(accuracy_score(train_y, train_pred)),
        test_accuracy=float(accuracy_score(test_y, test_pred)),
        test_macro_f1=float(f1_score(test_y, test_pred, average="macro", zero_division=0)),
        test_auroc=test_auroc,
        test_auprc=test_auprc,
        selected_windows=int(train_x.shape[0]),
        config=asdict(config),
    )
