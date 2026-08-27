from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import torch
from torch import nn

from lpfn.losses import frobenius_loss

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainingResult:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    wall_time_seconds: float = 0.0


class Trainer:
    """Minimal deterministic full-batch trainer for reference experiments."""

    def __init__(
        self,
        *,
        epochs: int = 500,
        learning_rate: float = 0.03,
        seed: int = 0,
        loss_fn: LossFunction = frobenius_loss,
        restore_best: bool = True,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.loss_fn = loss_fn
        self.restore_best = bool(restore_best)

    def fit(
        self,
        model: nn.Module,
        *,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> TrainingResult:
        torch.manual_seed(self.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        result = TrainingResult()
        best_state: dict[str, torch.Tensor] | None = None

        start = perf_counter()
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(x_train)
            loss = self.loss_fn(pred, y_train)
            loss.backward()
            optimizer.step()
            result.train_loss.append(float(loss.detach()))

            model.eval()
            with torch.no_grad():
                val = float(self.loss_fn(model(x_val), y_val).detach())
            result.val_loss.append(val)
            if val < result.best_val_loss:
                result.best_val_loss = val
                result.best_epoch = epoch
                if self.restore_best:
                    best_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }

        result.wall_time_seconds = perf_counter() - start
        if self.restore_best and best_state is not None:
            model.load_state_dict(best_state)
        return result
