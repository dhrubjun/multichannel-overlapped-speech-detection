
import math


class EarlyStopping:
    """Track improvements in a metric that should be maximized."""

    def __init__(self, patience=5, min_delta=0.0):
        if patience < 1:
            raise ValueError("Patience must be at least 1")

        if min_delta < 0:
            raise ValueError("min_delta must be nonnegative")

        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.best_epoch = None
        self.bad_epochs = 0

    def update(self, score, epoch):
        if not math.isfinite(score):
            raise ValueError("Early-stopping score must be finite")

        improved = (
            self.best_score is None
            or score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        should_stop = self.bad_epochs >= self.patience

        return improved, should_stop