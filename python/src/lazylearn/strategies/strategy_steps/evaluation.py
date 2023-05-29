from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


class Evaluator:
    def __init__(self):
        self.metrics = {
            "regression": [
                ("mae", mean_absolute_error),
                ("mse", mean_squared_error),
                ("mape", mean_absolute_percentage_error),
            ],
            "classification": [
                ("accuracy", accuracy_score),
                ("f1", f1_score),
                ("logloss", log_loss),
            ],
        }

    def evaluate(self, task, y_pred, y_true):
        return {name: func(y_true, y_pred) for name, func in self.metrics[task]}  # noqa
