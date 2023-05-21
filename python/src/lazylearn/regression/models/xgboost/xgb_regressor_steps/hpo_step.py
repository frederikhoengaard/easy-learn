from pipeline.pipeline import PipelineStep, RegressionPipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor


class HyperParameterOptimizationStep(PipelineStep):
    def __init__(self, n_splits=5, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.param_grid = {
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
            "gamma": [0, 0.25, 0.5, 1.0],
            "n_estimators": [100, 200, 300, 500, 1000],
        }

    def fit(self, pipeline: RegressionPipeline):
        xgbtuned = XGBRegressor()

        cv = KFold(n_splits=self.n_splits)
        xgbtunedreg = RandomizedSearchCV(
            xgbtuned,
            param_distributions=self.param_grid,
            scoring="neg_mean_squared_error",
            n_iter=20,
            n_jobs=-1,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
        )
        pipeline.regressor = xgbtunedreg

    def predict(self, pipeline: RegressionPipeline):
        pass
