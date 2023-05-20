from pipeline.pipeline import PipelineStep, RegressionPipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class RandomForestRegressorStep(PipelineStep):
    def __init__(self):
        self.regressor = RandomForestRegressor()

    def fit(self, pipeline: RegressionPipeline):
        self.regressor.fit(X=pipeline.train_features_df, y=pipeline.train_targets)

        # y_hat = self.regressor.predict(X=pipeline.holdout_features_df)
        # pipeline.holdout_score = mean_absolute_error(pipeline.holdout_targets, y_hat)

    def predict(self, pipeline: RegressionPipeline):
        raise NotImplementedError
