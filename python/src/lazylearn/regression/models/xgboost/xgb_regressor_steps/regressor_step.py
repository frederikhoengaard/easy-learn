from pipeline.pipeline import PipelineStep, RegressionPipeline
from xgboost import XGBRegressor


class XGBRegressorStep(PipelineStep):
    def __init__(self, random_state=None):
        self.regressor = XGBRegressor(
            n_estimators=1000, random_state=random_state
        )  # noqa

    def fit(self, pipeline: RegressionPipeline):
        pipeline.feature_list = [
            item for item in pipeline.feature_list if item != pipeline.target
        ]
        print("Fitting XGBRegressor")
        self.regressor.fit(
            X=pipeline.train_features_df[pipeline.feature_list],
            y=pipeline.train_targets,
        )  # noqa
        print("XGBRegressor fitted!")

    def predict(self, pipeline: RegressionPipeline):
        pipeline.tmp_pred = self.regressor.predict(
            X=pipeline.tmp_test[pipeline.feature_list]
        )
