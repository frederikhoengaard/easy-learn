from models.models import Dataset
from pipeline.pipeline import RegressionPipeline
from preprocessing.encoding.encoders import OrdinalConverter
from regression.models.randomforest.random_forest_steps.regressor_step import (
    RandomForestRegressorStep,
)
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressionRunner:
    def __init__(self, target, dataset):
        self.target = target
        self.dataset: Dataset = dataset
        self.pipeline = RegressionPipeline()

        self.pipeline.train_features_df = self.dataset.partitions["train"].copy()
        self.pipeline.train_targets = self.dataset.partitions["train"][target]
        self.pipeline.holdout_features_df = self.dataset.partitions["test"].copy()
        self.pipeline.holdout_targets = self.dataset.partitions["test"][target]

    def fit(self):
        # preprocess numeric vars
        cat_vars = self.dataset.type_collections["categorical"]

        self.pipeline.add(OrdinalConverter(cat_vars=cat_vars))

        # self.pipeline.add(RandomForestRegressorStep())

        self.pipeline.fit()

    def predict(self):
        raise NotImplementedError
