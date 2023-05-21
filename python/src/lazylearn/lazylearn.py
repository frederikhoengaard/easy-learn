from ingestion.ingestion_pipeline import Ingestion
from model_selection.splitters import test_train_splitter
from preprocessing.time.date_processor import date_processor
from preprocessing.time.duration import duration_builder
from regression.models.randomforest.randomforest import (  # noqa
    RandomForestRegressionRunner,
)
from strategies.strategy_builder import StrategyBuilder


class LazyLearner:
    def __init__(self, random_state=None):
        self.dataset = None
        self.task = None
        self.models = None
        self.leaderboard = None
        self.random_state = random_state
        self.target = None

    def create_project(self, data, target, task="infer"):
        # ingest data
        self.target = target
        self.dataset = Ingestion().run(data)

        if task == "infer":
            # if target is numeric then regression, else classification
            if self.dataset.column_type_map[target] == "numeric":
                self.task = "regression"
            else:
                self.task = "classification"

        # process dates

        self.dataset = date_processor(self.dataset)
        self.dataset = duration_builder(self.dataset)

        # split partitions

        self.dataset = test_train_splitter(
            self.dataset, random_state=self.random_state
        )  # noqa

        # set modelling configurations

    def run_autopilot(self):
        """
        TODO: Everything here must be abstracted away into strategies
        TODO: such that several models are run and their scores are added to
        TODO: the leaderboard

        :return:
        """
        sb = StrategyBuilder(task=self.task, dataset=self.dataset, target=self.target, random_state=self.random_state)
        self.leaderboard = sorted([model for model in sb.models], key=lambda x: x.score)

    def get_leaderboard(self):
        return [(item.name, item.score) for item in self.leaderboard]

