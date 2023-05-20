from ingestion.ingestion_pipeline import Ingestion
from model_selection.splitters import test_train_splitter
from preprocessing.time.date_processor import date_processor
from preprocessing.time.duration import duration_builder
from regression.models.randomforest.randomforest import RandomForestRegressionRunner


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
        simple_random_forest = RandomForestRegressionRunner(
            target=self.target, dataset=self.dataset
        )
        simple_random_forest.fit()

        return simple_random_forest
