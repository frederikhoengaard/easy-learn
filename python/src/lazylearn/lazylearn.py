from ingestion.ingestion_pipeline import Ingestion
from preprocessing.time.date_processor import date_processor


class LazyLearner:
    def __init__(self):
        self.dataset = None
        self.task = None
        self.models = None
        self.leaderboard = None

    def create_project(self, data, target, task="infer"):
        # ingest data
        self.dataset = Ingestion().run(data)

        if task == "infer":
            # if target is numeric then regression, else classification
            if self.dataset.column_type_map[target] == "numeric":
                self.task = "regression"
            else:
                self.task = "classification"

        # process dates

        self.dataset = date_processor(self.dataset)

        # preprocess

        # set modelling configurations

        # train

        # eval
