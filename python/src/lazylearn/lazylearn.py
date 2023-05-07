from ingestion.ingestion_pipeline import Ingestion


class LazyLearner:
    def __init__(self):
        self.dataset = None

    def create_project(self, data, target, task="infer"):
        # ingest data
        self.dataset = Ingestion().run(data)  # noqa

        # preprocess

        # set modelling configurations

        # train

        # eval
