from models.models import Dataset
from pipeline.pipeline import RegressionPipeline


class RandomForestRegressionPipeline(RegressionPipeline):
    def __init__(self):
        self.target = None
        self.dataset: Dataset = None

    def run(self):
        # preprocess numeric vars

        # preprocess categorical vars

        pass
