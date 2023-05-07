from typing import List

from models.models import Dataset
from pandas import DataFrame


class Pipeline:
    def __init__(self):
        self._has_run: bool = False
        self._steps: List[PipelineStep] = []

    def add(self, pipeline_step):
        self._steps.append(pipeline_step)

    def run(self):
        [step.apply(self) for step in self._steps]
        self._has_run = True


class PipelineStep:
    def apply(self, pipeline: Pipeline):
        pass


class IngestionPipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.raw_data = None
        self.df: DataFrame = None
        self.column_type_map: dict = None
        self.summary_stats: dict = {}

    def response(self):
        return Dataset(df=self.df, column_type_map=self.column_type_map)
