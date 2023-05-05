from ingestion.ingestion_pipeline_steps.data_parser_step import DataSourceParser  # noqa
from ingestion.ingestion_pipeline_steps.interpreter_step import ColumnTypeInterpreter # noqa
from pipeline.pipeline import IngestionPipeline


class Ingestion:
    def __init__(self):
        pass

    def run(self, data):
        pipeline = IngestionPipeline()
        pipeline.raw_data = data

        pipeline.add(DataSourceParser(data))

        pipeline.add(ColumnTypeInterpreter())

        pipeline.run()

        return pipeline.response()
