from pipeline.pipeline import IngestionPipeline, PipelineStep


class Ingestion:
    def __init__(self):
        pass

    def run(self, data):
        pipeline = IngestionPipeline()
        pipeline.raw_data = data

        pipeline.add(DataSourceParser(data))

        pipeline.add(ColumnInterpreter())

        pipeline.run()

        return pipeline.response()
