from pandas import Series
from pipeline.pipeline import IngestionPipeline
from tqdm import tqdm


class ColumnTypeInterpreter:
    def apply(self, pipeline: IngestionPipeline):
        """
        This method is responsible for inferring the
        types of the columns of the project dataset

        :param pipeline: parent IngestionPipeline
        :return:
        """
        columns = pipeline.df.columns
        column_types = {}

        for column_name in tqdm(columns):
            column_types[column_name] = self.analyze_column(
                pipeline.df[column_name]
            )  # noqa

        pipeline.column_type_map = column_types

    def analyze_column(self, column: Series):
        # is it numeric?
        values = set(column)
        types = set([type(value) for value in values])

        if self.numeric_test(types):
            return "numeric"

        return "object"

    @staticmethod
    def numeric_test(types: set):
        return all([item == float or item == int for item in types])

    @staticmethod
    def string_test(types: set):
        raise NotImplementedError

    @staticmethod
    def date_check(types: set):
        raise NotImplementedError

    @staticmethod
    def categorical_test(values: set):
        raise NotImplementedError
