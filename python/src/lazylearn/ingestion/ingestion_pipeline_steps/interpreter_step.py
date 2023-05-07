import pandas as pd
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
        self.df = pipeline.df
        columns = pipeline.df.columns
        column_types = {}

        for column_name in tqdm(columns):
            column_types[column_name] = self.analyze_column(
                pipeline.df[column_name]
            )  # noqa

        pipeline.column_type_map = column_types

    def analyze_column(self, column: Series):
        """

        :param column:
        :return:
        """
        values = column.tolist()
        types = [type(value) for value in values]

        if self.categorical_test(values):
            return "categorical"

        elif self.numeric_test(types):
            return "numeric"

        elif self.datetime_check(column):
            return "datetime"
        else:
            return "object"

    @staticmethod
    def categorical_test(values: list):
        """
        Tests whether a column is of categorical type.
        This is decided as the case if the number of unique values is
        less than 5% of the total number of values in the column.

        :param values: list of values of any type
        :return: True if column is categorical, False otherwise
        """
        n_total = len(values)
        n_unique = len(set(values))
        percentage_unique = n_unique / n_total

        if percentage_unique < 0.05:
            return True
        return False

    @staticmethod
    def numeric_test(types: list):
        """
        Tests whether a column is of numeric tyoe.
        This is decided as the case if all values
        of a column is either float or int.

        :param types: list of type objects
        :return: True if column is numeric, False otherwise
        """
        return all([item == float or item == int for item in set(types)])

    @staticmethod
    def string_test(types: set):
        raise NotImplementedError

    def datetime_check(self, column: Series):
        try:
            self.df[column.name] = pd.to_datetime(column)
            return True
        except Exception as e:  # noqa
            return False
