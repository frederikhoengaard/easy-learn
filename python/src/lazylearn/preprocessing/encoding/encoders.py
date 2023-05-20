from models.models import Dataset
from pipeline.pipeline import ModelPipeline


class OrdinalConverter:
    def __init__(
        self,
        cat_vars: list,
        max_cardinality: int = None,
        min_support: int = 5,
        other_category: bool = True,
        method: str = "freq",
    ):
        self.cat_vars = cat_vars
        self.card_max = max_cardinality
        self.min_support = min_support
        self.other_category = other_category
        self.method = method
        self.cat_freqs = {}
        self.cat_maps = {}

    def fit(self, pipeline: ModelPipeline):
        for var in self.cat_vars:
            pipeline.train_features_df = self.convert(pipeline.train_features_df, var)
            pipeline.feature_list.append(var)

    def convert(self, df, col_name):
        """

        :param df:
        :param col_name:
        :return:
        """
        if self.method == "freq":
            self.cat_freqs[col_name] = {}
            for item in df[col_name].tolist():
                if item in self.cat_freqs[col_name]:
                    self.cat_freqs[col_name][item] += 1
                else:
                    self.cat_freqs[col_name][item] = 1

            freq_pairs = sorted(
                [(key, val) for key, val in self.cat_freqs[col_name].items()],
                key=lambda x: x[1],
            )
            print(freq_pairs)
            self.cat_maps[col_name] = {key: val for key, val in freq_pairs}

            df[col_name] = df[col_name].apply(
                lambda x: self.cat_maps[col_name][x]
                if self.cat_maps[col_name][x] >= self.min_support
                else -1
            )
            return df
        else:
            raise ValueError("Unsupported encoding method, try [freq]")
