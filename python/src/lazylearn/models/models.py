from pandas import DataFrame


class Dataset:
    def __init__(self, df: DataFrame, column_type_map: dict):
        self.name = None
        self.description = None
        self.df = df
        self.column_type_map = column_type_map

    def save(self):
        raise NotImplementedError


class Model:
    def __init__(self):
        self.name = None

    def save(self, path: str):
        raise NotImplementedError


class Project:
    def __init__(self):
        self.name = None
        self.description = None

    def save(self, path: str):
        raise NotImplementedError
