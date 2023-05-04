class Dataset:
    def __init__(self):
        self.name = None
        self.description = None


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
