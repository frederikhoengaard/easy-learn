class OrdinalConverter:
    def __init__(
        self,
        max_cardinality: int = None,
        min_support: int = 5,
        other_category: bool = True,
        method: str = "freq",
    ):
        self.card_max = max_cardinality
        self.min_support = min_support
        self.other_category = other_category
        self.method = method

    def convert(self, df, col):
        pass
