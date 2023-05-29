from models.models import Dataset
from sklearn.model_selection import train_test_split


def test_train_splitter(
    dataset: Dataset, test_size: float, random_state=None
) -> Dataset:
    train_partition, test_partition = train_test_split(
        dataset.df, test_size=test_size, random_state=random_state
    )

    dataset.partitions["test"] = test_partition
    dataset.partitions["train"] = train_partition

    return dataset


def time_test_train_splitter(dataset: Dataset, test_size: float) -> Dataset:
    n = len(dataset.df)

    if isinstance(test_size, float):
        test_size = int(n * test_size)
    elif isinstance(test_size, int):
        assert test_size < n
        # TODO: Implement a warning of test size seems too big

    dataset.partitions["test"] = dataset.df.tail(test_size)
    dataset.partitions["train"] = dataset.df.tail(n - test_size)

    return dataset


def cv_splitter(dataset: Dataset) -> Dataset:
    return dataset
