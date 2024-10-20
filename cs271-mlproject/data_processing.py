import polars as pl
from typing import Optional, Tuple, List, Dict, Set, Generator, Callable

from dataclasses import dataclass, field

from torch.utils.data import Dataset

from embeddings import ArticleEmbedder

import constants as pc

import numpy as np


@dataclass
class RawHumanChatBotData:
    """
    A raw dataset for the Human Chat Bot dataset with basic
    functionality to load and inspect the data.
    """
    data: pl.DataFrame = field(repr=False)
    available_types: Set[str] = field(init=False, repr=False)

    def __post_init__(self):
        self.available_types = set(self.data["type"].unique().to_list())

    @classmethod
    def from_csv(cls,
                 path: str,
                 shuffle: bool = True,
                 seed: Optional[int] = pc.R_SEED,
                 n_rows: Optional[int] = None
                 ) -> "RawHumanChatBotData":
        """Get the dataset from a CSV file.

        Args:
            path (str): the path of the dataset
            shuffle (bool): whether or not to shuffle the dataset
                after reading the csv file. Defaults to True.
            seed (Optional[int]): A random seed used for shuffling.
            n_rows (Optional[int]): How many total rows, after shuffling,
                to return.

        Returns:
            RawHumanChatBotData: The data.
        """
        data = cls._read_csv_file(
            path=path,
            shuffle=shuffle,
            seed=seed,
            n_rows=n_rows
        )
        return cls(data)

    @staticmethod
    def _read_csv_file(
            path: str,
            shuffle: bool = True,
            seed: Optional[int] = None,
            n_rows: Optional[int] = None) -> pl.DataFrame:
        """Read the dataset as CSV, shuffle with seed and
        return only the first n_rows of the underlying data.

        Args:
            path (str): The path of the csv file
            shuffle (bool, optional): whether to shuffle the rows. Defaults to True.
            seed (Optional[int], optional): a particular seed. Defaults to None.
                None will mean that the shuffled will be different everytime.
            n_rows (Optional[int], optional): how many rows, after shuffling
                to return from the underlying dataset. Defaults to None.
                Using None will return all rows

        Returns:
            pl.DataFrame: A dataframe
        """
        print(f"Reading the dataset from {path!r}")
        data = pl.read_csv(path)
        print(f"Performing shuffling? {shuffle}; with seed = {seed}")
        shuffled_data = data.sample(fraction=1, shuffle=shuffle, seed=seed)
        if n_rows is not None:
            print(f"Returning only the first {n_rows} rows")
            shuffled_data = shuffled_data.head(n_rows)
        return shuffled_data

    @classmethod
    def train_test_split(cls,
                         path: str,
                         seed: Optional[int] = pc.R_SEED,
                         n_rows: Optional[int] = None,
                         train_percent: int = pc.TRAIN_PERCENT,
                         ) -> Tuple["RawHumanChatBotData", "RawHumanChatBotData"]:
        shuffled_data = cls._read_csv_file(
            path=path,
            shuffle=True,
            seed=seed,
            n_rows=n_rows
        )
        train_size = int(len(shuffled_data) * train_percent)
        train_data = shuffled_data.head(train_size)
        test_data = shuffled_data.tail(-train_size)
        return RawHumanChatBotData(train_data), RawHumanChatBotData(test_data)

    def get_total_articles_of_type(self, article_type: str) -> int:
        """Get the total number of articles present of type article_type

        Args:
            article_type (str): the article_type

        Returns:
            int: the article_type
        """
        return len(self.data.filter(type=article_type))

    @property
    def n_gpt_entries(self) -> int:
        """
        Get the number of chatbot entries in the dataset.
        """
        return len(self.data.filter(type="gpt"))

    @property
    def total_articles(self) -> int:
        return len(self.data)

    def get_articles(self) -> Generator[str, None, None]:
        """Get multiple articles of the underlying as a generator.
        """
        for text in self.data["text"]:
            yield text


@dataclass
class ReusableGenerator:
    """
    A reusable generator is simply a wrapper around a generator function
    that allows you to create a new generator from the function every time
    you need to iterate over it.
    """
    generator_func: Callable[[], Generator]

    def __iter__(self):
        return self.generator_func()


@dataclass
class HumanChatBotDataset(Dataset):
    """
    A PyTorch Dataset class for the Human Chat Bot dataset.
    """
    data: pl.DataFrame
    embedder: ArticleEmbedder = field(repr=False)
    article_type_to_classnum: Dict[str, int]

    def __post_init__(self):
        # assert that all types in the dataset
        # have a corresponding article type mapping
        types_in_dataset = self.data["type"].unique().to_list()
        missing_types = set(types_in_dataset) - \
            set(self.article_type_to_classnum.keys())
        if missing_types:
            raise ValueError(
                f"Missing classnumber for types: {missing_types}")

    def get_class_num(self, article_type: str) -> int:
        """
        Get the class number for the given article type.
        """
        return self.article_type_to_classnum[article_type]

    @classmethod
    def from_raw_data(
        cls,
        raw_data: RawHumanChatBotData,
        embedder: ArticleEmbedder,
        article_type_to_classnum: Dict[str, int]
    ) -> "HumanChatBotDataset":
        return cls(
            raw_data.data,
            embedder,
            article_type_to_classnum
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row = self.data[idx]
        text = row["text"].item()
        article_type = row["type"].item()
        label = self.get_class_num(article_type)
        return self.embedder.embed(text), label
