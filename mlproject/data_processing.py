import polars as pl
from typing import Optional, Tuple, Set, Generator, Callable, List, Dict
import pandas as pd
from dataclasses import dataclass, field
from mlproject import constants as pc
import json

from dataclasses_json import dataclass_json


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
class WikihowSubset:
    data: pl.DataFrame = field(repr=False)

    @classmethod
    def return_subset(
            cls,
            csv_path: str,
            # Default seed or pc.R_SEED if defined elsewhere
            seed: Optional[int] = None,
            n_rows: Optional[int] = 200000
    ) -> "WikihowSubset":
        data = cls._get_subset(
            csv_path=csv_path,
            seed=seed,
            n_rows=n_rows
        )
        return cls(data)

    def generate_queries(self) -> pl.DataFrame:
        print(self.data.columns)
        query_added = self.data.with_columns(
            pl.format(
                "I am writing an article titled '{}' for a WikiHow page. Write a paragraph of length {} whose title is '{}' for the {} section of this article.",
                pl.col("title"),
                pl.col("length"),
                pl.col("headline"),
                pl.col("sectionLabel")
            ).alias("query")
        )
        return query_added

    @staticmethod
    def _get_subset(
            csv_path: str,
            seed: Optional[int] = None,
            n_rows: Optional[int] = 200000) -> pl.DataFrame:
        print(f"Reading the dataset from {csv_path!r}")

        # Select specific columns
        selected_columns = ['title', 'headline', 'sectionLabel', 'text']

        data = pl.read_csv(csv_path, columns=selected_columns)
        data = data.drop_nulls(['text'])
        # Add a column for the original row number
        data = data.with_row_count("original_row_number")

        # Convert to Pandas for calculating word count
        df = data.to_pandas()

        df['title'] = (
            df['title']
            .str.strip()  # Remove leading/trailing spaces
            # Replace multiple spaces with a single space
            .replace(r'\s+', ' ', regex=True)
            .replace(r'\d+$', '', regex=True)  # Remove trailing numbers
        )

        df['word_count'] = df['text'].apply(
            lambda x: len(x.split()) if pd.notnull(x) else 0)
        df = df[df['word_count'] >= 40]
        df['length'] = df['word_count'].astype(str) + " words"
        # Drop the 'text' column after calculating 'length'
        df = df.drop(columns=['text', 'word_count'])

        # Convert back to Polars
        data = pl.from_pandas(df)

        # Sample n_rows rows with a specified seed for reproducibility
        print(f"Extracting subset of {n_rows} rows using seed {seed}.")
        sampled_data = data.sample(n=n_rows, seed=seed)
        return sampled_data


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


@dataclass_json
@dataclass
class NeuralNetworkExperimentResult:
    learning_rate: float
    epochs: int
    training_batch_size: int
    optimizer_name: str
    criterion_name: str
    training_accuracies: List[float] = field(default_factory=list)
    training_losses: List[float] = field(default_factory=list)
    testing_accuracies: List[float] = field(default_factory=list)
    testing_losses: List[float] = field(default_factory=list)
    # A single classification result is a dictionary with
    # N keys where N is the number of classes in the dataset.
    # This is the possible number of buckets samples can end up in.
    # Then, for each bucket, we analyze the make up. The make up
    # is how many samples (n) from class j are classified in bucket i.
    # As a result, to know how many camples from class j are classified
    # in bucket i, we can simply do classification_result[i][j].
    # However, because the classification results vary from epoch to epoch,
    # we store them in a list. The list is indexed by epoch.
    training_classification_results: List[Dict[int, Dict[int, int]]] = field(
        default_factory=list)
    testing_classification_results: List[Dict[int, Dict[int, int]]] = field(
        default_factory=list)

    def save(self, path: str):
        with open(path, mode="w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "NeuralNetworkExperimentResult":
        with open(path, mode="r") as f:
            json_obj = json.load(f)
            return cls.from_dict(json_obj)

    @property
    def training_loss(self):
        return self.training_losses[-1]

    @property
    def testing_loss(self):
        return self.testing_losses[-1]

    @property
    def testing_accuracy(self):
        return self.testing_accuracies[-1]

    @property
    def training_accuracy(self):
        return self.training_accuracies[-1]


@dataclass_json
@dataclass
class RunResult:
    original_dataset_name: str
    embedder_name: str
    experiment_result: NeuralNetworkExperimentResult

    def save(self, path: str):
        with open(path, mode="w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "RunResult":
        with open(path, mode="r") as f:
            json_obj = json.load(f)
            return cls.from_dict(json_obj)
