import polars as pl
from typing import Optional, Tuple, List, Dict, Set

from dataclasses import dataclass, field

from torch.utils.data import Dataset

from embeddings import ArticleEmbedder

import numpy as np


@dataclass
class RawHumanChatBotData:
    """
    A raw dataset for the Human Chat Bot dataset with basic
    functionality to load and inspect the data.
    """

    data: pl.DataFrame = field(repr=False)
    n_articles_per_type: Optional[int] = None
    available_types: Set[str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_articles_per_type is not None:
            get_all_types = self.data["type"].unique().to_list()
            series = []
            for i in get_all_types:
                partial_series = self.data.filter(type=i)
                if len(partial_series) < self.n_articles_per_type:
                    raise ValueError(
                        f"Type {i} has less than {self.n_articles_per_type} articles.")
                series.append(self.data.filter(type=i)[
                              :self.n_articles_per_type])

            self.data = pl.concat(series)
        self.available_types = set(self.data["type"].unique().to_list())

    @classmethod
    def from_csv(cls,
                 path: str,
                 n_articles_per_type: Optional[int] = None
                 ) -> "RawHumanChatBotData":
        """Get the dataset from a CSV file.

        Args:
            path (str): the path of the dataset
            n_articles_per_type (Optional[int], optional):
                Use it to truncate the internal data. Defaults to None.

        Returns:
            RawHumanChatBotData: The data.
        """
        data = pl.read_csv(path)
        return cls(data, n_articles_per_type)

    @property
    def n_human_entries(self) -> int:
        """
        Get the number of human entries in the dataset.
        """
        return len(self.data.filter(type="human"))

    @property
    def n_gpt_entries(self) -> int:
        """
        Get the number of chatbot entries in the dataset.
        """
        return len(self.data.filter(type="gpt"))

    def get_combined_text(self, article_type: str, n_paragraphs: Optional[int] = None) -> str:
        """Get multiple articles of the given type merged into a single string.

        Args:
            article_type (str): the article type
            n_paragraphs (Optional[int], optional): The number of paragraphs.
                Defaults to None. None means all paragraphs.

        Returns:
            str: a single string with all the paragraphs.
        """
        if article_type not in self.available_types:
            raise ValueError(f"Type {article_type} not in available types.")
        filtered_data_text = self.data.filter(
            type=article_type)["text"]
        if n_paragraphs is not None:
            filtered_data_text = filtered_data_text[:n_paragraphs]
        list_of_text = filtered_data_text.to_list()
        return "".join(list_of_text)


@dataclass
class HumanChatBotDataset(Dataset):
    """
    A PyTorch Dataset class for the Human Chat Bot dataset.
    """
    data: pl.DataFrame
    embedder_per_type: Dict[str, ArticleEmbedder] = field(repr=False)
    article_type_to_classnum: Dict[str, int]

    def __post_init__(self):
        # all types have have an embedder
        types_in_dataset = self.data["type"].unique().to_list()
        # create a set difference for better reporting
        missing_types = set(types_in_dataset) - \
            set(self.embedder_per_type.keys())
        if missing_types:
            raise ValueError(
                f"Missing embedder for types: {missing_types}")

    def get_class_num(self, article_type: str) -> int:
        """
        Get the class number for the given article type.
        """
        return self.article_type_to_classnum[article_type]

    @classmethod
    def from_csv(
        cls, path: str,
        embedder_per_type: Dict[str, ArticleEmbedder],
        article_type_to_classnum: Dict[str, int]
    ) -> "HumanChatBotDataset":
        """
        Load the dataset from a CSV file.
        """
        data = pl.read_csv(path)
        return cls(data, embedder_per_type, article_type_to_classnum)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        row = self.data[idx]
        text = row["text"].item()
        article_type = row["type"].item()
        label = self.get_class_num(article_type)
        embedder = self.embedder_per_type[article_type]
        return embedder.embed(text), label
