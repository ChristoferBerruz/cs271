"""
There are many Natural Language Processing (NLP) embedding techniques.

These techniques work on different aspects of text: words, sentences, paragraphs.

However, our research involves articles. As a result, we have
to define article embeddings that we can use to train our models.

This module contains all classes and functions related to article embeddings.
"""
import os
from abc import ABC, abstractmethod
import torch

from gensim.models import Word2Vec

import numpy as np

from dataclasses import dataclass, field

from text_helpers import sentence_word_tokenizer

from data_processing import RawHumanChatBotData, ReusableGenerator

from typing import Dict, ClassVar

import pickle


class ArticleEmbedder(ABC):
    """
    A class to adapt the embeddings to the model.
    """
    registered_subclasses: ClassVar[Dict[str, "ArticleEmbedder"]] = {}

    @abstractmethod
    def embed(self, article: str) -> torch.Tensor:
        """Embed an article into a single fixed-length vector.

        Args:
            article (str): the article text

        Returns:
            torch.Tensor: a numerical representation of the article
        """
        pass

    def __init_subclass__(cls) -> None:
        registry = ArticleEmbedder.registered_subclasses
        if cls.__name__ not in registry:
            ArticleEmbedder.registered_subclasses[cls.__name__] = cls
        return super().__init_subclass__()

    def save(self, save_path: str):
        """Save the embeddings to a path
        """
        print(f"Pickling {save_path!r}")
        with open(save_path, mode="wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_path: str) -> "ArticleEmbedder":
        print(f"Loading {cls.__name__} from {load_path!r}")
        with open(load_path, mode="rb") as f:
            return pickle.load(f)

    @classmethod
    @abstractmethod
    def by_training_on_raw_data(
        cls,
        training_data: RawHumanChatBotData,
        vector_size: int
    ) -> "ArticleEmbedder":
        pass


def train_word2vec_on_raw_data(
    dataset: RawHumanChatBotData,
    vector_size: int = 100,
    min_count: int = 1,
    window=5
) -> Word2Vec:
    """
    Train a Word2Vec model on the entire dataset.
    """
    print(
        f"Training Word2Vec model on {dataset.total_articles} articles...")

    def sentence_generator():
        for article in dataset.get_articles():
            for sentence in sentence_word_tokenizer(article):
                yield sentence

    data_gen = ReusableGenerator(sentence_generator)
    model = Word2Vec(data_gen, min_count=min_count,
                     vector_size=vector_size, window=window)
    return model


@dataclass
class CBOWWord2Vec(ArticleEmbedder):
    """
    A class to adapt the text to the Word2Vec model.
    """
    model: Word2Vec = field(repr=False)
    vector_size: int

    def embed(self, article: str) -> torch.Tensor:
        """Embed the article into a single fixed-length vector
        by calculating the average of the word embeddings in the entire
        article using Word2Vec.

        Args:
            article (str): the article text

        Returns:
            torch.Tensor: a numerical representation of the article
        """
        average_vector = np.zeros(self.vector_size, dtype=np.float32)
        n_words = 0
        for sentence in sentence_word_tokenizer(article):
            n_words += len(sentence)
            for word in sentence:
                if word in self.model.wv:
                    average_vector += self.model.wv[word]
        average_vector = average_vector / n_words
        return torch.from_numpy(average_vector)

    @classmethod
    def by_training_on_raw_data(
        cls,
        training_data: "RawHumanChatBotData",
        vector_size: int = 100
    ) -> "CBOWWord2Vec":
        word2vec = train_word2vec_on_raw_data(
            training_data,
            vector_size=vector_size,
            min_count=1,
            window=5
        )
        return cls(word2vec, vector_size)
