"""
There are many Natural Language Processing (NLP) embedding techniques.

These techniques work on different aspects of text: words, sentences, paragraphs.

However, our research involves articles. As a result, we have
to define article embeddings that we can use to train our models.

This module contains all classes and functions related to article embeddings.
"""
from abc import ABC, abstractmethod
import torch

from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

import numpy as np

from dataclasses import dataclass, field


class ArticleEmbedder(ABC):
    """
    A class to adapt the embeddings to the model.
    """

    @abstractmethod
    def embed(self, article: str) -> torch.Tensor:
        """Embed an article into a single fixed-length vector.

        Args:
            article (str): the article text

        Returns:
            torch.Tensor: a numerical representation of the article
        """
        pass


@dataclass
class Word2VecEmbedder(ArticleEmbedder):
    """
    A class to adapt the text to the Word2Vec model.
    """
    model: Word2Vec = field(repr=False)
    vector_size: int = field(init=False)

    def __post_init__(self):
        self.vector_size = self.model.vector_size

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
        article = article.replace("\n", " ")
        n_words = 0
        for i in sent_tokenize(article):
            for j in word_tokenize(i):
                k = j.lower()
                if k not in self.model.wv:
                    word_vector = np.zeros(self.vector_size, dtype=np.float32)
                else:
                    word_vector = self.model.wv[k]
                average_vector += word_vector
                n_words += 1
        average_vector = average_vector / n_words
        return torch.from_numpy(average_vector)
