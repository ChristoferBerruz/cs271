"""
There are many Natural Language Processing (NLP) embedding techniques.

These techniques work on different aspects of text: words, sentences, paragraphs.

However, our research involves articles. As a result, we have
to define article embeddings that we can use to train our models.

This module contains all classes and functions related to article embeddings.
"""
import urllib.request
from abc import ABC, abstractmethod
import torch
from gensim.models import Word2Vec
import numpy as np
from dataclasses import dataclass, field
from mlproject.text_helpers import sentence_word_tokenizer
from mlproject.data_processing import RawHumanChatBotData, ReusableGenerator
from typing import Dict, ClassVar
import pickle
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from InferSent.models import InferSent
import ssl

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

    @classmethod
    def return_pretrained(
            cls,
            training_data: None,
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

    @classmethod
    def return_pretrained(
            cls,
            training_data: None,
            vector_size: int = 100
    ) -> "CBOWWord2Vec":
        raise NotImplementedError(
            "Pre-trained Word2Vec models are not supported.")


@dataclass
class InferSentEmbedder(ArticleEmbedder):
    """
    A class to adapt the text to the Infersent model
    """
    model: InferSent = field(repr=False)
    vector_size: int
    model_path: str = field(
        default="/content/drive/MyDrive/Fall2024/CS271-MachineLearning/MLProject/InferSent/infersent2.pkl")
    w2v_path: str = field(
        default="/content/drive/MyDrive/Fall2024/CS271-MachineLearning/MLProject/InferSent/crawl-300d-2M.vec")

    def __post_init__(self):
        if not hasattr(self, 'model') or self.model is None:
            self.load_model()

    def load_model(self):
        print("Loading InferSent model...")
        model_version = 2
        params_model = {
            'bsize': 64,
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'pool_type': 'max',
            'dpout_model': 0.0,
            'version': model_version
        }
        self.model = InferSent(params_model)
        self.model.load_state_dict(torch.load(
            self.model_path, weights_only=True))
        self.model.set_w2v_path(self.w2v_path)
        self.model.build_vocab_k_words(K=100000)  # Set up vocabulary
        self.model = self.model.eval()

    def embed(self, article: str) -> torch.Tensor:
        """
        Embed the article into a single fixed-length vector
        by calculating the average of the InferSent model embeddings for all sentences.

        Args:
            article (str): The article text.

        Returns:
            torch.Tensor: A numerical representation of the article.
        """
        if not hasattr(self, 'model') or self.model is None:
            self.load_model()
        sentence_embeddings = []

        def sentence_generator():
            for sentence in sentence_word_tokenizer(article):
                yield sentence

        data_gen = ReusableGenerator(sentence_generator)
        sentences = []
        for sentences in data_gen:
            # Filter out empty or non-string sentences
            sentences = [s for s in sentences if isinstance(
                s, str) and s.strip()]

        for sentence in sentences:
            if sentence.strip():  # Ensure the sentence is not empty
                embeddings = self.model.encode([sentence], tokenize=True)
                sentence_embeddings.append(embeddings[0])

        if not sentence_embeddings:
            # Return a zero vector if no valid sentences are found
            embedding_dim = self.model.encoder_dim
            return torch.zeros(embedding_dim)

        average_embedding = np.mean(sentence_embeddings, axis=0)
        return torch.from_numpy(average_embedding)

    @classmethod
    def by_training_on_raw_data(
            cls,
            training_data: "RawHumanChatBotData",
            vector_size: int = 512,  # Default vector size is 4096 for Infersent
            model_path: str = "path/to/infersent.pkl",
            w2v_path: str = "path/to/word_vectors.txt"
    ) -> "InferSentEmbedder":

        params_model = {
            'bsize': 64, 'word_emb_dim': 200,
            'enc_lstm_dim': 256, 'pool_type': 'mean', #switched to 'mean' from 'max
            'dpout_model': 0.0, 'version': 2
        }
        infersent = InferSent(params_model)
        infersent.load_state_dict(torch.load(model_path), strict=False)
        infersent.set_w2v_path(w2v_path)

        # Build the vocabulary with the training data
        sentences = [sentence for article in training_data.get_articles()
                     for sentence in sentence_word_tokenizer(article)]
        infersent.build_vocab(sentences, tokenize=True)

        # Return the initialized InferSentEmbedder
        return cls(infersent, vector_size, model_path, w2v_path)


@dataclass
class USEEmbedder(ArticleEmbedder):
    """
        A class to generate embeddings using the Universal Sentence Encoder (USE).
        """
    model: hub.KerasLayer = field(repr=False)
    vector_size: int

    def __post_init__(self):
        # Load the USE model from TensorFlow Hub
        if self.model is None:
            print("Loading Universal Sentence Encoder...")
            # Create an unverified SSL context
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

            print("USE model loaded.")



    def embed(self, article: str) -> torch.Tensor:
        """
        Embed the article using the pre-trained USE model.

        Args:
            article (str): The article text.

        Returns:
            torch.Tensor: A numerical representation of the article.
        """
        # Use TensorFlow to generate embeddings
        if not hasattr(self, 'model') or self.model is None:
            self.__post_init__()

        # Generate embeddings for the entire article as a single text input
        embeddings = self.model([article])
        embedding_vector = embeddings.numpy().flatten()

        return torch.from_numpy(embedding_vector)

    @classmethod
    def embed_articles(cls, article: str) -> torch.Tensor:
        """
        Embed the article into a single fixed-length vector
        by calculating the average of the USE embeddings for all sentences.

        Args:
            article (str): the article text

        Returns:
            torch.Tensor: a numerical representation of the article
        """
        # Tokenize the article into sentences
        sentences = [" ".join(sentence)
                     for sentence in sentence_word_tokenizer(article)]
        # Generate USE embeddings for all sentences
        sentence_embeddings = cls.model(sentences).numpy()
        # Calculate the average embedding across all sentence embeddings
        average_embedding = np.mean(sentence_embeddings, axis=0)
        return torch.from_numpy(average_embedding)

    @classmethod
    def return_pretrained(
            cls,
            training_data: None,
            vector_size: int = 512
    ) -> "USEEmbedder":
        # Instantiate the class without requiring training data
        return cls(model=None, vector_size=vector_size)

    @classmethod
    def by_training_on_raw_data(
            cls,
            training_data: "RawHumanChatBotData",
            vector_size: int = 512  # Default embedding size for USE
    ) -> "USEEmbedder":
        # Initialize the USEEmbedder without any training (pre-trained model)
        print("Loading Universal Sentence Encoder...")
        model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

        return cls(model=model, vector_size=vector_size)


@dataclass
class SBERTEmbedder(ArticleEmbedder):
    """
    A class to generate embeddings using Sentence-BERT (SBERT).
    """
    model: SentenceTransformer = field(repr=False)
    vector_size: int

    def __post_init__(self):
        # Load the SBERT model if not already loaded
        print("Loading SBERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SBERT model loaded.")

    def embed(self, article: str) -> torch.Tensor:
        """
        Embed the article into a single fixed-length vector
        by calculating the average of the SBERT embeddings for all sentences.

        Args:
            article (str): The article text.

        Returns:
            torch.Tensor: A numerical representation of the article.
        """
        print("embedding article")
        # Tokenize the article into sentences and flatten to a list of strings
        sentences = [sentence for sentence_list in sentence_word_tokenizer(article) for sentence in sentence_list if
                     sentence.strip()]

        # Generate SBERT embeddings for each sentence
        sentence_embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # Calculate the average embedding across all sentence embeddings
        average_embedding = np.mean(sentence_embeddings, axis=0)

        return torch.from_numpy(average_embedding)

    @classmethod
    def embed_articles(cls, article: str) -> torch.Tensor:
        """
        Embed the article into a single fixed-length vector
        by calculating the average of the SBERT embeddings for all sentences.

        Args:
            article (str): the article text

        Returns:
            torch.Tensor: a numerical representation of the article
        """
        # Tokenize the article into sentences
        sentences = [" ".join(sentence)
                     for sentence in sentence_word_tokenizer(article)]
        # Generate SBERT embeddings for all sentences
        sentence_embeddings = cls.model.encode(
            sentences, convert_to_numpy=True)
        # Calculate the average embedding across all sentence embeddings
        average_embedding = np.mean(sentence_embeddings, axis=0)
        return torch.from_numpy(average_embedding)

    @classmethod
    def return_pretrained(
            cls,
            training_data: None,
            vector_size: int = 384
    ) -> "SBERTEmbedder":
        return cls(model=SentenceTransformer('all-MiniLM-L6-v2'), vector_size=vector_size)

    @classmethod
    def by_training_on_raw_data(
            cls,
            training_data: "RawHumanChatBotData",
            vector_size: int = 384  # Default embedding size for SBERT 'all-MiniLM-L6-v2'
    ) -> "SBERTEmbedder":
        # Initialize the SBERTEmbedder without any additional training (pre-trained model)
        return cls(model=SentenceTransformer('all-MiniLM-L6-v2'), vector_size=vector_size)
