import click

from typing import Optional
from data_processing import RawHumanChatBotData, HumanChatBotDataset, ReusableGenerator

from text_helpers import sentence_word_tokenizer

from gensim.models import Word2Vec
from pathlib import Path

from embeddings import Word2VecEmbedder
from models import LogisticRegression, SimpleMLP

import torch

from torch.utils.data import DataLoader


import constants as pc


@click.group()
def cli():
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--n-rows", type=int, default=None, help="Number of rows to read from the CSV file.")
@click.option("--train-percent", type=float, default=pc.TRAIN_PERCENT, help="Percentage of data to use for training.")
@click.option("--seed", type=int, default=pc.R_SEED, help="Random seed.")
@click.option("--vector-size", type=int, default=100, help="Size of the word vectors.")
@click.option("--min_count", type=int, default=1, help="Minimum frequency of words to include in the vocabulary.")
@click.option("--window", type=int, default=5, help="Size of the context window.")
@click.option("--save-file", type=click.Path(dir_okay=False, resolve_path=True), default="word2vec.model", help="Path to save the Word2Vec model.")
@click.option("--override-if-exists", is_flag=True, default=False, help="Override the file if it already exists.")
def train_word2vec(
    csv_path: str,
    n_rows: Optional[int],
    train_percent: float,
    seed: int,
    vector_size: int,
    min_count: int,
    window: int,
    save_file: str,
    override_if_exists: bool
):
    """Train a Word2Vec model on the entire dataset using Continuous Bag of Words (CBOW) algorithm.
    """
    if Path(save_file).exists() and not override_if_exists:
        raise click.exceptions.UsageError(
            f"File {save_file!r} already exists. Use --override-if-exists to override."
        )
    print(f"Training Word2Vec model on {csv_path}...")
    train_data, _ = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=seed,
        n_rows=n_rows,
        train_percent=train_percent
    )
    print(f"Training dataset is of size: {train_data.total_articles}")

    def sentence_generator():
        for article in train_data.get_articles():
            for sentence in sentence_word_tokenizer(article):
                yield sentence

    data_gen = ReusableGenerator(sentence_generator)
    model = Word2Vec(data_gen, min_count=min_count,
                     vector_size=vector_size, window=window)

    model.save(save_file)
    print(f"Word2Vec model saved at: {save_file!r}")


@cli.command()
@click.argument("csv_path", type=str)
@click.option(
    "--architecture",
    type=click.Choice(["logistic_regression", "simple_mlp"]),
    default="logistic_regression",
    help="The architecture to use for training."
)
@click.option(
    "--word2vec-model-path",
    type=click.Path(dir_okay=False, exists=True, resolve_path=True),
    default="word2vec.model",
    help="Path to the Word2Vec model."
)
@click.option("--learning-rate", type=float, default=0.001, help="Learning rate for the optimizer.")
@click.option("--epochs", type=int, default=10, help="Number of epochs to train the model.")
def train_architecture_on_word2vec(
        csv_path: str,
        architecture: str,
        word2vec_model_path: str,
        learning_rate: float = 0.001,
        epochs: int = 10,
):
    word2vec_model = Word2Vec.load(word2vec_model_path)
    word2vec_embedder = Word2VecEmbedder(word2vec_model)
    if architecture == "logistic_regression":
        model = LogisticRegression(
            input_dim=word2vec_embedder.vector_size, output_dim=2)
    elif architecture == "simple_mlp":
        model = SimpleMLP(
            input_dim=word2vec_embedder.vector_size, output_dim=2)

    train_data, test_data = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=pc.R_SEED,
        train_percent=pc.TRAIN_PERCENT
    )

    train_dataset = HumanChatBotDataset.from_raw_data(
        train_data, embedder=word2vec_embedder, article_type_to_classnum=pc.ARTICLE_TYPES_TO_CLASSNUM
    )
    test_dataset = HumanChatBotDataset.from_raw_data(
        test_data, embedder=word2vec_embedder, article_type_to_classnum=pc.ARTICLE_TYPES_TO_CLASSNUM
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.train(
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_klass=torch.optim.Adam,
        criterion_klass=torch.nn.CrossEntropyLoss,
        epochs=epochs,
        learning_rate=learning_rate
    )


def main():
    cli()


if __name__ == "__main__":
    main()
