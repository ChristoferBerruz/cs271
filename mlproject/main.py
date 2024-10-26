import click

from functools import wraps

from typing import Optional
from mlproject.data_processing import RawHumanChatBotData

from mlproject.datasets import HumanChatBotDataset

from mlproject.embeddings import ArticleEmbedder

from mlproject.models import NNBaseModel
from pathlib import Path

from mlproject import constants as pc


@click.group()
def cli():
    pass


def read_raw_data_common_options(func):
    @click.option(
        "--csv-path",
        type=click.Path(exists=True),
        required=True,
        help="The path to the raw data CSV file."
    )
    @click.option(
        "--n-rows",
        type=int,
        default=None,
        help="The number of rows to read from the CSV file."
    )
    @click.option(
        "--seed",
        type=int,
        default=pc.R_SEED,
        help="The random seed used for shuffling."
    )
    @click.option(
        "--train-percent",
        type=float,
        default=pc.TRAIN_PERCENT,
        help="The percentage of the data to use for training."
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


AVAILABLE_EMBEDDERS = list(ArticleEmbedder.registered_subclasses.keys())
AVAILABLE_NN_MODELS = list(NNBaseModel.registry.keys())


@cli.command()
@read_raw_data_common_options
@click.option(
    "--name",
    type=click.Choice(AVAILABLE_EMBEDDERS),
    default=AVAILABLE_EMBEDDERS[0],
)
@click.option(
    "--vector-size",
    type=int,
    default=100,
    help="The size of the embedding vectors."
)
@click.option(
    "--embedder-file",
    type=click.Path(resolve_path=True),
    help="The path to save the embedder file"
)
def train_embedder(
    csv_path: str,
    n_rows: Optional[int],
    seed: int,
    train_percent: float,
    name: str,
    vector_size: int,
    embedder_file: str
):
    train_data, _ = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=seed,
        n_rows=n_rows,
        train_percent=train_percent
    )
    embedder_klass = ArticleEmbedder.registered_subclasses[name]
    embedder = embedder_klass.by_training_on_raw_data(
        training_data=train_data,
        vector_size=vector_size
    )
    if not embedder_file:
        embedder_file = Path(f"{name.lower()}.model").resolve().as_posix()
    print(f"Saving model: {name!r} to location: {embedder_file}")
    embedder.save(embedder_file)


@cli.command()
@read_raw_data_common_options
@click.option(
    "--embedder-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True
)
@click.option(
    "--train-file-path",
    type=click.Path(dir_okay=False),
    required=True
)
@click.option(
    "--test-file-path",
    type=click.Path(dir_okay=False),
    required=True
)
def embed_raw_data(
    csv_path: str,
    n_rows: Optional[int],
    seed: int,
    train_percent: float,
    embedder_file: str,
    train_file_path: str,
    test_file_path: str
):
    train_data, test_data = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=seed,
        n_rows=n_rows,
        train_percent=train_percent
    )

    embedder = ArticleEmbedder.load(embedder_file)
    train_dataset, test_dataset = HumanChatBotDataset.from_train_test_raw_data(
        train_data=train_data,
        test_data=test_data,
        embedder=embedder,
        article_type_to_classnum=pc.ARTICLE_TYPES_TO_CLASSNUM
    )
    train_file_path = Path(train_file_path).resolve().as_posix()
    test_file_path = Path(test_file_path).resolve().as_posix()
    train_dataset.save(train_file_path)
    test_dataset.save(test_file_path)
    print(f"Saved embedded training dataset at: {train_file_path!r}")
    print(f"Saved embedded test dataset at: {test_file_path!r}")


@cli.command()
@click.option(
    "--training-dataset-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True
)
@click.option(
    "--testing-dataset-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True
)
@click.option(
    "--model-name",
    type=click.Choice(AVAILABLE_NN_MODELS),
    required=True
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="The number of epochs to train the model."
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    help="The learning rate for the optimizer."
)
def train_nn_model(
    training_dataset_path: str,
    testing_dataset_path: str,
    model_name: str,
    epochs: int,
    learning_rate: float
):
    train_dataset = HumanChatBotDataset.load(training_dataset_path)
    test_dataset = HumanChatBotDataset.load(testing_dataset_path)
    model_klass = NNBaseModel.registry[model_name]
    model: NNBaseModel = model_klass(input_dim=train_dataset.embedding_size,
                                     output_dim=train_dataset.number_of_classes)
    result = model.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        learning_rate=learning_rate
    )


def main():
    cli()


if __name__ == "__main__":
    main()
