import click
import os
from collections import defaultdict
from functools import wraps

from typing import Optional, Tuple
from mlproject.data_processing import RawHumanChatBotData, WikihowSubset, RunResult, AdaboostResult
from mlproject.datasets import HumanChatBotDataset, ImageByCrossMultiplicationDataset, ImageFolderDataset
from mlproject.embeddings import ArticleEmbedder, InferSentEmbedder
from mlproject.models import NNBaseModel, CNN2D

from typing import Optional

from torch.optim.optimizer import required

from mlproject.data_processing import RawHumanChatBotData, WikihowSubset
from mlproject.datasets import HumanChatBotDataset
from mlproject.embeddings import ArticleEmbedder, InferSentEmbedder,USEEmbedder, SBERTEmbedder
from mlproject.models import NNBaseModel
from pathlib import Path
from mlproject import constants as pc
import polars as pl
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder

import numpy as np

import torch

import re


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
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the raw data CSV file."
)
@click.option(
    "--seed",
    type=int,
    default=pc.R_SEED,
    help="The seed used for random subset."
)
@click.option(
    "--nrows",
    type=int,
    default=200000,
    help="The number of rows in subset."
)
@click.option(
    "--subset-path",
    type=click.Path(resolve_path=True),
    help="The path to save the subset csv"
)
def get_wikihow_subset(
        csv_path: str,
        seed: int,
        nrows: int,
        subset_path: Optional[str] = None
):
    wikihow_subset = WikihowSubset.return_subset(
        csv_path=csv_path, seed=seed, n_rows=nrows)
    if subset_path:
        print(f"Saving subset to {subset_path}")
        wikihow_subset.data.write_csv(subset_path)


@cli.command()
@click.option(
    "--subset-path",
    type=click.Path(resolve_path=True),
    help="The path to save the subset csv"
)
@click.option(
    "--queries-path",
    type=click.Path(resolve_path=True),
    help="The path to save the generated queries CSV."
)
def get_queries(
    subset_path: Optional[str] = None,
    queries_path: Optional[str] = None
):
    wikihow_subset = pl.read_csv(subset_path)
    data = WikihowSubset(wikihow_subset)
    # Generate queries
    queries_df = data.generate_queries()

    # Save the queries if a path is provided
    if queries_path:
        print(f"Saving queries to {queries_path}")
        queries_df.select("query").write_csv(queries_path)
    else:
        print("Generated queries:")
        print(queries_df.select("query"))


@cli.command()
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
def load_pretrained(
    name: str,
    vector_size: int,
    embedder_file: str
):
    train_data = None
    embedder_klass = ArticleEmbedder.registered_subclasses[name]
    embedder = embedder_klass.return_pretrained(
        training_data=train_data,
        vector_size=vector_size)
    if not embedder_file:
        embedder_file = Path(f"{name.lower()}.model").resolve().as_posix()
    print(f"Saving model: {name!r} to location: {embedder_file}")
    embedder.save(embedder_file)


@cli.command()
@click.option(
    "--name",
    type=click.Choice(AVAILABLE_EMBEDDERS),
    default="InferSentEmbedder",
    help="The name of the embedder to use."
)
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the raw data CSV file."
)
@click.option(
    "--embedded-base-path",
    type=str,
    required=True,
    help="The base path to save the embedded text."
)
@click.option(
    "--n-rows",
    type=int,
    required=False,
    help="The number of rows to read from the CSV file."
)
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the InferSent model file."
)
@click.option(
    "--w2v-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the word vectors file."
)
@click.option(
    "--vector-size",
    type=int,
    default=4096,
    help="The size of the embedding vectors."
)
def embed_data_with_infersent(
        name: str,
        csv_path: str,
        embedded_base_path: str,
        n_rows: int,
        model_path: str,
        w2v_path: str,
        vector_size: int
):
    """
    Load the pretrained InferSent embedder and use it to embed data from the CSV file.
    """

    # Load the InferSent embedder
    print("Loading the InferSent embedder...")
    embedder = InferSentEmbedder(
        model=None, vector_size=vector_size, model_path=model_path, w2v_path=w2v_path)
    embedder.load_model()

    # Load the dataset
    batch_size = 5000
    # Determine the total number of rows
    print("Calculating total rows in dataset...")
    total_df = pl.read_csv(csv_path, truncate_ragged_lines=True)
    total_rows = total_df.shape[0]
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)
    for batch_num in range(6, num_batches):
        start_row = batch_num * batch_size
        end_row = min(start_row + batch_size, total_rows)

        print(f"Processing batch {batch_num + 1}/{num_batches} (rows {start_row} to {end_row - 1})...")

        # Load the batch
        df = pl.read_csv(csv_path,
                         skip_rows=start_row,
                         n_rows=(end_row - start_row),
                         truncate_ragged_lines=True,
                         schema_overrides={"type": pl.Utf8, "text": pl.Utf8},
                         infer_schema_length=10000,
                         ignore_errors=True,
                         null_values=["Grow Plants from Seeds"])
        df.columns = ["type", "text"]
        # Create the dataset object
        dataset = RawHumanChatBotData(data=df)

        embeddings = HumanChatBotDataset.from_raw_data(dataset, embedder)  # embedder.embed_articles(dataset)
        # Embed the data
        print("Starting to embed data...")
        embeddings = HumanChatBotDataset.from_raw_data(dataset, embedder)

        # Create a DataFrame with embeddings and indices
        embeddings_df = embeddings.data

        # Save the batch embeddings to a file
        batch_embedded_path = f"{embedded_base_path}_batch_{batch_num + 1}.csv"
        print(f"Saving embeddings to {batch_embedded_path}...")
        embeddings_df.write_csv(batch_embedded_path)

    print("All batches processed and saved successfully.")

@cli.command()
@click.option("--name", required=False,default="USEEmbedder", help="Name of the embedder.")
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the raw data CSV file."
)
@click.option(
    "--embedded-base-path",
    type=str,
    required=True,
    help="The base path to save the embedded text."
)
@click.option(
    "--vector-size",
    type=int,
    default=512,
    help="The size of the embedding vectors."
)
@click.option(
    "--start-batch",
    required = False,
    type=int,
    default=1
)
def embed_data_with_use(name, csv_path, embedded_base_path, vector_size, start_batch):
    # Load the USE embedder
    print("Loading the USE embedder...")
    embedder = USEEmbedder(model=None, vector_size=vector_size)

    # Define batch size
    batch_size = 5000

    # Calculate total rows in dataset
    print("Calculating total rows in dataset...")
    total_df = pl.read_csv(csv_path, truncate_ragged_lines=True, ignore_errors=True)
    total_rows = total_df.shape[0]
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)

    for batch_num in range(start_batch - 1, num_batches):
        start_row = batch_num * batch_size
        end_row = min(start_row + batch_size, total_rows)

        print(f"Processing batch {batch_num + 1}/{num_batches} (rows {start_row} to {end_row - 1})...")

        # Load the batch
        df = pl.read_csv(
            csv_path,
        skip_rows=start_row,
        n_rows=(end_row - start_row),
        has_header=False,  # Treats all rows as data, not header
        truncate_ragged_lines=True,
        ignore_errors=True,
        null_values=["Grow Plants from Seeds"])

        df.columns = ["type", "text"]
        df = df.with_columns(
            pl.lit("llama").alias("type")  # Assigns "llama" to all rows in the "type" column
        )
        df = df.with_columns([
            pl.col("type").cast(pl.Utf8),
            pl.col("text").cast(pl.Utf8)
        ])
        df = df.drop_nulls(subset=["text"])
        df = df.filter(pl.col("text").str.strip_chars() != "")

        # Create the dataset object
        dataset = RawHumanChatBotData(data=df)

        # Embed the data
        print("Starting to embed data...")
        embeddings = HumanChatBotDataset.from_raw_data(dataset, embedder)

        # Save the batch embeddings to a file
        batch_embedded_path = f"{embedded_base_path}_batch_{batch_num + 1}.csv"
        print(f"Saving embeddings to {batch_embedded_path}...")
        embeddings.data.write_csv(batch_embedded_path)

    print("All batches processed and saved successfully.")

@cli.command()
@click.option("--name", required=False,default="SBERTEmbedder", help="Name of the embedder.")
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the raw data CSV file."
)
@click.option(
    "--embedded-base-path",
    type=str,
    required=True,
    help="The base path to save the embedded text."
)
@click.option(
    "--vector-size",
    type=int,
    default=384,
    help="The size of the embedding vectors."
)
@click.option(
    "--start-batch",
    required = False,
    type=int,
    default=1
)
def embed_data_with_sbert(name, csv_path, embedded_base_path, vector_size, start_batch):
    # Load the InferSent embedder
    print("Loading the SBERT embedder...")
    embedder = SBERTEmbedder(model=None, vector_size=vector_size)
    embedder.__post_init__()

    # Load the dataset
    batch_size = 5000
    # Determine the total number of rows
    print("Calculating total rows in dataset...")
    total_df = pl.read_csv(csv_path, truncate_ragged_lines=True)
    total_rows = total_df.shape[0]
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)
    for batch_num in range(start_batch-1, num_batches):
        start_row = batch_num * batch_size
        end_row = min(start_row + batch_size, total_rows)

        print(f"Processing batch {batch_num + 1}/{num_batches} (rows {start_row} to {end_row - 1})...")

        # Load the batch
        df = pl.read_csv(csv_path,
                         skip_rows=start_row,
                         n_rows=(end_row - start_row),
                         truncate_ragged_lines=True,
                         schema_overrides={"type": pl.Utf8, "text": pl.Utf8},
                         infer_schema_length=10000,
                         ignore_errors=True,
                         null_values=["\"Grow Plants from Seeds\n"])
        df.columns = ["type", "text"]
        # Create the dataset object
        dataset = RawHumanChatBotData(data=df)

        # Embed the data
        print("Starting to embed data...")
        embeddings = HumanChatBotDataset.from_raw_data(dataset, embedder)

        # Create a DataFrame with embeddings and indices
        embeddings_df = embeddings.data

        # Save the batch embeddings to a file
        batch_embedded_path = f"{embedded_base_path}_batch_{batch_num + 1}.csv"
        print(f"Saving embeddings to {batch_embedded_path}...")
        embeddings_df.write_csv(batch_embedded_path)

    print("All batches processed and saved successfully.")


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
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the InferSent model file"
)
@click.option(
    "--w2v-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the word vectors file"
)
def train_infersent_embedder(csv_path: str, n_rows: Optional[int], seed: int, train_percent: int, name: str, vector_size: int, embedder_file: str, model_path: str, w2v_path: str):
    train_data, _ = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=seed,
        n_rows=n_rows,
        train_percent=train_percent
    )
    embedder_klass = InferSentEmbedder
    embedder = embedder_klass.by_training_on_raw_data(
        training_data=train_data,
        vector_size=vector_size,
        model_path=model_path,
        w2v_path=w2v_path
    )
    if not embedder_file:
        embedder_file = Path(f"{name.lower()}.model").resolve().as_posix()
    print(f"Saving model: {name!r} to location: {embedder_file}")
    embedder.save(embedder_file)


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
    print("Loading the datasets...")
    train_data, _ = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=seed,
        n_rows=n_rows,
        train_percent=train_percent
    )
    print("Getting the embedder class and training the embedder...")
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
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    required=True,
    help="The path to the raw data CSV file."
)
@click.option(
    "--subset-path",
    type=click.Path(resolve_path=True),
    required=True,
    help="Where to save the subset"
)
def get_and_store_subset(csv_path, subset_path):
    print("loading raw data")
    df = pl.read_csv(csv_path)
    print("getting human subset")
    human_df = df.filter(pl.col("type") == "human").sample(n=50000)
    print("getting gpt subset")
    gpt_df = df.filter(pl.col("type") == "gpt").sample(n=50000)
    print("combining subsets")
    sampled_df = pl.concat([human_df, gpt_df])
    print("dropping all columns besides type and text ")
    result_df = sampled_df.select(["type", "text"])
    print("saving to subset_path")
    result_df.write_csv(subset_path)
    print("saved")

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
        embedder=embedder
    )
    train_file_path = Path(train_file_path).resolve().as_posix()
    test_file_path = Path(test_file_path).resolve().as_posix()
    train_dataset.save(train_file_path)
    test_dataset.save(test_file_path)
    print(f"Saved embedded training dataset at: {train_file_path!r}")
    print(f"Saved embedded test dataset at: {test_file_path!r}")


def get_information_from_embedded_path(some_path: str) -> Tuple[str, str]:
    """Give the path to an embedded dataset, return
    the name of the original dataset, the name of the embedder
    and whether training or testing.

    Args:
        some_path (str): some path to an embedded dataset

    Returns:
        Tuple[str, str, bool]: orig_ds_name, embedder_name
    """
    base_name = os.path.basename(some_path).split(".")[0]
    pattern = r"(?P<embedder>\w+)_(?P<ds_name>\w+).*"
    match = re.match(pattern, base_name)
    if not match:
        raise ValueError(
            f"Could not extract the dataset name and embedder name from {some_path!r}")
    embedder_name = match.group("embedder")
    orig_ds_name = match.group("ds_name")
    return orig_ds_name, embedder_name


def get_embedded_training_and_testing_ds(ds: str, test_ds: Optional[str], seed: Optional[int] = None) -> Tuple[HumanChatBotDataset, HumanChatBotDataset]:
    if ds and test_ds:
        print(
            f"Both ds and test_ds provided. Assuming {ds!r} is the training dataset and returning both datasets.")
        train_dataset = HumanChatBotDataset.load(ds)
        test_dataset = HumanChatBotDataset.load(test_ds)
    else:
        print(
            f"Only ds provided. Splitting {ds!r} into training and test datasets. Using seed = {seed}, train_percent = {pc.TRAIN_PERCENT}")
        # Load the csv file
        df = pl.read_csv(ds)
        # shuffle and split
        shuffled_data = df.sample(fraction=1, shuffle=True, seed=seed)
        train_size = int(len(shuffled_data) * pc.TRAIN_PERCENT)
        train_data = shuffled_data.head(train_size)
        test_data = shuffled_data.tail(-train_size)
        train_dataset = HumanChatBotDataset(train_data)
        test_dataset = HumanChatBotDataset(test_data)
    return train_dataset, test_dataset

def get_image_training_and_testing_ds(
    ds: str,
    test_ds: Optional[str]
) -> Tuple[ImageFolderDataset, ImageFolderDataset]:
    if ds and test_ds:
        return ImageFolderDataset(ds), ImageFolderDataset(test_ds)
    else:
        # get the one level down subfolders
        subfolders = [x for x in os.listdir(ds) if os.path.isdir(os.path.join(ds, x))]
        train_dirs = [folder for folder in subfolders if "train" in folder]
        test_dirs = [folder for folder in subfolders if "test" in folder]
        if not train_dirs or not test_dirs:
            raise ValueError(
                f"Could not find train and test subfolders in {ds!r}")
        full_train_path = os.path.join(ds, train_dirs[0])
        full_test_path = os.path.join(ds, test_dirs[0])
        return ImageFolderDataset(full_train_path), ImageFolderDataset(full_test_path)


@cli.command()
@click.option(
    "--ds",
    type=click.Path(exists=True),
    help="The path to the dataset to use for training.",
    required=True
)
@click.option(
    "--test-ds",
    type=click.Path(exists=True),
    default=None,
    help="The path to the test dataset. If used, it will be assumed that --ds is the training dataset."
)
@click.option(
    "--seed",
    type=int,
    default=pc.R_SEED,
    help="The seed to use for splitting ds into training and test datasets. If --test-ds is provided, this will be ignored."
)
@click.option(
    "--model-name",
    type=click.Choice(AVAILABLE_NN_MODELS),
    required=True,
    help="The name of the model to use for training."
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
@click.option(
    "--save-dir",
    type=click.Path(resolve_path=True, file_okay=False, exists=True),
    help="The directory to save the results.",
    default="."
)
def train_nn_model(
    ds: str,
    test_ds: str,
    seed: int,
    model_name: str,
    epochs: int,
    learning_rate: float,
    save_dir: str
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)
    # BEGIN: Validation of filenames
    ds_name, embedder_name = get_information_from_embedded_path(ds)
    # END: Validation of filenames
    # base identifier
    identifier = f"{ds_name}_{embedder_name}_{model_name}"
    run_f_name = f"{identifier}.json".lower()
    full_run_path = Path(save_dir).joinpath(run_f_name)
    model_klass = NNBaseModel.registry[model_name]
    model: NNBaseModel = None
    if model_name in ("CNN2D", "CNNLstm"):
        train_dataset, test_dataset = get_image_training_and_testing_ds(
            ds, test_ds)
        image_height = train_dataset.image_height
        image_width = train_dataset.image_width
        n_classes = train_dataset.number_of_classes
        model = model_klass(image_height=image_height,
                            image_width=image_width,
                            n_classes=n_classes,
                            learning_rate=learning_rate
                            )
    else:
        train_dataset, test_dataset = get_embedded_training_and_testing_ds(
            ds=ds, test_ds=test_ds, seed=seed)
        embedding_size = train_dataset.embedding_size
        n_classes = train_dataset.number_of_classes
        model = model_klass(input_dim=embedding_size,
                            output_dim=n_classes, learning_rate=learning_rate)

    # send model to GPU
    print(f"Sending model to device: {device}")
    model.to(device)
    result = model.run_training(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=epochs
    )
    run_result = RunResult(
        original_dataset_name=ds_name,
        embedder_name=embedder_name,
        experiment_result=result
    )
    print(f"Saving experiment results at: {full_run_path!r}")
    run_result.save(full_run_path)

    # save the model and its optimizer
    model_dir = os.path.join(save_dir, f"{identifier}")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{identifier}.model")
    optimizer_path = os.path.join(model_dir, f"{identifier}.optimizer")
    print(f"Saving model at: {model_path!r}")
    torch.save(model.state_dict(), model_path)
    if getattr(model, "optimizer", None):
        print(f"Saving model's optimizer at: {optimizer_path!r}")
        torch.save(model.optimizer.state_dict(), optimizer_path)
    else:
        print("No optimizer found. Skipping saving optimizer.")


@cli.command()
@click.option(
    "--ds",
    type=click.Path(exists=True, dir_okay=False),
    required=True
)
@click.option(
    "--test-ds",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="The path to the test dataset. If used, it will be assumed that --ds is the training dataset."
)
@click.option(
    "--seed",
    type=int,
    default=pc.R_SEED,
    help="The seed to use for splitting ds into training and test datasets. If --test-ds is provided, this will be ignored."
)
@click.option(
    "--save-dir",
    type=click.Path(),
    help="The directory to save the results.",
    default="."
)
def convert_into_image_dataset(
    ds: str,
    test_ds: str,
    seed: int,
    save_dir: str
):
    ds_name, embedder_name = get_information_from_embedded_path(ds)
    train_dataset, test_dataset = get_embedded_training_and_testing_ds(
        ds=ds, test_ds=test_ds, seed=seed)

    subfolder = os.path.join(save_dir, f"{embedder_name}_{ds_name}")
    os.makedirs(subfolder, exist_ok=True)
    train_path = os.path.join(
        subfolder, f"{embedder_name}_{ds_name}_train")
    test_path = os.path.join(subfolder, f"{embedder_name}_{ds_name}_test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    ImageByCrossMultiplicationDataset.convert_and_save(
        train_dataset, train_path
    )
    ImageByCrossMultiplicationDataset.convert_and_save(
        test_dataset, test_path
    )


@cli.command()
@click.option(
    "--ds",
    type=click.Path(exists=True, dir_okay=False),
    required=True


)
@click.option(
    "--test-ds",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="The path to the test dataset. If used, it will be assumed that --ds is the training dataset."
)
@click.option(
    "--seed",
    type=int,
    default=pc.R_SEED,
    help="The seed to use for splitting ds into training and test datasets. If --test-ds is provided, this will be ignored."
)
@click.option(
    "--save-dir",
    type=click.Path(resolve_path=True, file_okay=False, exists=True),
    help="The directory to save the results.",
    default="."
)
def adaboost(ds: str, test_ds: str, seed: int, save_dir: str):
    print("Loading datasets...")
    train_dataset, test_dataset = get_embedded_training_and_testing_ds(
        ds, test_ds, seed)
    ds_name, embedder_name = get_information_from_embedded_path(ds)

    results_path = os.path.join(
        save_dir, f"{ds_name}_{embedder_name}_adaboost.json")
    print("Adapting data for scikit-learn...")
    train_X = train_dataset.get_samples_as_X()
    _train_Y = train_dataset.get_sample_labels()
    test_X = test_dataset.get_samples_as_X()
    _test_Y = test_dataset.get_sample_labels()
    le = LabelEncoder()
    n_training = len(_train_Y)
    all_y = np.concatenate((_train_Y, _test_Y))
    all_y = le.fit_transform(all_y)
    train_Y = all_y[:n_training]
    test_Y = all_y[n_training:]
    print("Training the model...")
    model = AdaBoostClassifier(algorithm="SAMME")
    model.fit(train_X, train_Y)
    accuracy = model.score(test_X, test_Y)
    predictions = model.predict(test_X)
    makeup = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(test_Y, predictions):
        i = int(true)
        j = int(pred)
        makeup[i][j] += 1
    adaboost_result = AdaboostResult(
        original_ds_name=ds_name,
        embedder_name=embedder_name,
        accuracy=accuracy,
        classification_makeup=makeup
    )
    print(f"Accuracy: {accuracy*100:.4f}%")
    print("Saving results...")
    adaboost_result.save(results_path)


def main():
    cli()


if __name__ == "__main__":
    main()
