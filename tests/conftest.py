import pytest

from mlproject.data_processing import RawHumanChatBotData

from mlproject.datasets import HumanChatBotDataset, TwoDImagesDataset

from mlproject.embeddings import CBOWWord2Vec


@pytest.fixture(scope="session")
def skinny_dataset_path():
    yield "/home/cberruz/SJSU/Fall2024/CS271/project_data/skinny_dataset.csv"


@pytest.fixture(scope="session")
def n_rows():
    yield 1000


@pytest.fixture(scope="session")
def human_chat_bot_data(skinny_dataset_path, n_rows):
    yield RawHumanChatBotData.train_test_split(skinny_dataset_path, n_rows=n_rows)


@pytest.fixture(scope="session")
def training_data(human_chat_bot_data):
    yield human_chat_bot_data[0]


@pytest.fixture(scope="session")
def testing_data(human_chat_bot_data):
    yield human_chat_bot_data[1]


@pytest.fixture(scope="session")
def embedder(training_data):
    yield CBOWWord2Vec.by_training_on_raw_data(
        training_data=training_data,
        vector_size=100
    )


@pytest.fixture(scope="session")
def training_dataset(training_data, embedder):
    yield HumanChatBotDataset.from_raw_data(training_data, embedder)


@pytest.fixture(scope="session")
def testing_dataset(testing_data, embedder):
    yield HumanChatBotDataset.from_raw_data(testing_data, embedder)


@pytest.fixture(scope="session")
def training_dataset_img2d(training_dataset):
    yield TwoDImagesDataset(training_dataset)


@pytest.fixture(scope="session")
def testing_dataset_img2d(testing_dataset):
    yield TwoDImagesDataset(testing_dataset)
