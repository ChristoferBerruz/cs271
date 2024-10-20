from models import LogisticRegression
from embeddings import Word2VecEmbedder
from data_processing import RawHumanChatBotData, HumanChatBotDataset, ReusableGenerator
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

import constants as pc


import argparse
from typing import Optional, Iterable
import torch
from torch.utils.data import DataLoader, random_split
import nltk


def train_word2vec_on_raw_data(
    dataset: RawHumanChatBotData,
    vector_size: int = 100,
    min_count: int = 1,
    window=5
) -> Word2Vec:
    """
    Train a Word2Vec model on the entire dataset.
    """
    print(f"Training Word2Vec model on {dataset.total_articles} articles...")

    def get_articles_wrapper():
        return dataset.get_articles()

    articles = ReusableGenerator(get_articles_wrapper)

    def sentence_generator():
        for article in articles:
            article = article.replace("\n", " ")
            for i in sent_tokenize(article):
                sentence = []
                for j in word_tokenize(i):
                    sentence.append(j.lower())
                yield sentence

    data_gen = ReusableGenerator(sentence_generator)
    model = Word2Vec(data_gen, min_count=min_count,
                     vector_size=vector_size, window=window)
    return model


def main(csv_path: str, n_rows: Optional[int] = None):
    """A simple example of how to use Word2Vec with a Logistic Regression model
    for classification.
    """
    print(
        f"Word2vec training using data from {csv_path} and n_articles {n_rows}")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    vector_size = 100
    print("Loading the HumanChatBotDataset...")

    train_data, test_data = RawHumanChatBotData.train_test_split(
        path=csv_path,
        seed=pc.R_SEED,
        n_rows=n_rows,
        train_percent=pc.TRAIN_PERCENT
    )

    word2vec = train_word2vec_on_raw_data(
        train_data,
        vector_size=vector_size,
        min_count=1,
        window=5
    )
    word2vec.save("word2vec.model")
    embedder = Word2VecEmbedder(word2vec)

    article_to_class_num = {
        "gpt": 0,
        "human": 1
    }

    train_dataset = HumanChatBotDataset.from_raw_data(
        train_data, embedder=embedder, article_type_to_classnum=article_to_class_num
    )
    test_dataset = HumanChatBotDataset.from_raw_data(
        test_data, embedder=embedder, article_type_to_classnum=article_to_class_num
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # create the model
    model = LogisticRegression(vector_size, len(article_to_class_num))
    model.train(
        train_loader,
        test_loader,
        torch.optim.Adam,
        torch.nn.CrossEntropyLoss,
        epochs=10,
        learning_rate=0.001
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path", type=str, default="/home/cberruz/CS271/project_data/dataset.csv", help="The path to the dataset")
    parser.add_argument(
        "--n-articles", type=int, default=None,
        help="Use this if you want to truncate the dataset to a maximum number of articles per type."
    )

    args = parser.parse_args()
    main(
        args.csv_path,
        args.n_articles
    )
