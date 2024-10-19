from models import LogisticRegression
from embeddings import Word2VecEmbedder
from data_processing import RawHumanChatBotData, HumanChatBotDataset, ReusableGenerator
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec


import argparse
from typing import Optional, Iterable
import torch
from torch.utils.data import DataLoader, random_split
import nltk


def train_word2vec_on_article_type(
    dataset: RawHumanChatBotData,
    article_type: str,
    vector_size: int = 100,
    min_count: int = 1,
    window=5
) -> Word2Vec:
    """
    Train a Word2Vec model on the entire dataset.
    """
    print(f"Training Word2Vec model on {article_type} articles...")

    def get_articles_wrapper():
        return dataset.get_articles(article_type)

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


def main(csv_path: str, n_articles_per_type: Optional[int] = None):
    """A simple example of how to use Word2Vec with a Logistic Regression model
    for classification.
    """
    print(
        f"Word2vec training using data from {csv_path} and n_articles {n_articles_per_type}")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    vector_size = 100
    print("Loading the HumanChatBotDataset...")

    raw_data = RawHumanChatBotData.from_csv(
        csv_path,
        n_articles_per_type=n_articles_per_type
    )

    word2vec_gpt = train_word2vec_on_article_type(
        raw_data, "gpt", vector_size=vector_size)
    word2vec_human = train_word2vec_on_article_type(
        raw_data, "human", vector_size=vector_size)

    embedders_per_type = {
        "gpt": Word2VecEmbedder(word2vec_gpt),
        "human": Word2VecEmbedder(word2vec_human)
    }
    article_type_to_class_num = {
        "gpt": 0,
        "human": 1
    }
    # save them, so we can reuse them later.
    word2vec_gpt.save("word2vec_gpt.model")
    word2vec_human.save("word2vec_human.model")

    dataset = HumanChatBotDataset(
        raw_data.data, embedders_per_type, article_type_to_class_num
    )
    # pin the seed so we can reproduce the results
    ran = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], ran)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # create the model
    model = LogisticRegression(vector_size, len(article_type_to_class_num))
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
