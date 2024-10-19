from models import LogisticRegression
from embeddings import Word2VecEmbedder
from data_processing import RawHumanChatBotData, HumanChatBotDataset, ReusableGenerator
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

from typing import Optional, Iterable
import torch
from torch.utils.data import DataLoader, random_split
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')


def train_word2vec_on_article_type(
    dataset: RawHumanChatBotData,
    article_type: str,
    n_articles: Optional[int] = None,
    vector_size: int = 100,
    min_count: int = 1,
    window=5
) -> Word2Vec:
    """
    Train a Word2Vec model on the chatbot data.
    """
    print(f"Training Word2Vec model on {article_type} articles...")

    def get_articles_wrapper():
        return dataset.get_articles(article_type, n_articles)

    articles = ReusableGenerator(get_articles_wrapper)

    def data_generator():
        for article in articles:
            article = article.replace("\n", " ")
            for i in sent_tokenize(article):
                for j in word_tokenize(i):
                    yield j.lower()

    data_gen = ReusableGenerator(data_generator)
    model = Word2Vec(data_gen, min_count=min_count,
                     vector_size=vector_size, window=window)
    return model


def main():
    """A simple example of how to use Word2Vec with a Logistic Regression model
    for classification.
    """
    vector_size = 100
    # This basically truncates the entire available
    # dataset to 10,000 articles per type
    total_articles_per_type = 10_000
    print("Loading the HumanChatBotDataset...")
    raw_data = RawHumanChatBotData.from_csv(
        "/home/cberruz/CS271/project_data/dataset.csv",
        n_articles_per_type=total_articles_per_type
    )

    articles_used_in_training = 10_000

    word2vec_gpt = train_word2vec_on_article_type(
        raw_data, "gpt", n_articles=articles_used_in_training, vector_size=vector_size)
    word2vec_human = train_word2vec_on_article_type(
        raw_data, "human", n_articles=articles_used_in_training, vector_size=vector_size)

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
    main()
