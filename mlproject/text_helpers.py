from functools import wraps
from typing import Generator, List
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize


def run_once(func):
    """
    A decorator that runs a function only once.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return func(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


def run_before_decorator_factory(func):
    """
    A decorator factory that runs a function before the decorated function.
    """
    def decorator(g):
        @wraps(g)
        def wrapper(*args, **kwargs):
            func()
            return g(*args, **kwargs)
        return wrapper
    return decorator


@run_once
def download_nltk_data():
    """
    Download the necessary NLTK data.
    """
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('punkt')
    print("Download complete.")


requires_nltk_data = run_before_decorator_factory(download_nltk_data)


sent_tokenize = requires_nltk_data(sent_tokenize)
word_tokenize = requires_nltk_data(word_tokenize)


def sentence_word_tokenizer(
        text: str,
        remove_newlines: bool = True,
        lower_case: bool = True) -> Generator[List[str], None, None]:
    """
    Returns a stream of sentences, where each sentence is the list
    of its word tokens.

    Args:
        text (str): the text to tokenize
        remove_newlines (bool): whether to remove newlines
        lower_case (bool): whether to convert the text to lower case
    """
    if remove_newlines:
        text = text.replace("\n", " ")
    for i in sent_tokenize(text):
        sentence = []
        for j in word_tokenize(i):
            word = j.lower() if lower_case else j
            sentence.append(word)
        yield sentence
