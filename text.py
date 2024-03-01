
import string
from functools import reduce

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))
puctuation = set(string.punctuation)

stop_words.add('et')
stop_words.add('al')
puctuation.add('·')
puctuation.add('``')
puctuation.add("''")


def composite(*func):
    def compose(f, g):
        return lambda x: f(g(x))

    return reduce(compose, reversed(func), lambda x: x)


def to_sentences(text: str) -> list[str]:
    return nltk.sent_tokenize(text)


def to_tokenized(text: str):
    return word_tokenize(text)


def remove_stop_words(text: list[str]):
    return [w for w in text if w not in stop_words]


def remove_single_char(text: list[str]):
    return [w for w in text if len(w) > 1]


def remove_numbers(text: list[str]):
    return [w for w in text if not w.isdigit()]


def remove_delimiters(text: list[str]):
    return [w for w in text if nltk.pos_tag([w])[0][1] != 'DT']


def remove_punctuation(text: list[str]):
    return [w for w in text if w not in puctuation]


def to_stem(text: list[str]):
    stemmer = PorterStemmer()

    return [stemmer.stem(word) for word in text]


def to_lemmatize(text: list[str]):
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in text]
