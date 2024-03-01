
import re
import string
import sys
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


# Regex para remover caracteres ilegais em XML
_illegal_unichrs = [(0x00, 0x08), (0x0B, 0x0C), (0x0E, 0x1F),
                    (0x7F, 0x84), (0x86, 0x9F),
                    (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF)]
if sys.maxunicode >= 0x10000:
    _illegal_unichrs.extend([(0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF),
                             (0x3FFFE, 0x3FFFF), (0x4FFFE, 0x4FFFF),
                             (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                             (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF),
                             (0x9FFFE, 0x9FFFF), (0xAFFFE, 0xAFFFF),
                             (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                             (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF),
                             (0xFFFFE, 0xFFFFF), (0x10FFFE, 0x10FFFF)])

_illegal_ranges = ["%s-%s" % (chr(low), chr(high))
                   for (low, high) in _illegal_unichrs]

illegal_xml_chars_RE = re.compile(u'[%s]' % u''.join(_illegal_ranges))
