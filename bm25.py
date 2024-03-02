import numpy as np

from text import (composite, remove_punctuation, remove_stop_words,
                  to_lemmatize, to_stem, to_tokenized)


#
#   Implentação da função BM25 sem considerar o IDF
#
def bm25_no_idf(corpus: list[str], doc: str, query: str, **kwargs) -> float:
    K = 2.0
    B = 0.75

    def term_freq(words: list[str], term: str) -> int:
        return words.count(term)

    prepare = composite(
        to_tokenized,
        remove_stop_words,
        remove_punctuation,
        to_lemmatize,
        to_stem
    )

    words = prepare(doc)
    query = prepare(query)

    points = 0

    avg_words = kwargs.get('avg_words',  np.mean(
        [len(prepare(d)) for d in corpus]))

    for q in query:
        tf = term_freq(words, q)

        points += tf * (K + 1) / (tf + K *
                                  (1 - B + B * len(words) / avg_words))

    return points
