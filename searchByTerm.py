import os

import bm25
import leitor

def search_by_term(search_term, directory_path):
    article_corpus = []

    list_of_results = []

    for archive in os.listdir(directory_path):
        if archive.endswith(".pdf"):
            archive_path = os.path.join(directory_path, archive)
            article_corpus.append(leitor.extrair_texto(archive_path))
    i = 0

    for archive in os.listdir(directory_path):
        if archive.endswith(".pdf"):
            points = bm25.bm25_no_idf(
                article_corpus, article_corpus[i], search_term)
            i += 1
            list_of_results.append(points)

    return list_of_results
