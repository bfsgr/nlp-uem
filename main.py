import os
import re
import sys

import nltk
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule

from bm25 import bm25_no_idf
from leitor import extrair_texto
from text import composite, remove_punctuation, to_sentences, to_tokenized


class ScyPaper:
    def __init__(self, text: str):
        self.text = text
        self.objetives = self.search_for_objective()

    #
    #   Essa função ranqueia um objetivo baseado na sua posição no texto
    #   o escore BM25 de palavras chave do objetivo
    #   e o tamanho do objetivo em relação ao menor objetivo
    #   De forma que objetivos no começo do texto, com palavras chave e maior tamanho
    #   são ranqueados mais alto
    #
    #
    def __ranquear_objetivo(list_of_sentences: list[str], sentence: tuple[int, str]) -> float:
        query = [
            'objective',
            'aim',
            'goal',
            'purpose',
            'novel',
            'paper',
            'achieve',
            'target',
            'scope',
            'problem',
            'solution',
            'emphasis',
            'study',
            'attempts',
            'present',
            'approach',
            'provide',
            'introduces',
            'enhance',
            'solving',
            'focus',
        ]

        points = bm25_no_idf(
            list_of_sentences, sentence[1], ' '.join(query))

        sentence_index = sentence[0]

        locality = (1.0 - (sentence_index /
                    (len(list_of_sentences) + 1)))

        return points + locality

    def search_for_objective(self):
        sentences = to_sentences(self.text)

        in_paper_re = re.compile(
            r'\b(?:in this paper|we propose|this paper presents?|this paper proposes?|is proposed in this paper)\b', re.IGNORECASE)

        maybe_objective = set()

        for (index, sentence) in enumerate(sentences):
            # se conter palavras como "in this paper" ou "we propose" é um forte indicativo de objetivo
            if (in_paper_re.match(sentence)):
                maybe_objective.add(index)
                continue

            words = composite(
                to_tokenized,
                remove_punctuation,
            )(sentence)

            tagged = nltk.pos_tag(words)

            tree = nltk.Tree('DOC', [(token, pos)
                                     for token, pos in tagged])

            # busca por estruturas gramaticais que indicam objetivo
            OBJECTIVE_FORMATS = [
                ChunkRule(
                    '<DT><NN><VBZ><DT>?<JJ>?<N.*>', 'Delimitador, substantivo, verbo, substantivo'),  # this paper proposes (a new) security

                ChunkRule('(<IN><DT>)?<NN><PRP><VB>.*',
                          'substantivo, pronome verbo'),  # (in this) paper we present

                ChunkRule('<PRP><VBP><DT|CD>?<JJ>?<NN>',
                          'Pronome, verbo-participio, dilimitador, adjetivo, substantivo'),  # we propose ((a?, three?) new?) approaches

                ChunkRule('<NN|NNP><VBZ><VBN>',
                          'Substantivo/Nome próprio, verbo-presente, verbo-presente-participio'),  # something is proposed
            ]

            chunk_parser = RegexpChunkParser(
                OBJECTIVE_FORMATS, chunk_label='OBJECTIVE')

            chunks = chunk_parser.parse(tree)

            for chunk in chunks.subtrees():
                if chunk.label() == 'OBJECTIVE':
                    maybe_objective.add(index)

        objective_with_index = [(i, sentences[i])
                                for i in list(maybe_objective)]

        objectives = [obj[1] for obj in objective_with_index]

        sorted_objetives = sorted(objective_with_index,
                                  key=lambda x: ScyPaper.__ranquear_objetivo(
                                      objectives, x), reverse=True)[:3]

        return [s[1] for s in sorted_objetives]


def process_file(path: str):
    text = extrair_texto(path)

    paper = ScyPaper(text)

    print("=====================================\n")
    print("FILE: ", path)
    if len(paper.objetives) > 0:
        with open(path + '.txt', 'w') as file:
            for objective in paper.objetives:
                print("Objetive => ", objective + '\n')
                file.write("Objetive => " + objective + '\n\n')
    else:
        print("=> No objectives found\n")
    print("=====================================\n")


def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    PASTA = './artigos-teste'

    i = 0

    args = sys.argv[1:]

    if (len(args) > 0):
        path = args[0]

        if os.path.isfile(path) and path.endswith('.pdf'):
            process_file(path)
            return

        if os.path.isdir(path):
            PASTA = path

    for filename in os.listdir(PASTA):
        if not filename.endswith('.pdf'):
            continue

        process_file(os.path.join(PASTA, filename))

        i += 1


if (__name__ == '__main__'):
    main()
