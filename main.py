import csv
import os
import re
import sys
from collections import namedtuple

import nltk
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule

from bm25 import bm25_no_idf
from leitor import extrair_texto
from text import composite, remove_punctuation, to_sentences, to_tokenized

IndexToSentence = namedtuple('IndexToSentence', ['index', 'text'])


class ScyPaper:
    def __init__(self, text: str):
        self.text = self.clear_text(text)
        self.objective = self.search_for_objective()
        self.problem = self.search_for_problem()

    def clear_text(self, text: str) -> str:
        # remove todo texto até a primeira ocorrência de "abstract"
        until_abstract = re.compile(
            r'[\s\S]*?abstract', re.IGNORECASE | re.MULTILINE)

        r = re.sub(until_abstract, '', text.strip(), count=1)
        return r

    def match_grammar(self, sentence: str, grammar: list[ChunkRule]) -> bool:
        words = composite(
            to_tokenized,
            remove_punctuation,
        )(sentence)

        tagged = nltk.pos_tag(words)

        tree = nltk.Tree('DOC', [(token, pos)
                                 for token, pos in tagged])

        chunk_parser = RegexpChunkParser(
            grammar, chunk_label='MATCHED')

        chunks = chunk_parser.parse(tree)

        for chunk in chunks.subtrees():
            if chunk.label() == 'MATCHED':
                return True

        return False

    def search_for_objective(self) -> str:
        #
        #   Essa função busca por um objetivo no texto
        #   baseado em estruturas gramaticais e palavras-chave
        #   que indicam um objetivo
        #

        def ranquear_objetivo(list_of_sentences: list[str], sentence: IndexToSentence) -> float:
            #
            #   Essa função ranqueia um objetivo baseado na sua posição no texto
            #   e o escore de query BM25
            #   De forma que objetivos no começo do texto, com palavras da query
            #   pontuem mais
            #
            query = [
                'objective',
                'paper',
                'problem',
                'present',
                'approach',
                'proposes',
                'proposed',
                'explores'
            ]

            points = bm25_no_idf(
                list_of_sentences, sentence.text, ' '.join(query))

            locality = (1.0 - (sentence.index /
                        (len(list_of_sentences) + 1)))

            return points + locality

        sentences = to_sentences(self.text)

        in_paper_re = re.compile(
            r'\b(?:in this paper|we propose|this paper presents?|this paper proposes?|is proposed in this paper)\b', re.IGNORECASE)

        maybe_objective = set()

        for (index, sentence) in enumerate(sentences):
            # se conter palavras como "in this paper" ou "we propose" é um forte indicativo de objetivo
            if (in_paper_re.match(sentence)):
                maybe_objective.add(index)
                continue

            # descarta frases vazias
            if (sentence.strip() == ''):
                continue

            GRAMMAR = [
                # this paper proposes a new security
                # this paper proposes a method
                # this paper proposes improved standards
                ChunkRule(
                    '<DT><NN><VBZ><DT>?<JJ>?<N.*>',
                    'Delimitador, substantivo, verbo, substantivo'),

                # paper we present
                # in this paper we present
                ChunkRule('(<IN><DT>)?<NN><PRP><VB>.*',
                          'substantivo, pronome verbo'),

                # we propose a method
                # we propose three methods
                # we propose three new methods
                # we propose a new method
                ChunkRule('<PRP><VBP><DT|CD>?<JJ>?<NN>',
                          'Pronome, verbo-participio, delimitador, adjetivo, substantivo'),

                # something is proposed
                # TurboJPEG is proposed
                ChunkRule('<NN|NNP><VBZ><VBN>',
                          'Substantivo/Nome próprio, verbo-presente, verbo-presente-participio'),
            ]

            if self.match_grammar(sentence, GRAMMAR):
                maybe_objective.add(index)

        objective_with_index = [IndexToSentence(i, sentences[i])
                                for i in maybe_objective]

        objectives = [obj.text for obj in objective_with_index]

        sorted_objetives = sorted(objective_with_index,
                                  key=lambda x: ranquear_objetivo(objectives, x), reverse=True)

        match = sorted_objetives[0][1] if len(
            sorted_objetives) > 0 else 'No objective found'

        return match.replace('\n', ' ').strip()

    def search_for_problem(self) -> str:

        #
        #   Essa função busca por um problema no texto
        #   baseado em estruturas gramaticais e palavras-chave
        #

        def ranquear_problema(list_of_sentences: list[str], sentence: IndexToSentence) -> float:
            #
            #   Essa função ranqueia um problema baseado na sua posição no texto
            #   e o escore de query BM25
            #   De forma que problemas no começo do texto, com palavras da query
            #   pontuem mais
            #
            query = [
                'problem',
                'issue',
                'lacks',
                'challenge',
                'difficult',
                'solve'
            ]

            points = bm25_no_idf(
                list_of_sentences, sentence.text, ' '.join(query))

            locality = (1.0 - (sentence.index /
                        (len(list_of_sentences) + 1)))

            return points + locality

        sentences = to_sentences(self.text)

        maybe_problem = set()

        for (index, sentence) in enumerate(sentences):
            if (sentence.strip() == ''):
                continue

            GRAMMAR = [
                # the well-known problem of
                ChunkRule(
                    '<DT|CD><JJ><NN|NNS><IN>',
                    'test'),
                # lacks better security
                ChunkRule(
                    '<NNS><JJ><N.*>',
                    'test2'),
                # such as
                ChunkRule(
                    '<JJ><IN>',
                    'test3'),
                # security has always been
                ChunkRule(
                    '<NNS><VBZ><RB>?<VBN>',
                    'test4'),
                # this can prevent
                ChunkRule(
                    '<DT><MD><VB>',
                    'test4'),
                # by solving the
                ChunkRule(
                    '<IN><VBG><DT>',
                    'test5'),

            ]
            if self.match_grammar(sentence, GRAMMAR):
                maybe_problem.add(index)

        problems_with_index = [IndexToSentence(i, sentences[i])
                               for i in maybe_problem]

        problems = [obj.text for obj in problems_with_index]

        sorted_problems = sorted(problems_with_index,
                                 key=lambda x: ranquear_problema(problems, x), reverse=True)

        match = sorted_problems[0][1] if len(
            sorted_problems) > 0 else 'No problem found'

        return match.replace('\n', ' ').strip()


def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    pasta = './artigos-teste'

    args = sys.argv[1:]

    if (len(args) > 0):
        path = args[0]

        if os.path.isfile(path) and path.endswith('.pdf'):
            text = extrair_texto(path)

            paper = ScyPaper(text)

            print("\n=====================================\n")
            print("Arquivo: ", path + '\n')
            print("Objetivo => ", paper.objective + '\n')
            print("Problema => ", paper.problem + '\n')

            with open(path + '.csv', 'w') as file:
                writer = csv.DictWriter(
                    file, fieldnames=['name', 'objective', 'problem'])
                writer.writeheader()
                writer.writerow(
                    {'name': path, 'objective': paper.objective, 'problem': paper.problem})

            return

        if os.path.isdir(path):
            pasta = path

    for filename in os.listdir(pasta):
        if filename.endswith('.pdf'):
            fullpath = os.path.join(pasta, filename)

            text = extrair_texto(fullpath)

            paper = ScyPaper(text)

            print("\n=====================================\n")
            print("Arquivo: ", fullpath + '\n')
            print("Objetivo => ", paper.objective + '\n')
            print("Problema => ", paper.problem + '\n')

            with open(fullpath + '.csv', 'w') as file:
                writer = csv.DictWriter(
                    file, fieldnames=['name', 'objective', 'problem'], delimiter=';')
                writer.writeheader()
                writer.writerow(
                    {'name': fullpath, 'objective': paper.objective, 'problem': paper.problem})


if (__name__ == '__main__'):
    main()
