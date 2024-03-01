import argparse
import os
import re
import sys
from collections import Counter, namedtuple
from xml.etree import ElementTree

import nltk
from nltk.chunk import RegexpChunkParser
from nltk.chunk.regexp import ChunkRule

from bm25 import bm25_no_idf
from leitor import extrair_texto
from text import (composite, illegal_xml_chars_RE, remove_delimiters,
                  remove_numbers, remove_punctuation, remove_single_char,
                  remove_stop_words, to_sentences, to_tokenized)

IndexToSentence = namedtuple('IndexToSentence', ['index', 'text'])


class ScyPaper:
    text: str
    bag_of_words: Counter
    objective: str
    problem: str
    references: list[str]

    def __init__(self, text: str):
        self.text = self.clear_text(text)
        self.references = self.find_references(text)

    def count_words(self, text: str) -> Counter:
        words = composite(
            to_tokenized,
            remove_stop_words,
            remove_punctuation,
            remove_numbers,
            remove_single_char,
            remove_delimiters,
        )(text)

        self.bag_of_words = Counter(words)

        return self.bag_of_words

    def clear_text(self, text: str) -> str:
        # remove todo texto até a primeira ocorrência de "abstract"
        until_abstract = re.compile(
            r'[\s\S]*?abstract', re.IGNORECASE | re.MULTILINE)

        r = re.sub(until_abstract, '', text.strip(), count=1)

        # remove todo texto após a última ocorrência de "references"
        after_references = re.compile(
            r'references[\s\S]*', re.IGNORECASE | re.MULTILINE)

        r = re.sub(after_references, '', r.strip(), count=1)

        return r

    def find_references(self, text: str) -> list[str]:
        # remove todo texto até a primeira ocorrência de "references"
        until_references = re.compile(
            r'[\s\S]*?(references|bibliography)', re.IGNORECASE | re.MULTILINE)

        isolated = re.sub(until_references, '', text.strip(), count=1)

        # [1]  X ... 1.
        reference_match = r'\[[0-9]+\] [A-Z].*[\s\S]*?[0-9]\.'

        references = re.findall(reference_match, isolated)

        for i in range(len(references)):
            references[i] = references[i].replace('\n', ' ').strip()

        return references

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

            most_common = nltk.pos_tag(
                [w for w, _ in self.bag_of_words.most_common(10)])

            # Adiciona os substantivos mais comuns a query
            for word, pos in most_common:
                if (pos.startswith('N')):
                    query.append(word)

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

        self.objective = match.replace('\n', ' ').strip()

        return self.objective

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

        self.problem = match.replace('\n', ' ').strip()

        return self.problem


def show_results(file: str, paper: ScyPaper):
    print("\n=====================================\n")
    print("Arquivo: ", file + '\n')
    print("Objetivo => ", paper.objective + '\n')
    print("Problema => ", paper.problem + '\n')
    print("Termos mais citados =>")

    for word, count in paper.bag_of_words.most_common(10):
        print(word, str(count))

    print('\n')

    print("Referências =>")
    for ref in paper.references:
        print(ref + '\n')


def write_to_file(file: str, paper: ScyPaper):
    root = ElementTree.Element('paper')
    filename = ElementTree.SubElement(root, 'filename')
    filename.text = os.path.basename(file)

    objective = ElementTree.SubElement(root, 'objective')
    objective.text = paper.objective

    problem = ElementTree.SubElement(root, 'problem')
    problem.text = paper.problem

    most_cited = ElementTree.SubElement(root, 'most_cited')
    for word, count in paper.bag_of_words.most_common(10):
        word_node = ElementTree.SubElement(most_cited, 'word')
        word_node.text = word
        word_node.set('count', str(count))

    references = ElementTree.SubElement(root, 'references')
    for ref in paper.references:
        ref_node = ElementTree.SubElement(references, 'ref')
        ref_node.text = ref

    tree = ElementTree.ElementTree(root)

    ElementTree.indent(tree)

    content = ElementTree.tostring(
        root, encoding='unicode', xml_declaration=True)

    content = illegal_xml_chars_RE.sub('', content)

    with open(file + '.xml', 'wb') as f:
        f.write(content.encode('utf-8'))


def main():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    parser = argparse.ArgumentParser(
        prog='Paper Analyzer')

    parser.add_argument('path', help='Path to a file or directory', type=str)
    parser.add_argument(
        '-s', '--search', help='Search for a term in a file', type=str)

    args = parser.parse_args()

    path = args.path

    if os.path.isfile(path) and path.endswith('.pdf'):
        text = extrair_texto(path)

        paper = ScyPaper(text)

        paper.count_words(text)
        paper.search_for_objective()
        paper.search_for_problem()

        show_results(path, paper)
        write_to_file(path, paper)

        return

    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.pdf'):
                fullpath = os.path.join(path, filename)

                text = extrair_texto(fullpath)

                paper = ScyPaper(text)

                paper.count_words(text)
                paper.search_for_objective()
                paper.search_for_problem()

                show_results(fullpath, paper)
                write_to_file(fullpath, paper)


if (__name__ == '__main__'):
    main()
