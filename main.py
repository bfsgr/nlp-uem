import argparse
import os
import re
import sys
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from xml.etree import ElementTree

import nltk
import numpy as np
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
    sentences: list[str]
    bag_of_words: Counter
    objective: str
    problem: str
    method: str
    contribuitions: str
    references: list[str]

    def __init__(self, text: str):
        self.text = self.clear_text(text)
        self.references = self.find_references(text)
        self.sentences = to_sentences(self.text)

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

        # ([1]  X ...) [2]
        reference_match = r'(\[[0-9]+\].*[\s\S]*?(?=\[[0-9]+\]|$))'

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

        def ranquear_objetivo(list_of_sentences: list[str], sentence: IndexToSentence, avg_len: int) -> float:
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
                [w for w, _ in self.bag_of_words.most_common(5)])

            # Adiciona os substantivos mais comuns a query
            for word, pos in most_common:
                if (pos.startswith('N')):
                    query.append(word)

            points = bm25_no_idf(
                list_of_sentences, sentence.text, ' '.join(query), avg_words=avg_len)

            locality = (1.0 - (sentence.index /
                        (len(list_of_sentences) + 1)))

            return points + locality

        in_paper_re = re.compile(
            r'\b(?:in this paper|we propose|this paper presents?|this paper proposes?|is proposed in this paper)\b', re.IGNORECASE)

        maybe_objective = set()

        for (index, sentence) in enumerate(self.sentences):
            # se conter palavras como "in this paper" ou "we propose" é um forte indicativo de objetivo
            if (in_paper_re.findall(sentence)):
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

        objective_with_index = [IndexToSentence(i, self.sentences[i])
                                for i in maybe_objective]

        objectives = [obj.text for obj in objective_with_index]

        avg_len = np.mean([len(obj) for obj in objectives])

        sorted_objetives = sorted(objective_with_index,
                                  key=lambda x: ranquear_objetivo(objectives, x, avg_len), reverse=True)

        match = sorted_objetives[0][1] if len(
            sorted_objetives) > 0 else 'No objective found'

        self.objective = match.replace('\n', ' ').strip()

        return self.objective

    def search_for_problem(self) -> str:

        #
        #   Essa função busca por um problema no texto
        #   baseado em estruturas gramaticais e palavras-chave
        #

        def ranquear_problema(list_of_sentences: list[str], sentence: IndexToSentence, avg_len: int) -> float:
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
                list_of_sentences, sentence.text, ' '.join(query), avg_words=avg_len)

            locality = (1.0 - (sentence.index /
                        (len(list_of_sentences) + 1)))

            return points + locality

        maybe_problem = set()

        for (index, sentence) in enumerate(self.sentences):
            if (sentence.strip() == ''):
                continue

            GRAMMAR = [
                # the well-known problem of
                ChunkRule(
                    '<DT|CD><JJ><NN|NNS><IN>',
                    'Delimitador|Cardinal, Adjetivo, Substantivo, Preposição'),
                # lacks better security
                ChunkRule(
                    '<NNS><JJ><N.*>',
                    'Substantivo plural, Adjetivo, Substantivo/Nome próprio'),
                # such as
                ChunkRule(
                    '<JJ><IN>',
                    'Adjetivo, Preposição'),
                # security has always been
                ChunkRule(
                    '<NNS><VBZ><RB>?<VBN>',
                    'Substantivo plural, verbo-presente, advérbio?, verbo-presente-participio'),
                # this can prevent
                ChunkRule(
                    '<DT><MD><VB>',
                    'Delimitador, verbo-modal, verbo'),
                # by solving the
                ChunkRule(
                    '<IN><VBG><DT>',
                    'Preposição, verbo-gerundio, delimitador'),

            ]
            if self.match_grammar(sentence, GRAMMAR):
                maybe_problem.add(index)

        problems_with_index = [IndexToSentence(i, self.sentences[i])
                               for i in maybe_problem]

        problems = [obj.text for obj in problems_with_index]

        avg_len = np.mean([len(obj) for obj in problems])

        sorted_problems = sorted(problems_with_index,
                                 key=lambda x: ranquear_problema(problems, x, avg_len), reverse=True)

        match = sorted_problems[0][1] if len(
            sorted_problems) > 0 else 'No problem found'

        self.problem = match.replace('\n', ' ').strip()

        return self.problem

    def search_for_methods(self) -> str:

        #
        #   Essa função busca pela metodologia no texto
        #   baseado em estruturas gramaticais e palavras-chave
        #

        def ranquear_metodos(list_of_sentences: list[str], sentence: IndexToSentence, avg_len: int) -> float:
            #
            #   Essa função ranqueia metodologias baseado na sua posição no texto
            #   e o escore de query BM25
            #   De forma que metodologias no começo do texto, com palavras da query
            #   pontuem mais
            #
            query = [
                'analysis',
                'methodology',
                'content',
                'survey',
                'review',
                'evaluation',
                'comparative',
                'extended',
                'overview',
                'state-of-the-art'
                'discussed',
                'evaluated',
                'compared',
                'paper',
                'simulation',
                'utilizing',
                'investigate',
                'experiment',
                'relies'

            ]

            points = bm25_no_idf(
                list_of_sentences, sentence.text, ' '.join(query), avg_words=avg_len)

            return points

        maybe_method = set()

        comparative_re = re.compile(
            r'\b(?:comparative analysis?|by utilizing|this paper|evaluation of|analysis of?|is? extended|relies on|experimentation)\b', re.IGNORECASE)

        negative_re = re.compile(
            r'contribution|section|the associate editor|discussion', re.IGNORECASE)

        for (index, sentence) in enumerate(self.sentences):
            if (comparative_re.search(sentence) and not negative_re.search(sentence)):
                maybe_method.add(index)
                continue

            if (negative_re.search(sentence)):
                continue

            if (sentence.strip() == ''):
                continue

            GRAMMAR = [
                # problem is extended to the
                ChunkRule(
                    '<NN><VBZ><VBN><TO><DT>',
                    'Substantivo, verbo-presente, verbo-presente-participio, preposição, delimitador'),
                # comparative analysis of several ALP
                ChunkRule(
                    '<JJ><NN><IN><JJ><NN|NNP|NNS>',
                    'Adjetivo, substantivo, preposição, adjetivo, substantivo'),
                # using the measurement methodology
                ChunkRule(
                    '<VBG><DT><NN>',
                    'Verbo-gerundio, delimitador, substantivo'),
                # we investigate the performance
                ChunkRule(
                    '<PRP><VBP><DT><JJ>',
                    'Pronome, verbo-presente, delimitador, adjetivo'),
                # evaluated and compared to
                ChunkRule(
                    '<VBN><CC><VBN><TO>',
                    'Verbo-presente-participio, conjunção-coordenativa, verbo-presente-participio, preposição'),
                # experiments are conducted in this paper
                ChunkRule(
                    '<NNS><VBP><VBN><IN><DT><NN>',
                    'Substantivo plural, verbo-presente, verbo-presente-participio'),

            ]

            if self.match_grammar(sentence, GRAMMAR):
                maybe_method.add(index)

        method_with_index = [IndexToSentence(i, self.sentences[i])
                             for i in maybe_method]

        methods = [obj.text for obj in method_with_index]

        avg_len = np.mean([len(obj) for obj in methods])

        sorted_methods = sorted(method_with_index,
                                key=lambda x: ranquear_metodos(methods, x, avg_len), reverse=True)

        match = sorted_methods[0][1] if len(
            sorted_methods) > 0 else 'No method found'

        self.method = match.replace('\n', ' ').strip()

        return self.method

    def search_for_contribuitions(self) -> str:

        #
        #   Essa função busca pelas contribuições no texto
        #   baseado em estruturas gramaticais e palavras-chave
        #

        def ranquear_contribuicoes(list_of_sentences: list[str], sentence: IndexToSentence, avg_len: int) -> float:
            #
            #   Essa função ranqueia contribuições baseado na sua posição no texto
            #   e o escore de query BM25
            #   De forma que metodologias no começo do texto, com palavras da query
            #   pontuem mais
            #
            query = [
                'contribuition',
                'paper',
                'summarized',
                'results',
                'offers',
                'highlights',
            ]

            localidade = (1.0 - (sentence.index /
                                 (len(list_of_sentences) + 1)))

            points = bm25_no_idf(
                list_of_sentences, sentence.text, ' '.join(query), avg_words=avg_len)

            return points - localidade

        maybe_contrib = set()

        comparative_re = re.compile(
            r'contribution|contribute|we proposed|based on the results|demonstrate|similar|in this paper|this paper', re.IGNORECASE)

        negative_re = re.compile(
            r'section|objective', re.IGNORECASE)

        for (index, sentence) in enumerate(self.sentences):
            if (comparative_re.search(sentence) and not negative_re.search(sentence)):
                maybe_contrib.add(index)
                continue

            if (negative_re.search(sentence)):
                continue

            if (sentence.strip() == ''):
                continue

            GRAMMAR = [
                # the main contribution of this paper
                ChunkRule(
                    '<DT><JJ><NN><IN><DT><NN>',
                    'Delimitador, adjetivo, substantivo, preposição, delimitador, substantivo'),
                # Based on the results of
                ChunkRule(
                    '<VBN><IN><DT><NNS><IN>',
                    'Verbo-presente-participio, preposição, delimitador, substantivo plural, preposição'),
                # 'gives similar security with
                ChunkRule(
                    '<VBZ><JJ><NN><IN>',
                    'Verbo-presente, adjetivo, substantivo, preposição'),
            ]

            if self.match_grammar(sentence, GRAMMAR):
                maybe_contrib.add(index)

        contrib_with_index = [IndexToSentence(i, self.sentences[i])
                              for i in maybe_contrib]

        contribs = [obj.text for obj in contrib_with_index]

        avg_len = np.mean([len(obj) for obj in contribs])

        sorted_contribs = sorted(contrib_with_index,
                                 key=lambda x: ranquear_contribuicoes(contribs, x, avg_len), reverse=True)

        match = sorted_contribs[0][1] if len(
            sorted_contribs) > 0 else 'No contribution found'

        self.contribuitions = match.replace('\n', ' ').strip()

        return self.contribuitions


def show_results(file: str, paper: ScyPaper):
    print("\n=====================================\n")
    print("Arquivo: ", file + '\n')
    print("Objetivo => ", paper.objective + '\n')
    print("Problema => ", paper.problem + '\n')
    print("Metodologia => ", paper.method + '\n')
    print("Contribuições => ", paper.contribuitions + '\n')

    print("Termos mais citados =>")

    for word, count in paper.bag_of_words.most_common(10):
        print(word, str(count))

    print('\n')

    # print("Referências =>")
    # for ref in paper.references:
    #     print(ref + '\n')


def write_to_file(file: str, paper: ScyPaper):
    root = ElementTree.Element('paper')
    filename = ElementTree.SubElement(root, 'filename')
    filename.text = os.path.basename(file)

    objective = ElementTree.SubElement(root, 'objective')
    objective.text = paper.objective

    problem = ElementTree.SubElement(root, 'problem')
    problem.text = paper.problem

    method = ElementTree.SubElement(root, 'method')
    method.text = paper.method

    contribuitions = ElementTree.SubElement(root, 'contribuitions')
    contribuitions.text = paper.contribuitions

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


def process_file(fullpath: str) -> ScyPaper:
    text = extrair_texto(fullpath)

    paper = ScyPaper(text)

    paper.count_words(text)
    paper.search_for_contribuitions()
    paper.search_for_objective()
    paper.search_for_problem()
    paper.search_for_methods()

    write_to_file(fullpath, paper)

    return paper


def main():
    parser = argparse.ArgumentParser(
        prog='Paper Analyzer')

    parser.add_argument('path', help='Path to a file or directory', type=str)
    parser.add_argument(
        '-s', '--search', help='Search for a term in a file', type=str)

    args = parser.parse_args()

    path = args.path

    if os.path.isfile(path) and path.endswith('.pdf'):
        paper = process_file(path)

        show_results(path, paper)

        return

    if os.path.isdir(path):
        with ProcessPoolExecutor() as executor:
            futures = dict()

            for filename in os.listdir(path):
                if filename.endswith('.pdf'):
                    fullpath = os.path.join(path, filename)

                    futures[executor.submit(process_file, fullpath)] = fullpath

            for future in as_completed(futures):
                fullpath = futures[future]

                paper = future.result()

                show_results(fullpath, paper)


if (__name__ == '__main__'):
    main()
