import os
from xml.etree import ElementTree as ET

import PyPDF2


def extrair_texto(path: str) -> str:
    if (os.path.isfile(path + '.cache')):
        with open(path + '.cache', 'r') as cache:
            return cache.read()

    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        text: str = ''

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()

    with open(path + '.cache', 'w') as cache:
        cache.write(text)

    return text


def xml_reader(file_path):
    # Carregar o arquivo XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extrair as informações das tags
    filename = root.find('filename').text
    objective = root.find('objective').text
    problem = root.find('problem').text
    method = root.find('method').text
    contribuitions = root.find('contribuitions').text

    # Extrair informações da tag most_cited e armazenar em uma lista de tuplas
    most_cited = [('Quant', 'Termo')]
    for word_tag in root.find('most_cited'):
        count = word_tag.attrib['count']
        word = word_tag.text
        most_cited.append((count, word))

    return filename, objective, problem, method, contribuitions, most_cited
