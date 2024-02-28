import PyPDF2


def extrair_texto(path: str) -> str:
    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        text: str = ''

        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()

    return text
