import os
import tkinter
import tkinter.messagebox
from tkinter import filedialog

import customtkinter
from CTkTable import *

import leitor
import main
import searchByTerm

# Modes: "System" (standard), "Dark", "Light"
customtkinter.set_appearance_mode("System")
# Themes: "blue" (standard), "green", "dark-blue"
customtkinter.set_default_color_theme("blue")


class MainWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("FuSam")
        self.geometry(f"{1200}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Botao de selecionar diretorio de artigos
        self.button_change_directory = customtkinter.CTkButton(self, text="Selecionar diretório", width=200, height=80, corner_radius=10, font=customtkinter.CTkFont(
            size=20, weight="bold"), hover_color="#144870", command=self.init_directory)
        self.button_change_directory.grid(row=1, column=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(
            self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Analisador\nde texto\ncientifico", font=customtkinter.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.controller_search = None

    def brutal_init(self):

        self.frame_diretorio = customtkinter.CTkScrollableFrame(
            self.sidebar_frame, label_text="Diretório selecionado")
        self.frame_diretorio.grid(
            row=1, column=0,  padx=15, pady=15, sticky="nsew")

        self.button_change_directory.configure(command=self.change_directory)

        if not self.articles_path:
            for arquivo in os.listdir(self.articles_directory_path):
                if arquivo.endswith(".xml"):
                    self.articles_path.append(os.path.join(
                        self.articles_directory_path, arquivo))

        self.articles_titles_formatted = []
        self.articles_titles_formatted.append(("Artigos",))

        for title, in self.articles_titles:
            new_title = ''

            # Inserir '\n' a cada 30 caracteres
            for i in range(0, len(title), 25):
                new_title += title[i:i+25] + '\n'

            # Armazenar o novo título na nova lista de títulos
            self.articles_titles_formatted.append((new_title,))

        self.table_articles = CTkTable(self.frame_diretorio, width=190, values=self.articles_titles_formatted,
                                       header_color="#144870", hover_color='#1F6AA5', command=self.select_article)
        self.table_articles.grid(row=0, column=0, padx=5, pady=5)
        self.table_articles.select_row(0)

        self.appearance_mode_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="Modo de aparencia:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=[
                                                                       "Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=6, column=0, padx=20, pady=(10, 10))

        # most quoted terms
        self.frame_termos = customtkinter.CTkScrollableFrame(
            self, width=160, label_text="Termos mais citados")
        self.frame_termos.grid(row=0, column=2, padx=15,
                               pady=15, sticky="nsew")

        self.article_title, self.article_objective, self.article_problem, self.article_method, self.article_contribuitions, self.most_quoted_terms = leitor.xml_reader(
            self.articles_path[0])

        self.table_terms = CTkTable(
            self.frame_termos, width=40, values=self.most_quoted_terms, header_color="#144870", hover_color="#144870")
        self.table_terms.grid(row=0, column=0, padx=5, pady=5)

        # create main entry and button
        self.entry = customtkinter.CTkEntry(
            self, placeholder_text="Digite um termo para buscar nos artigos")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(
            20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text="Buscar", text_color=(
            "gray10", "#DCE4EE"), command=self.search_by_term)
        self.main_button_1.grid(row=3, column=3, padx=(
            20, 20), pady=(20, 20), sticky="nsew")

        self.number_of_searches = 0

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=0, column=1, padx=(
            20, 0), pady=(20, 0), sticky="nsew")

        # set default values
        self.appearance_mode_optionemenu.set("System")
        self.textbox.insert("0.0", "Informações extraídas\n\n" + "Titulo:\n" + self.article_title + "\n\nObjetivo:\n" + self.article_objective +
                            "\n\nProblema:\n" + self.article_problem + "\n\nMetodo:\n" + self.article_method + "\n\nContribuições:\n" + self.article_contribuitions)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def init_directory(self):
        self.articles_directory_path = filedialog.askdirectory()

        # Lista para armazenar os títulos dos artigos
        self.articles_titles = []
        self.articles_path = []

        # Iterar sobre os arquivos na pasta
        for arquivo in os.listdir(self.articles_directory_path):
            if arquivo.endswith(".pdf"):
                self.articles_titles.append((arquivo,))
            if arquivo.endswith(".xml"):
                self.articles_path.append(os.path.join(
                    self.articles_directory_path, arquivo))

        if not self.articles_path:
            estado_retorno = main.main(self.articles_directory_path)
        self.brutal_init()

    def change_directory(self):
        self.frame_diretorio.destroy()
        self.init_directory()

    def select_article(self, informacoes_da_celula_selecionada):

        if self.controller_search != None:
            for indice, tupla in enumerate(self.articles_titles_formatted):
                if tupla[0] == informacoes_da_celula_selecionada['value']:
                    indice_celula_selecionada = indice
        else:
            indice_celula_selecionada = informacoes_da_celula_selecionada['row']

        # Desseliciona uma linha caso exista uma selecionada anteriormente
        indice_linha_selecionada_anteriormente = self.table_articles.get_selected_row()[
            'row_index']
        if (indice_linha_selecionada_anteriormente != None):
            self.table_articles.deselect_row(
                indice_linha_selecionada_anteriormente)

        if indice_celula_selecionada != 0:
            # Seleciona visualmente a linha inteira da célula selecionada
            self.table_articles.select_row(indice_celula_selecionada)

            for caminho in self.articles_path:
                caminho_modificado = os.path.basename(caminho)
                if caminho_modificado[:20] == informacoes_da_celula_selecionada['value'][:20]:
                    novo_caminho = caminho

            # Ajustando a interface de acordo com o artigo selecionado
            self.textbox.delete("0.0", customtkinter.END)
            self.article_title, self.article_objective, self.article_problem, self.article_method, self.article_contribuitions, self.most_quoted_terms = leitor.xml_reader(
                novo_caminho)
            self.textbox.insert("0.0", "Informações extraídas\n\n" + "Titulo:\n" + self.article_title + "\n\nObjetivo:\n" + self.article_objective +
                                "\n\nProblema:\n" + self.article_problem + "\n\nMetodo:\n" + self.article_method + "\n\nContribuições:\n" + self.article_contribuitions)
            self.table_terms.configure(values=self.most_quoted_terms)

    def search_by_term(self):
        self.term_entered = self.entry.get()

        if self.number_of_searches == 0:
            self.frame_search = customtkinter.CTkScrollableFrame(
                self, width=280, label_text=f"Resultado da busca do termo <{self.term_entered}>")
            self.frame_search.grid(
                row=0, column=3, padx=15, pady=15, sticky="nsew")

            self.points_results_of_search = searchByTerm.search_by_term(
                self.term_entered, self.articles_directory_path)
            self.results_of_search = []

            for i in range(len(self.points_results_of_search)):
                self.results_of_search.append(
                    (self.articles_titles_formatted[i+1][0], round(float(self.points_results_of_search[i]), 3)))

            self.results_of_search_sorted = [('Artigo', 'Pontuacao')] + sorted(
                self.results_of_search, key=lambda x: x[1], reverse=True)

            self.table_search = CTkTable(self.frame_search, width=80, values=self.results_of_search_sorted,
                                         header_color="#144870", hover_color="#144870", command=self.select_article)
            self.table_search.grid(row=0, column=0, padx=5, pady=5)
        else:
            self.frame_search.configure(
                label_text=f"Resultado da busca do termo <{self.term_entered}>")
            self.points_results_of_search = searchByTerm.search_by_term(
                self.term_entered, self.articles_directory_path)

            for i in range(len(self.points_results_of_search)):
                self.results_of_search.append(
                    (self.articles_titles_formatted[i+1][0], round(float(self.points_results_of_search[i]), 3)))

            self.results_of_search_sorted = [('Artigo', 'Pontuacao')] + sorted(
                self.results_of_search, key=lambda x: x[1], reverse=True)
            self.table_search.configure(values=self.results_of_search_sorted)

        self.controller_search = 2


if __name__ == "__main__":
    main_window = MainWindow()
    main_window.mainloop()
