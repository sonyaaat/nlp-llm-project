import gradio as gr
import utils
from datasets import load_dataset, concatenate_datasets
from langchain.docstore.document import Document as LangchainDocument
from tqdm import tqdm
import pickle
from ragatouille import RAGPretrainedModel
import chunker
import retriver
from rerankers import Reranker
import rag
import nltk
import config
import os
import warnings
import sys
import logging

logging.getLogger("langchain").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class AnswerSystem:
    def __init__(self, rag_system) -> None:
        self.rag_system = rag_system
    
    def answer_generate(self, question, bm_25_flag, semantic_flag, temperature):
        answer, relevant_docs = self.rag_system.answer(
            question=question,
            temperature=temperature,
            bm_25_flag=bm_25_flag,
            semantic_flag=semantic_flag,
            num_retrieved_docs = 10,
            num_docs_final = 5
        )
        formatted_docs = "\n\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(relevant_docs)])
        return answer, formatted_docs


def run_app(rag_model):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # RealTimeData Monthly Collection - BBC News Documentation Assistant

            Welcome! This system is designed to help you explore and find insights from the RealTimeData Monthly Collection - BBC News dataset.  
            For example:  
            
            - *"What position does Josko Gvardiol play, and how much did Manchester City pay for him?"*  
            - *"Who are the presidents of the United States?"*
            - *"Which player won the Ballon d'Or in 2024?"*(BM25+combined search)
            - *"What notable achievements did Jack Draper accomplish during his breakout season on the ATP Tour in 2025?"*
            - *"How did Oscar Piastri and Lando Norris perform in sprint qualifying at the Sao Paulo Grand Prix, and what challenges did Max Verstappen face?"*

            """
        )

        # Поля вводу
        question_input = gr.Textbox(label="Enter your question:",
                                    placeholder="E.g., What position does Josko Gvardiol play, and how much did Manchester City pay for him?")
        bm25_checkbox = gr.Checkbox(label="Enable BM25-based retrieval", value=True)  # BM25 flag
        semantic_checkbox = gr.Checkbox(label="Enable Semantic Search", value=True)  # Semantic flag
        temperature_slider = gr.Slider(label="Response Temperature", minimum=0.1, maximum=1.0, value=0.2,
                                       step=0.1)  # Temperature

        # Кнопка пошуку
        search_button = gr.Button("Search")

        # Поля виводу
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=5)
        docs_output = gr.Textbox(label="Relevant Documents", interactive=False, lines=10)

        # Логіка пошуку
        system = AnswerSystem(rag_model)

        search_button.click(
            system.answer_generate,
            inputs=[question_input, bm25_checkbox, semantic_checkbox, temperature_slider],  # Всі параметри
            outputs=[answer_output, docs_output]
        )

    # Запуск додатку
    demo.launch(debug=True, share=True)


def get_rag_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')

    if os.path.exists(config.DOCUMENTS_PATH):
        print(f"Loading preprocessed documents from {config.DOCUMENTS_PATH}")
        with open(config.DOCUMENTS_PATH, "rb") as file:
            docs_processed = pickle.load(file)
    else:
        print("Processing documents...")
        datasets_list = [
            utils.align_features(load_dataset("RealTimeData/bbc_news_alltime", config)["train"])
            for config in tqdm(config.AVAILABLE_DATASET_CONFIGS)
        ]

        ds = concatenate_datasets(datasets_list)

        RAW_KNOWLEDGE_BASE = [
            LangchainDocument(
                page_content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "published_date": doc["published_date"],
                    "authors": doc["authors"],
                    "section": doc["section"],
                    "description": doc["description"],
                    "link": doc["link"]
                }
            )
            for doc in tqdm(ds)
        ]

        docs_processed = chunker.split_documents(512, RAW_KNOWLEDGE_BASE)

        print(f"Saving preprocessed documents to {config.DOCUMENTS_PATH}")
        with open(config.DOCUMENTS_PATH, "wb") as file:
            pickle.dump(docs_processed, file)

    return docs_processed


if __name__ == '__main__':
    docs_processed = get_rag_data()

    bm25 = retriver.create_bm25(docs_processed)

    KNOWLEDGE_VECTOR_DATABASE = retriver.create_vector_db(docs_processed)

    RERANKER = Reranker('cross-encoder')

    rag_generator = rag.RAGAnswerGenerator(
        docs=docs_processed,
        bm25=bm25,
        knowledge_index=KNOWLEDGE_VECTOR_DATABASE,
        reranker=RERANKER
    )

    run_app(rag_generator)