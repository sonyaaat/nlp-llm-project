import config
import utils
from nltk.tokenize import word_tokenize
from typing import List
import nltk
import torch
import pickle
from langchain.docstore.document import Document as LangchainDocument
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import os


def create_vector_db(docs: List[LangchainDocument]):
    db_path: str = config.DB_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    if os.path.exists(db_path):
        print(f"Завантаження векторної бази даних з {db_path}")
        knowledge_vector_database = FAISS.load_local(
            db_path, 
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return knowledge_vector_database
    elif docs is not None:
        print("Створення нової векторної бази даних")
        knowledge_vector_database = FAISS.from_documents(
            docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
        knowledge_vector_database.save_local(db_path)
        print(f"Векторна база даних збережена в {db_path}")
        return knowledge_vector_database
    else:
      raise ValueError(
            """Documents are missing! 
            Please load the documents and set get_data=True in app.py."""
        )
    


def create_bm25(docs: List[LangchainDocument]):
    bm25_path: str = config.BM25_PATH
    if os.path.exists(bm25_path):
        print(f"Завантаження BM25 індексу з {bm25_path}")
        with open(bm25_path, "rb") as file:
            bm25 = pickle.load(file)
        return bm25
    elif docs is not None:
        print("Створення нового BM25 індексу")
        tokenized_docs = [word_tokenize(doc.page_content.lower()) for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        with open(bm25_path, "wb") as file:
            pickle.dump(bm25, file)
        print(f"BM25 індекс збережено в {bm25_path}")
        return bm25
    else:
      raise ValueError(
            """Documents are missing! 
            Please load the documents and set get_data=True in app.py."""
        )


def search(docs_processed, bm_25: BM25Okapi, vector_db: FAISS, query, top_k, use_bm25=True, use_semantic_search=True):
    if use_bm25 and use_semantic_search:
        bm25_retriever = BM25Retriever.from_documents(docs_processed)
        bm25_retriever.k = top_k
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )      
        result = ensemble_retriever.invoke(query)
        return result

    elif use_bm25:
        tokenized_query = word_tokenize(query.lower())
        result = bm_25.get_top_n(tokenized_query, [doc.page_content for doc in docs_processed], n=top_k)

    elif use_semantic_search:
        result = vector_db.similarity_search(query, k=top_k)
    else:
       result = []
    return result
    