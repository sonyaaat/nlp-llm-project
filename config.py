import os


RAG_PROMPT = """
              You are an advanced Retrieval-Augmented Generation (RAG) Assistant.

              Your task is to answer user questions based only on the provided documents. Use the context from the documents to generate a response.

              **Guidelines:**
              1. **Always cite sources**: When information is derived from a document, reference it by citing the chunk number in square brackets, e.g., [Chunk 1], where relevant information is used.
              2. If the answer cannot be determined from the provided documents, state: "The answer cannot be determined from the provided documents."
              3. After each answer, provide a numbered list of the retrieved chunks.

              Please follow these instructions to generate accurate and well-cited answers based on the documents.
              """
LLM_ONLY_PROMPT = """You are an Assistant. If no documents are retrieved, answer the question based on general knowledge."""

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['GROQ_API_KEY'] = "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"
DB_PATH = "vector_database.faiss"
BM25_PATH = "bm25_index.pkl"
DOCUMENTS_PATH = "processed_documents.pkl"
EMBEDDING_MODEL_NAME = "thenlper/gte-small"

AVAILABLE_DATASET_CONFIGS = [
    '2024-11'
]
