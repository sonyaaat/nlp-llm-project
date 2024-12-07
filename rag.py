from typing import Optional, List, Tuple
from langchain.docstore.document import Document as LangchainDocument
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from ragatouille import RAGPretrainedModel
from litellm import completion
from rerankers import Reranker
import os
import retriver
import config


class RAGAnswerGenerator:
    def __init__(self, docs: List[LangchainDocument], bm25: BM25Okapi, knowledge_index: FAISS, reranker: Optional[RAGPretrainedModel] = None):
        self.bm25 = bm25
        self.knowledge_index = knowledge_index
        self.docs = docs
        self.reranker = reranker
        self.llm_key = os.environ['GROQ_API_KEY']

    def retrieve_documents(
        self,
        question: str,
        num_retrieved_docs: int,
        bm_25_flag: bool,
        semantic_flag: bool
    ) -> List[str]:
        print("=> Retrieving documents...")
        relevant_docs = []

        if bm_25_flag or semantic_flag:
            result = retriver.search(
                self.docs,
                self.bm25,
                self.knowledge_index,
                question,
                use_bm25=bm_25_flag,
                use_semantic_search=semantic_flag,
                top_k=num_retrieved_docs
            )
            if bm_25_flag and semantic_flag:
                relevant_docs = [doc.page_content for doc in result]
                return relevant_docs
            elif bm_25_flag:
                relevant_docs = result
                return relevant_docs
            elif semantic_flag:
                relevant_docs = [doc.page_content for doc in result]
                return relevant_docs
                

    def rerank_documents(self, question: str, documents: List[str], num_docs_final: int) -> List[str]:
        if self.reranker and documents:
            print("=> Reranking documents...")

            metadata = [{'source': f'doc_{i}'} for i in range(len(documents))]
            doc_ids = list(range(len(documents)))

            results = self.reranker.rank(query=question, docs=documents, doc_ids=doc_ids, metadata=metadata)
            final = results.top_k(num_docs_final)
            return [result.document.text for result in final]

        return documents

    def format_context(self, documents: List[str]) -> str:
        if not documents:
            return "No retrieved documents available."
        return "\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(documents)])

    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float,
    ) -> str:
        print("=> Generating answer...")
        if context.strip() == "No retrieved documents available.":
            response = completion(
                model="groq/llama3-8b-8192",
                messages=[
                    {"role": "system", "content": config.LLM_ONLY_PROMPT},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                api_key=self.llm_key,
                temperature=temperature
            )
        else:
            response = completion(
                model="groq/llama3-8b-8192",
                messages=[
                    {"role": "system", "content": config.RAG_PROMPT},
                    {"role": "user", "content": f""" Context: {context} Question: {question} """}
                ],
                api_key=self.llm_key,
                temperature=temperature
            )
        return response.get("choices", [{}])[0].get("message", {}).get("content", "No response content found")

    def answer(self, question: str, temperature: float, num_retrieved_docs: int = 30, num_docs_final: int = 5, bm_25_flag=True, semantic_flag=True) -> Tuple[str, List[str]]:
        relevant_docs = self.retrieve_documents(question, num_retrieved_docs, bm_25_flag, semantic_flag)
        print(len(relevant_docs))
        relevant_docs = self.rerank_documents(question, relevant_docs, num_docs_final)
        print(len(relevant_docs))
        context = self.format_context(relevant_docs)
        answer = self.generate_answer(question, context, temperature)
        document_list = [f"[{i + 1}] {doc}" for i, doc in enumerate(relevant_docs)] if relevant_docs else []
        return answer, document_list