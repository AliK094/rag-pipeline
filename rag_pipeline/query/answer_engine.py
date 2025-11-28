from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import textwrap

import ollama
from langchain_core.documents import Document

from rag_pipeline.query.query_engine import RAGQueryEngine


@dataclass
class RAGAnswerEngine:
    """
    End-to-end RAG engine on top of RAGQueryEngine + Ollama.

    - Uses RAGQueryEngine to retrieve top-k chunks from Chroma.
    - Calls a local Ollama model (e.g., llama3.1:8b) to generate an answer.
    """

    index_dir: str
    collection_name: str = "hotpot_corpus"

    # Retrieval config
    top_k: int = 5
    model_name_embed: str = "BAAI/bge-base-en-v1.5"
    device_embed: Optional[str] = None  # "cpu" | "cuda" | "mps"
    batch_size_embed: int = 128

    # Generation config (Ollama)
    llm_model: str = "llama3.1:8b"
    max_new_tokens: int = 256
    temperature: float = 0.2

    # Optional system prompt to steer behavior
    system_prompt: str = (
        "You are a helpful assistant answering questions based on the provided context. "
        "Answer concisely and do not hallucinate; if the answer is not in the context, "
        "say that you do not know."
    )

    _retriever: RAGQueryEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Create the retrieval engine
        self._retriever = RAGQueryEngine(
            index_dir=self.index_dir,
            collection_name=self.collection_name,
            model_name=self.model_name_embed,
            device=self.device_embed,
            batch_size=self.batch_size_embed,
            top_k_default=self.top_k,
        )

    # ------------------------------------------------------------------
    def _build_context(self, docs: List[Document]) -> str:
        """
        Concatenate retrieved docs into a context string for the LLM.

        Each chunk is prefixed with [i] Title and truncated to keep the prompt size reasonable.
        """
        parts: List[str] = []
        for i, d in enumerate(docs, start=1):
            md = d.metadata or {}
            title = md.get("title") or f"Document {i}"
            header = f"[{i}] {title}"

            snippet = textwrap.shorten(
                d.page_content,
                width=600,
                placeholder=" ..."
            )

            parts.append(f"{header}\n{snippet}")

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    def _docs_to_context_list(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Convert retrieved Documents to the dict structure used in the UI."""
        contexts: List[Dict[str, Any]] = []
        for rank, d in enumerate(docs, start=1):
            md = d.metadata or {}
            contexts.append(
                {
                    "rank": rank,
                    "score": None,  # can be filled if you ever switch to similarity_search_with_score
                    "content": d.page_content,
                    "title": md.get("title"),
                    "doc_id": md.get("doc_id"),
                    "source_path": md.get("source_path"),
                    "doc_type": md.get("doc_type"),
                    "chunk_id": md.get("chunk_id"),
                }
            )
        return contexts

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Public helper to just retrieve documents."""
        k = k or self.top_k
        return self._retriever.search(query, k=k)

    def retrieve_with_metadata(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Public helper that mirrors RAGQueryEngine.search_with_metadata."""
        return self._retriever.search_with_metadata(query, k=k)

    # ------------------------------------------------------------------
    def answer(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Run full RAG: retrieve top-k docs, call Ollama, return answer + contexts.

        Returns a dict:
            {
                "answer": str,
                "contexts": List[dict],  # same shape as retrieve_with_metadata
            }
        """
        query = (query or "").strip()
        if not query:
            return {"answer": "", "contexts": []}

        k = k or self.top_k

        # 1) Retrieve
        docs = self._retriever.search(query, k=k)
        if not docs:
            return {
                "answer": "I could not find any relevant context for this question in the index.",
                "contexts": [],
            }

        # 2) Build context string
        context_str = self._build_context(docs)

        # 3) Call Ollama (chat style)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Use only the context below to answer the question.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {query}\n\n"
                    "Answer:"
                ),
            },
        ]

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_new_tokens,
                },
            )
            answer_text = response["message"]["content"].strip()
        except Exception as e:
            # Fail safely
            answer_text = f"Error calling Ollama: {e}"

        contexts = self._docs_to_context_list(docs)
        return {"answer": answer_text, "contexts": contexts}
