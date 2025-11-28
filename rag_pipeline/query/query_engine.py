from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from rag_pipeline.ingestion.embedder import LocalBGEEmbedder, ChromaBGEEmbedder
from rag_pipeline.ingestion.chromadb import ChromaDB


@dataclass
class RAGQueryEngine:
    """
    Thin query layer on top of a persisted ChromaDB index.
    """

    index_dir: str
    collection_name: str = "hotpot_corpus"
    model_name: str = "BAAI/bge-base-en-v1.5"
    device: Optional[str] = None       # "cpu" | "cuda" | "mps"
    batch_size: int = 128
    top_k_default: int = 5

    def __post_init__(self) -> None:
        # Normalize paths
        index_path = Path(self.index_dir).expanduser().resolve()
        if not index_path.exists():
            raise FileNotFoundError(f"Index directory does not exist: {index_path}")
        self.index_dir = str(index_path)

        # Recreate the embedder (must match what you used during indexing)
        self._embedder = LocalBGEEmbedder(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
        )

        # We reuse your ChromaDB helper BUT with reset_db=False (read / append mode)
        self._chroma_db = ChromaDB(
            persist_dir=self.index_dir,
            collection_name=self.collection_name,
            embedder=self._embedder,
        )
        # LangChain vectorstore (implements `similarity_search`)
        self._vectorstore = self._chroma_db._db

    # ------------------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Run a similarity search over the vectorstore.

        Parameters
        ----------
        query:
            User query text.
        k:
            Number of top results to return (defaults to top_k_default).
       
        Returns
        -------
        List[Document]
        """
        if not query or not query.strip():
            return []

        k = k or self.top_k_default
        docs: List[Document] = self._vectorstore.similarity_search(query, k=k)
        return docs

    def search_with_metadata(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Same as search() but returns a list of dicts with metadata.
        Useful for UI, debugging, evaluation logs, etc.
        """
        docs = self.search(query, k=k)
        results = []
        for rank, d in enumerate(docs, start=1):
            md = d.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "score": None,
                    "content": d.page_content,
                    "title": md.get("title"),
                    "doc_id": md.get("doc_id"),
                    "source_path": md.get("source_path"),
                    "doc_type": md.get("doc_type"),
                    "chunk_id": md.get("chunk_id"),
                }
            )
        return results
