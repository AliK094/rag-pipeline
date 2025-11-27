from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunker:
    """
    Simple, production-ready character-based chunker for RAG.

    - Uses RecursiveCharacterTextSplitter.
    - Accepts LangChain Document objects.
    - Adds stable metadata (chunk_id, chunk_index, global_chunk_index).
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    add_metadata: bool = True

    separators: Optional[List[str]] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    def __post_init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    # ------------------------------------------------------------------
    def chunk_documents(self, docs: Sequence[Document]) -> List[Document]:
        """Split documents into character-based chunks."""
        if not docs:
            return []

        chunks = self._splitter.split_documents(list(docs))

        if self.add_metadata:
            self._enrich_metadata(chunks)

        return chunks

    # ------------------------------------------------------------------
    def _enrich_metadata(self, chunks: List[Document]) -> None:
        """
        Add metadata to each chunk:
        - chunk_index              (within document)
        - global_chunk_index       (within whole set)
        - chunk_id                 (unique stable id)
        """
        counters: Dict[str, int] = {}

        for global_idx, doc in enumerate(chunks):
            md = dict(doc.metadata or {})

            # Prefer doc_id or source_path-like stable identifiers
            parent_key = (
                md.get("doc_id")
                or md.get("title")
                or md.get("source_path")
                or md.get("file_name")
                or md.get("source")
                or f"doc_{global_idx}"
            )

            safe_key = str(parent_key).replace(":", "_").replace("/", "_")

            local_idx = counters.get(safe_key, 0)
            counters[safe_key] = local_idx + 1

            md["chunk_index"] = local_idx
            md["global_chunk_index"] = global_idx
            md["chunk_id"] = f"{safe_key}::chunk-{local_idx}"

            doc.metadata = md
