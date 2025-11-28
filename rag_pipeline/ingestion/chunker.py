from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentChunker:
    """
    Character-based chunker for RAG.

    - Uses RecursiveCharacterTextSplitter.
    - Accepts LangChain Document objects.
    - Optionally enriches chunks with metadata:

        chunk_index           (index within parent document)
        global_chunk_index    (index within the full chunk list)
        chunk_id              (stable string: "<parent_key>::chunk-<idx>")
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    add_metadata: bool = True

    # Priority of metadata fields used to build a stable parent key
    id_priority: Sequence[str] = field(
        default_factory=lambda: ("doc_id", "title", "source_path", "file_name", "source")
    )

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

        # LangChain splitters generally expect a list
        chunks = self._splitter.split_documents(list(docs))

        if self.add_metadata:
            self._enrich_metadata(chunks)

        return chunks

    # ------------------------------------------------------------------
    def _enrich_metadata(self, chunks: List[Document]) -> None:
        """
        Add metadata to each chunk in-place:
        - chunk_index              (within document)
        - global_chunk_index       (within whole set)
        - chunk_id                 (unique stable id)
        """
        counters: Dict[str, int] = defaultdict(int)

        for global_idx, doc in enumerate(chunks):
            md = dict(doc.metadata or {})

            # Derive a stable parent key
            parent_key: Any = None
            for key in self.id_priority:
                if key in md and md[key]:
                    parent_key = md[key]
                    break

            if parent_key is None:
                parent_key = f"doc_{global_idx}"

            safe_key = str(parent_key).replace(":", "_").replace("/", "_")

            local_idx = counters[safe_key]
            counters[safe_key] += 1

            md["chunk_index"] = local_idx
            md["global_chunk_index"] = global_idx
            md["chunk_id"] = f"{safe_key}::chunk-{local_idx}"

            doc.metadata = md
