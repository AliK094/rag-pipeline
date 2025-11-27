from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

from rag_pipeline.ingestion.embedder import LocalBGEEmbedder
from tqdm import tqdm

PathLike = Union[str, Path]


@dataclass
class ChromaDB:
    """
    Simple helper for building and persisting a local ChromaDB collection
    from pre-chunked LangChain Document objects.

    This class does only the final ingestion step:
        - Take already-prepared chunks (Documents)
        - Use an existing embedder to generate embeddings
        - Write the resulting collection to a persistent Chroma directory

    It does NOT:
        - Load raw documents
        - Clean/transform documents
        - Chunk documents
        - Deduplicate documents
        - Create its own embedder

    All preprocessing steps must be handled upstream.
    """
    persist_dir: PathLike
    collection_name: str
    embedder: LocalBGEEmbedder

    client: Optional[chromadb.PersistentClient] = field(init=False, default=None)
    _db: Optional[Chroma] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # Normalize and ensure persistence directory exists
        persist_path = Path(self.persist_dir).expanduser().resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(persist_path)

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        if self.embedder is None:
            raise ValueError(
                "ChromaDB requires an existing embedder instance. "
                "You passed embedder=None."
            )
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self._db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedder,
            persist_directory=self.persist_dir,
        )

    # ------------------------------------------------------------------
    def build(self, chunks: Sequence[Document]) -> Chroma:
        """
        Append Documents to the existing Chroma collection.

        This is APPEND behavior: it does NOT delete old vectors.

        Parameters
        ----------
        chunks:
            Sequence of LangChain Document objects.
            These must already be loaded, cleaned, chunked, and optionally deduplicated
            before being passed to this method.

        Returns
        -------
        Chroma
            The constructed and persisted Chroma vector store.
        """
        chunks = list(tqdm(chunks, desc="Building vectorstore"))

        if not chunks:
            raise ValueError("No chunks provided to ChromaDB.build().")

        print(
            f"[ChromaDB] Creating collection '{self.collection_name}' with "
            f"{len(chunks)} chunks in: {self.persist_dir}"
        )

        self._db.add_documents(chunks)

        self._db.persist()

        print("[ChromaDB] Chroma collection successfully built and persisted.")
        return self._db
