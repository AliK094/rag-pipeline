from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import chromadb
from chromadb.api.types import EmbeddingFunction
from rag_pipeline.ingestion.embedder import LocalBGEEmbedder, ChromaBGEEmbedder
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

PathLike = Union[str, Path]


@dataclass
class ChromaDB:
    """
    Helper for building and persisting a local ChromaDB collection
    from pre-chunked LangChain Document objects.

    """

    persist_dir: PathLike
    collection_name: str
    embedder: EmbeddingFunction
    reset_db: bool = False

    # HNSW / ANN configuration (sane defaults for cosine + BGE)
    hnsw_space: str = "cosine"          # l2 | cosine | ip
    hnsw_construction_ef: int = 200     # index build quality
    hnsw_M: int = 32                    # graph connectivity
    hnsw_search_ef: int = 40            # query-time accuracy vs speed
    hnsw_batch_size: int = 100          # BF → HNSW flush size
    hnsw_sync_threshold: int = 1000     # when to sync index to disk

    client: Optional[chromadb.PersistentClient] = field(init=False, default=None)
    _db: Optional[Chroma] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        # Normalize and ensure persistence directory exists
        persist_path = Path(self.persist_dir).expanduser().resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir = str(persist_path)

        if self.embedder is None:
            raise ValueError(
                "ChromaDB requires an existing embedder instance. "
                "You passed embedder=None."
            )

        # Initialize Chroma persistent client and vectorstore
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        if self.reset_db:
            try:
                self.client.delete_collection(self.collection_name)
                print(f"[ChromaDB] Deleted existing collection '{self.collection_name}'.")
            except Exception:
                # Collection may not exist yet; ignore
                pass

        chroma_embedder = ChromaBGEEmbedder(self.embedder)

        # HNSW configuration: passed at collection creation time
        collection_metadata = {
            "hnsw:space": self.hnsw_space,
            "hnsw:construction_ef": self.hnsw_construction_ef,
            "hnsw:M": self.hnsw_M,
            "hnsw:search_ef": self.hnsw_search_ef,
            "hnsw:batch_size": self.hnsw_batch_size,
            "hnsw:sync_threshold": self.hnsw_sync_threshold,
        }

        if isinstance(self.embedder, LocalBGEEmbedder):
            chroma_embedder = self.embedder
        elif isinstance(self.embedder, ChromaBGEEmbedder):
            chroma_embedder = self.embedder.base  # unwrap
        else:
            raise TypeError(
                f"Unsupported embedder type: {type(self.embedder).__name__}. "
                "Pass LocalBGEEmbedder or ChromaBGEEmbedder."
            )

        # LangChain Chroma wrapper – this will create/get the collection with HNSW config
        self._db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=chroma_embedder,
            persist_directory=self.persist_dir,
            collection_metadata=collection_metadata,
        )

    # ------------------------------------------------------------------
    def build(self, chunks: Sequence[Document], batch_size: int = 5000) -> Chroma:
        """
        Append Documents to the existing Chroma collection.

        This is APPEND behavior: it does NOT delete old vectors.

        Parameters
        ----------
        chunks:
            Sequence of LangChain Document objects.
            These must already be loaded, cleaned, and chunked

        Returns
        -------
        Chroma
            The constructed and persisted Chroma vector store.
        """
        if not chunks:
            raise ValueError("No chunks provided to ChromaDB.build().")

        # Ensure list materialization (e.g., if a generator was passed)
        chunks_list: List[Document] = list(chunks)
        total = len(chunks_list)

        print(
            f"[ChromaDB] Creating collection '{self.collection_name}' with "
            f"{total} chunks in: {self.persist_dir}"
        )

        for start in tqdm(range(0, total, batch_size), desc="Building vectorstore"):
            end = min(start + batch_size, total)
            batch = chunks_list[start:end]
            self._require_db().add_documents(batch)

        self._require_db().persist()
        print("[ChromaDB] Chroma collection successfully built and persisted.")
        return self._require_db()

    # ------------------------------------------------------------------
    def _require_db(self) -> Chroma:
        """Internal helper to assert the vectorstore has been initialized."""
        if self._db is None:
            raise RuntimeError(
                "Chroma vectorstore is not initialized. "
                "Did you forget to instantiate ChromaDB?"
            )
        return self._db
    
    def _delete_collection_if_exists(self) -> None:
        """Internal helper to delete the underlying Chroma collection."""
        assert self.client is not None
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # Collection may not exist yet; ignore
            pass