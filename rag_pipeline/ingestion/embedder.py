from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from chromadb.api.types import EmbeddingFunction


@dataclass
class LocalBGEEmbedder:
    """
    Local embedding wrapper for RAG using BGE models via sentence-transformers.

    - Supports embedding texts, Documents, and queries.

    Parameters
    ----------
    model_name:
        Hugging Face / sentence-transformers model id.
        Common choices:
          - "BAAI/bge-small-en-v1.5"
          - "BAAI/bge-base-en-v1.5"
          - "BAAI/bge-large-en-v1.5"
    batch_size:
        Number of texts per forward pass.
    device:
        "cpu", "cuda", "mps", etc. If None, sentence-transformers chooses.
    normalize_embeddings:
        Whether to L2-normalize embeddings (recommended for cosine sim).
    use_query_instruction:
        If True, prepend query instruction for embed_query().
    query_instruction:
        Instruction text for queries (BGE default recommended).
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 128
    device: Optional[str] = None
    normalize_embeddings: bool = True
    use_query_instruction: bool = True
    query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )

    _model: SentenceTransformer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device or "cpu",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a sequence of raw texts (documents/passages).

        Returns
        -------
        List[List[float]]
            Embedding vectors corresponding to each input text, in order.
        """
        if not texts:
            return []

        vectors: List[List[float]] = []
        n = len(texts)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = texts[start:end]

            batch_vecs = self._model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=False,
                normalize_embeddings=self.normalize_embeddings,
            )

            for vec in batch_vecs:
                vectors.append(vec.tolist())

        return vectors
    
    def embed_documents(
        self,
        docs: Sequence[Union[Document, str]],
    ) -> List[List[float]]:
        """
        Embed a list of documents.

        Supports both:
          - Sequence[Document] (will use doc.page_content), and
          - Sequence[str]      (used by LangChain/Chroma internally).

        This makes the class compatible with:
          - manual calls: embed_documents(docs: List[Document])
          - Chroma / LangChain: embed_documents(texts: List[str])
        """
        if not docs:
            return []

        # If the first element is a Document, assume they all are.
        first = docs[0]
        if isinstance(first, Document):
            texts = [d.page_content for d in docs]  # type: ignore[arg-type]
        else:
            # Assume it's already a list of strings
            texts = [str(d) for d in docs]

        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.

        For BGE models, it is recommended to prepend a query instruction.
        """
        if self.use_query_instruction and self.query_instruction:
            to_encode = self.query_instruction + text
        else:
            to_encode = text

        vec = self._model.encode(
            to_encode,
            show_progress_bar=False,
            convert_to_numpy=False,
            normalize_embeddings=self.normalize_embeddings,
        )

        return vec.tolist()
    
class ChromaBGEEmbedder(EmbeddingFunction):
    """
    Thin wrapper around LocalBGEEmbedder for ChromaDB.
    Chroma will ONLY call __call__(input: List[str]).
    """

    def __init__(self, base_embedder: LocalBGEEmbedder):
        self.base = base_embedder

    def __call__(self, input: List[str]):
        """
        ChromaDB embedding call.

        Parameters
        ----------
        input : List[str]
            List of raw strings to embed.
        """
        # Chroma always sends list[str]
        return self.base.embed_texts(input)
