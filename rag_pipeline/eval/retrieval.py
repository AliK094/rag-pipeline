from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from langchain_core.documents import Document

# ----------------------------------------------------------------------
class SupportsRetrieve(Protocol):
    """
    Minimal protocol for retrieval backends.

    This is satisfied by:
      - LangChain vector stores (e.g., Chroma) that implement similarity_search
      - LangChain retrievers that implement get_relevant_documents
    """

    def similarity_search(self, query: str, k: int) -> List[Document]:
        ...

    def get_relevant_documents(self, query: str) -> List[Document]:
        ...


@dataclass
class RetrievalExample:
    """
    Single evaluation example for retrieval.

    Attributes
    ----------
    query:
        User query / question string.
    positive_ids:
        Iterable of IDs that are considered relevant for this query.
        Typically these correspond to metadata fields like `doc_id` or `title`.
    metadata:
        Optional extra info about the example (e.g., dataset id, split).
    """

    query: str
    positive_ids: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """
    Aggregated retrieval metrics over a set of examples.
    """

    hit_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float

    def pretty(self) -> str:
        """Human-readable string summary."""
        lines = ["RetrievalMetrics:"]
        for k in sorted(self.hit_at_k):
            lines.append(f"  hit@{k}:    {self.hit_at_k[k]:.4f}")
        for k in sorted(self.recall_at_k):
            lines.append(f"  recall@{k}: {self.recall_at_k[k]:.4f}")
        lines.append(f"  MRR:        {self.mrr:.4f}")
        return "\n".join(lines)


class RetrievalEvaluator:
    """
    Systematic evaluator for retrieval metrics over a set of queries.

    It assumes:
      - You have a retriever / vector store (e.g., Chroma) that can return
        a ranked list of Documents for a query.
      - Each Document has some metadata key (e.g., 'doc_id' or 'title')
        that you can use to compare against ground-truth `positive_ids`.

    It computes:
      - hit@k      : fraction of queries where at least one relevant ID is in top-k
      - recall@k   : average recall over queries at cutoff k
      - MRR        : mean reciprocal rank of the first relevant hit
    """

    def __init__(
        self,
        retriever: SupportsRetrieve,
        id_key: str = "doc_id",
        ks: Sequence[int] = (1, 5, 10),
    ) -> None:
        """
        Parameters
        ----------
        retriever:
            Any object that implements `similarity_search(query, k)` or
            `get_relevant_documents(query)`.
        id_key:
            Metadata key used to identify parent documents, e.g. 'doc_id' or 'title'.
        ks:
            The list of k values at which to compute hit@k and recall@k.
        """
        self.retriever = retriever
        self.id_key = id_key
        self.ks = sorted(set(ks))

    # ------------------------------------------------------------------
    def evaluate(
        self,
        examples: Sequence[RetrievalExample],
        k_max: Optional[int] = None,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality over a set of examples.

        Parameters
        ----------
        examples:
            Iterable of RetrievalExample objects.
        k_max:
            Maximum rank depth to retrieve for each query. If None, uses max(ks).

        Returns
        -------
        RetrievalMetrics
            Aggregated hit@k, recall@k and MRR.
        """
        if not examples:
            raise ValueError("No examples provided to RetrievalEvaluator.evaluate().")

        k_max = k_max or max(self.ks)

        # Counters
        hit_counts: Dict[int, int] = {k: 0 for k in self.ks}
        recall_sums: Dict[int, float] = {k: 0.0 for k in self.ks}
        mrr_sum = 0.0
        n_valid = 0

        for ex in examples:
            positives = [str(pid) for pid in ex.positive_ids if pid is not None]
            if not positives:
                # Skip examples without ground truth
                continue

            n_valid += 1
            retrieved_docs = self._retrieve(ex.query, k_max)
            retrieved_ids = self._to_parent_ids(retrieved_docs)

            # First relevant hit rank
            first_hit_rank: Optional[int] = None
            for rank, rid in enumerate(retrieved_ids, start=1):
                if rid in positives:
                    first_hit_rank = rank
                    break

            if first_hit_rank is not None:
                mrr_sum += 1.0 / first_hit_rank

            pos_set = set(positives)

            for k in self.ks:
                top_ids = set(retrieved_ids[:k])

                # hit@k: any intersection?
                if pos_set & top_ids:
                    hit_counts[k] += 1

                # recall@k: fraction of relevant docs retrieved in top-k
                recall_sums[k] += len(pos_set & top_ids) / max(1, len(pos_set))

        if n_valid == 0:
            raise ValueError("No valid examples with positive_ids were provided.")

        hit_at_k = {k: hit_counts[k] / n_valid for k in self.ks}
        recall_at_k = {k: recall_sums[k] / n_valid for k in self.ks}
        mrr = mrr_sum / n_valid

        return RetrievalMetrics(hit_at_k=hit_at_k, recall_at_k=recall_at_k, mrr=mrr)

    # Internal helpers
    def _retrieve(self, query: str, k: int) -> List[Document]:
        """Call the underlying retriever in a flexible way."""
        if hasattr(self.retriever, "similarity_search"):
            return self.retriever.similarity_search(query, k=k)
        if hasattr(self.retriever, "get_relevant_documents"):
            docs = self.retriever.get_relevant_documents(query)
            return docs[:k]
        raise TypeError(
            "Retriever must implement `similarity_search` or `get_relevant_documents`."
        )

    def _to_parent_ids(self, docs: Sequence[Document]) -> List[str]:
        """
        Convert retrieved Documents to parent IDs using the chosen metadata key.

        If id_key is missing, falls back to other reasonable metadata fields.
        """
        ids: List[str] = []
        for d in docs:
            md = d.metadata or {}
            # Try the configured key first, then some common fallbacks
            raw_id = (
                md.get(self.id_key)
                or md.get("title")
                or md.get("source_path")
                or md.get("chunk_id")
                or "UNKNOWN"
            )
            ids.append(str(raw_id))
        return ids
