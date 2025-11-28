import json
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma


from rag_pipeline.ingestion.embedder import LocalBGEEmbedder, ChromaBGEEmbedder
from rag_pipeline.ingestion.chromadb import ChromaDB
from rag_pipeline.eval.retrieval import (
    RetrievalExample,
    RetrievalEvaluator,
)


def build_vectorstore(
    persist_dir: str = "data/indexes/chroma/hotpot",
    collection_name: str = "hotpot_corpus",
    model_name: str = "BAAI/bge-base-en-v1.5",
) -> Chroma:
    """
    Recreates the Chroma vectorstore pointing at the existing persisted index.

    This MUST use the same persist_dir, collection_name, and embedding model
    as used when building the index.
    """
    # 1) Recreate the embedder (same config as the index)
    local_embedder = LocalBGEEmbedder(
        model_name=model_name,
        batch_size=128,
        device=None,  # "cpu" / "cuda" / "mps" if you want to force it
    )

    # 2) Initialize ChromaDB helper
    chroma_db = ChromaDB(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedder=local_embedder,
    )

    # This is the LangChain Chroma vectorstore, which implements similarity_search
    return chroma_db._db


def load_test_json(
    eval_path: str,
    max_examples: int | None = None,
) -> List[RetrievalExample]:
    """
    Load the first eval example from JSON and turn it into a RetrievalExample.

    """
    p = Path(eval_path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"No records found in eval file: {p}")

    examples: List[RetrievalExample] = []
    for rec in (data[:max_examples] if max_examples is not None else data):
        query = rec["question"]
        positive_titles: List[str] = rec.get("gold_doc_titles_in_corpus") or []
        if not positive_titles:
            continue  # skip items without ground truth

        examples.append(
            RetrievalExample(
                query=query,
                positive_ids=positive_titles,
                metadata={
                    "id": rec.get("id"),
                    "source": rec.get("source"),
                    "level": rec.get("level"),
                    "type": rec.get("type"),
                },
            )
        )
    if not examples:
        raise ValueError("No valid examples with positive_ids were found.")
    return examples


def main() -> None:
    eval_json_path = "data/corpus_test/dev_set/hotpot_qa_eval_dev_relaxed_100000.json"  # <-- change to your file
    persist_dir = "data/indexes/chroma/hotpot"
    collection_name = "hotpot_corpus"
    model_name = "BAAI/bge-base-en-v1.5"

    # Build / load the vectorstore from disk
    vectorstore = build_vectorstore(
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
    )

    # Load the first eval example
    examples = load_test_json(eval_json_path)
    # Run retrieval evaluation on that single example
    evaluator = RetrievalEvaluator(
        retriever=vectorstore,
        id_key="title",      # because we used gold_doc_titles_in_corpus
        ks=(1, 5, 10, 20),
    )
    metrics = evaluator.evaluate(examples, k_max=10)

    print("=== Retrieval eval on dev set ===")
    print(metrics.pretty())


if __name__ == "__main__":
    main()
