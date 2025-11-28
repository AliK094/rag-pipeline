from pathlib import Path
import chromadb

from rag_pipeline.ingestion.embedder import LocalBGEEmbedder, ChromaBGEEmbedder


def main():
    persist_dir = Path("data/indexes/chroma/hotpot")
    collection_name = "hotpot_corpus"

    # 1) Init embedder (adjust model_name / device if needed)
    base = LocalBGEEmbedder(
        model_name="BAAI/bge-base-en-v1.5",
        device="mps"
    )
    embedder = ChromaBGEEmbedder(base)

    # 2) Load Chroma persistent collection
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedder,
    )

    # 3) Test query
    query = "Who won the 1971 Rose Bowl?"
    k = 5

    print(f"\nQuery: {query!r}")
    print(f"Top-{k} results:\n")

    result = collection.query(
        query_texts=[query],
        n_results=k,
    )

    # 4) Pretty-print first result set
    ids = result["ids"][0]
    docs = result["documents"][0]
    metadatas = result["metadatas"][0]

    # BUG FIX: this line had a bracket error
    distances = result.get("distances", [[None] * len(ids)])[0]

    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metadatas, distances), start=1
    ):
        title = meta.get("title") or meta.get("source") or "<no title>"
        print(f"Rank {rank}")
        print(f"  id        : {doc_id}")
        print(f"  title     : {title}")
        print(f"  distance  : {dist}")
        print(f"  text[:500]: {doc[:500].replace('\\n', ' ')}")
        print("-" * 60)

    print(result["metadatas"][0][0])

if __name__ == "__main__":
    main()
