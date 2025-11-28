import argparse
from pathlib import Path
from typing import Optional

from rag_pipeline.ingestion.data_loader import load_corpus
from rag_pipeline.ingestion.chunker import DocumentChunker
from rag_pipeline.ingestion.embedder import LocalBGEEmbedder
from rag_pipeline.ingestion.chromadb import ChromaDB

from tqdm import tqdm


def build_index(
    corpus_dir: str,
    index_dir: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    model_name: str = "BAAI/bge-base-en-v1.5",
    device: Optional[str] = None,
    batch_size: int = 128,
    collection_name: str = "hotpot_corpus",
    reset_db: bool = False,
) -> None:

    corpus_path = Path(corpus_dir).expanduser().resolve()
    index_path = Path(index_dir).expanduser().resolve()
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"[build_index] Loading corpus from: {corpus_path}")
    docs = load_corpus(corpus_path)
    print(f"[build_index] Loaded {len(docs)} raw documents")

    # --- Chunk documents ---
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for doc in tqdm(docs, desc="Chunking documents"):
        chunks.extend(chunker.chunk_documents([doc]))
        
    print(f"[build_index] Created {len(chunks)} chunks")

    # --- Initialize embedder ---
    embedder = LocalBGEEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    print(f"[build_index] Using embedder model: {model_name} (device={device})")

    # --- Build ChromaDB ---
    chroma_db = ChromaDB(
        persist_dir=index_path,
        collection_name=collection_name,
        embedder=embedder,
        reset_db=reset_db,
    )

    coll = chroma_db.client.get_collection(name=chroma_db.collection_name)
    print("Count before:", coll.count())

    print(f"[build_index] Building ChromaDB collection '{collection_name}'...")
    chroma_db.build(chunks)
    print("[build_index] Done.")

    coll = chroma_db.client.get_collection(name=chroma_db.collection_name)
    print("Count after:", coll.count())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index for RAG corpus.")
    parser.add_argument("--corpus_dir", type=str, required=True)
    parser.add_argument("--index_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", type=str, default="mps")  # default for mac!
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--collection_name", type=str, default="hotpot_corpus")
    parser.add_argument("--reset_db", action="store_true", help="Whether to reset the existing DB.")

    args = parser.parse_args()

    build_index(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        collection_name=args.collection_name,
        reset_db=args.reset_db,
    )
