# RAG Pipeline

This repository provides a lightweight retrieval-augmented generation (RAG) pipeline focused on offline corpus ingestion and fast local retrieval with ChromaDB and BAAI BGE embeddings. It also includes a Streamlit app powered by a local Ollama model for interactive question answering on top of your index.

## Features
- Ingests corpora from JSON, PDF, TXT, HTML, and other common document types using LangChain loaders.
- Splits documents into overlapping chunks with rich metadata to enable traceability.
- Generates embeddings locally via BGE models from `sentence-transformers`.
- Persists vectors to a ChromaDB collection for rapid local querying.
- Includes a minimal retrieval script to validate the built index.

## Project Structure
- `rag_pipeline/ingestion/data_loader.py`: Loads corpus files or directories into `langchain` `Document` objects with standardized metadata.
- `rag_pipeline/ingestion/chunker.py`: Splits documents into stable, metadata-rich chunks using `RecursiveCharacterTextSplitter`.
- `rag_pipeline/ingestion/embedder.py`: Wraps local BGE models for embedding documents and queries; provides a Chroma-compatible wrapper.
- `rag_pipeline/ingestion/chromadb.py`: Writes prepared chunks into a persistent ChromaDB collection.
- `rag_pipeline/ingestion/build_index.py`: CLI entry point to run the full ingestion pipeline end-to-end.
- `rag_pipeline/eval/quick_retrieval.py`: Example script to query an existing Chroma collection.
- `rag_pipeline/eval/evaluation.py`: Simple retrieval evaluation over JSON QA datasets.
- `apps/rag_app.py`: Streamlit app that exposes the RAG pipeline over a web UI, using Ollama as the LLM.

## Setup
1. **Python version**: 3.11+.
2. **Ollama** (for the RAG app):
   - Install Ollama from the official site.
   - Pull a compatible model, e.g.:
     ```bash
     ollama pull llama3.1:8b
     ```
3. **Install dependencies** using your preferred tool:
   ```bash
   pip install -r requirements.txt
   # or
   pip install -e .
   ```
4. **Model download**: The default embedder uses `BAAI/bge-base-en-v1.5`. Ensure your environment can download Hugging Face models (or pre-download them).

The `requirements.txt` file is aligned with the `pyproject.toml` dependencies and includes:
- ChromaDB + LangChain components for retrieval and document handling.
- `sentence-transformers` for BGE embeddings.
- `unstructured[all-docs]` and related libraries for rich document loading.
- Optional extras for notebooks (e.g., `datasets`, `ipykernel`).

## Building an Index
Run the ingestion CLI to load a corpus, chunk it, embed the chunks, and persist a ChromaDB collection.
```bash
python -m rag_pipeline.ingestion.build_index \
  --corpus_dir /path/to/corpus \
  --index_dir data/indexes/chroma/hotpot \
  --chunk_size 512 \
  --chunk_overlap 64 \
  --model_name BAAI/bge-base-en-v1.5 \
  --device cpu \
  --batch_size 128 \
  --collection_name hotpot_corpus
```
Key options:
- `--corpus_dir`: Directory containing your documents (JSON, PDF, TXT, HTML, etc.).
- `--index_dir`: Directory where the ChromaDB collection will be persisted/created.
- `--device`: Set to `cuda`, `mps`, or `cpu` depending on hardware.
- `--chunk_size` & `--chunk_overlap`: Control chunking granularity for retrieval quality.

## Quick Retrieval Demo
Once an index exists, run the demo script to issue a sample query against the persisted collection:
```bash
python -m rag_pipeline.eval.quick_retrieval
```
The script:
- Initializes the same BGE embedder used for indexing.
- Opens the ChromaDB collection at `data/indexes/chroma/hotpot`.
- Prints the top results (ids, titles, distances, and text snippets) for a sample query.

## Streamlit RAG App
After you have built an index and have Ollama running locally, you can launch the Streamlit UI:

```bash
streamlit run apps/rag_app.py
```

The app:
- Uses `RAGAnswerEngine` (`rag_pipeline/query/answer_engine.py`) to retrieve top‑k chunks from Chroma and call an Ollama model (default: `llama3.1:8b`).
- Lets you inspect the LLM answer, the retrieved chunks, and their metadata (doc id, chunk id, source path).

By default it expects the index at:
- `data/indexes/chroma/hotpot`
- collection name: `hotpot_corpus`

These defaults can be adjusted in `apps/rag_app.py` if your index lives elsewhere or you want to change the model/device.

## Retrieval Evaluation
For simple retrieval metrics on a QA dataset (e.g. HotpotQA):

1. Place your eval JSON file under `data/corpus_test/...` (or update the path in `rag_pipeline/eval/evaluation.py`).
2. Run:
   ```bash
   python -m rag_pipeline.eval.evaluation
   ```
3. The script will:
   - Recreate the Chroma vectorstore pointing at your existing index.
   - Load the eval JSON into `RetrievalExample` objects.
   - Print hit@k, recall@k, and MRR metrics.

## Notebooks
The `notebook/` directory contains example Jupyter notebooks for dataset preparation and document exploration. They use `datasets`, `pandas`, and `ipykernel`; these are included in `requirements.txt` so you can run them after installing the dependencies.

## Notes
- Hidden files are skipped during ingestion, and the pipeline prints progress every few thousand documents.
- The ingestion step appends to an existing Chroma collection; remove the directory first if you need a clean rebuild.
- You can adjust separators or metadata enrichment in `DocumentChunker` to fit your corpus.
