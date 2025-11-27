import json
from pathlib import Path
from typing import List, Union, Dict, Any

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

PathLike = Union[str, Path]


def load_json_corpus(path: PathLike) -> List[Document]:
    """
    Load a JSON corpus file and convert it into a List[Document].

    Expected JSON format:
    [
      {"id": ..., "title": ..., "text": ..., "source": ..., "metadata": {...}},
      ...
    ]
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Corpus file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"JSON corpus must be a list of objects: {p}")

    docs: List[Document] = []
    for obj in raw:
        text = obj.get("text", "")
        if not text:
            continue

        metadata: Dict[str, Any] = dict(obj.get("metadata", {}) or {})
        metadata["doc_id"] = obj.get("id")
        metadata["title"] = obj.get("title")
        metadata["source"] = obj.get("source", "hotpot_qa")
        metadata["corpus_file"] = str(p)

        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs

def load_pdf_file(path: PathLike) -> List[Document]:
    """
    Load a PDF as Documents using PyPDFLoader.
    """
    p = Path(path).expanduser().resolve()
    loader = PyPDFLoader(str(p))
    docs = loader.load()
    return docs


def load_txt_file(path: PathLike) -> List[Document]:
    """
    Load a TXT file as Documents.
    """
    p = Path(path).expanduser().resolve()
    loader = TextLoader(str(p), encoding="utf-8")
    docs = loader.load()
    return docs


def load_html_file(path: PathLike) -> List[Document]:
    """
    Load an HTML file as Documents.
    """
    p = Path(path).expanduser().resolve()
    loader = BSHTMLLoader(str(p))
    docs = loader.load()
    return docs

def load_unstructured_file(path: PathLike) -> List[Document]:
    """
    Fallback loader for other formats using UnstructuredLoader
    (DOCX, PPTX, etc.).
    """
    p = Path(path).expanduser().resolve()
    loader = UnstructuredLoader(
        file_path=str(p),
        partition_via_api=False,
        languages=["eng"],
    )
    docs = loader.load()  # List[Document]
    # return _normalize_and_filter_docs(docs, p)
    return docs

def load_corpus_directory(path: PathLike, recursive: bool = True) -> List[Document]:
    """
    Load an entire corpus directory.

    Rules:
      - *.json  → load_json_corpus()
      - *.pdf   → load_pdf_file()
      - *.txt   → load_txt_file()
      - *.html  → load_html_file()
      - others  → load_unstructured_file()
    """
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    all_docs: List[Document] = []
    iterator = p.rglob("*") if recursive else p.glob("*")

    num_docs = 0

    for file_path in iterator:
        if not file_path.is_file():
            continue

        # Skip hidden files (.DS_Store, etc.)
        if file_path.name.startswith("."):
            continue

        ext = file_path.suffix.lower()

        try:
            if ext == ".json":
                docs = load_json_corpus(file_path)
            elif ext == ".pdf":
                docs = load_pdf_file(file_path)
            elif ext == ".txt":
                docs = load_txt_file(file_path)
            elif ext in {".html", ".htm"}:
                docs = load_html_file(file_path)
            else:
                docs = load_unstructured_file(file_path)

            all_docs.extend(docs)
            num_docs += len(docs)

            if num_docs and num_docs % 5000 == 0:
                print(f"[data_loader] Processed {num_docs} docs so far...")

        except Exception as e:
            print(f"[WARN] Skipping {file_path}: {e}")

    return all_docs
