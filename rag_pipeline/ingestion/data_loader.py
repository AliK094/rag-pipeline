import json
from pathlib import Path
from typing import List, Union, Dict, Any, Sequence

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader

PathLike = Union[str, Path]


def _resolve_path(path: PathLike) -> Path:
    """Resolve a path and ensure it exists."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p


def load_json_corpus(path: PathLike) -> List[Document]:
    """
    Load a JSON corpus file and convert it into a List[Document].

    Expected JSON format:
    [
      {"id": ..., "title": ..., "text": ..., "source": ..., "metadata": {...}},
      ...
    ]
    """
    p = _resolve_path(path)
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
        metadata["source_path"] = str(p)
        metadata["doc_type"] = "json"

        docs.append(
            Document(
                page_content=text,
                metadata=metadata,
            )
        )

    return docs


def _attach_common_metadata(
    docs: List[Document],
    source_path: Path,
    doc_type: str,
) -> List[Document]:
    """Attach standard metadata fields to each document."""
    for d in docs:
        # Do not blow away existing metadata
        d.metadata = {
            **(d.metadata or {}),
            "source_path": str(source_path),
            "doc_type": doc_type,
        }
    return docs


def load_pdf_file(path: PathLike) -> List[Document]:
    """Load a PDF as Documents using PyPDFLoader."""
    p = _resolve_path(path)
    loader = PyPDFLoader(str(p))
    docs = loader.load()
    return _attach_common_metadata(docs, p, doc_type="pdf")


def load_txt_file(path: PathLike) -> List[Document]:
    """Load a TXT file as Documents using TextLoader."""
    p = _resolve_path(path)
    loader = TextLoader(str(p), encoding="utf-8")
    docs = loader.load()
    return _attach_common_metadata(docs, p, doc_type="txt")


def load_html_file(path: PathLike) -> List[Document]:
    """Load an HTML file as Documents using BSHTMLLoader."""
    p = _resolve_path(path)
    loader = BSHTMLLoader(str(p))
    docs = loader.load()
    return _attach_common_metadata(docs, p, doc_type="html")


def load_unstructured_file(path: PathLike) -> List[Document]:
    """
    Fallback loader for other formats using UnstructuredLoader
    (DOCX, PPTX, etc.).
    """
    p = _resolve_path(path)
    loader = UnstructuredLoader(
        file_path=str(p),
        partition_via_api=False,
        languages=["eng"],
    )
    docs = loader.load()
    return _attach_common_metadata(docs, p, doc_type="unstructured")


def _load_file_by_extension(path: Path) -> List[Document]:
    """Load a single file based on its extension."""
    ext = path.suffix.lower()

    if ext == ".json":
        return load_json_corpus(path)
    elif ext == ".pdf":
        return load_pdf_file(path)
    elif ext == ".txt":
        return load_txt_file(path)
    elif ext in {".html", ".htm"}:
        return load_html_file(path)
    else:
        return load_unstructured_file(path)



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
    p = _resolve_path(path)
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
            # In a real package, you'd likely use logging instead of print
            print(f"[WARN] Skipping {file_path}: {e}")

    return all_docs


def load_corpus(
    paths: Union[PathLike, Sequence[PathLike]],
    recursive: bool = True,
) -> List[Document]:
    """
    General-purpose corpus loader.

    Supports:
      - A single directory path  → walks and loads all files inside.
      - A single file path       → loads that file only.
      - A list/tuple of paths    → each can be a file or a directory.

    Examples
    --------
    Load a directory:

        docs = load_corpus("data/corpus")

    Load a single PDF:

        docs = load_corpus("data/docs/paper.pdf")

    Load a mix of files and directories:

        docs = load_corpus([
            "data/corpus",
            "notes/extra.txt",
            "report.pdf",
        ])
    """
    # Normalize to a sequence
    if isinstance(paths, (str, Path)):
        paths_seq: Sequence[PathLike] = [paths]
    else:
        paths_seq = paths

    all_docs: List[Document] = []

    for p in paths_seq:
        resolved = _resolve_path(p)
        if resolved.is_dir():
            docs = load_corpus_directory(resolved, recursive=recursive)
        else:
            docs = _load_file_by_extension(resolved)
        all_docs.extend(docs)

    return all_docs
