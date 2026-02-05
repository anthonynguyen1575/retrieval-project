"""
Lab 3 FastAPI API.
@authors: Anthony Nguyen and Sebastian Silva
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 0.1.0+w26
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunk documents into smaller pieces for better retrieval."""

    def __init__(self, chunk_size: int = 300, overlap: int = 30):
        if overlap < 0:
            raise ValueError("Overlap must be non-negative.")
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str) -> list[dict]:
        """Split the given text into overlapping chunks."""

        words = text.split()
        if len(words) < self.chunk_size:
            return [{"id": f"{doc_id}_0", "text": text, "metadata": {"chunk": 0, "doc_id": doc_id}}]

        chunks, start, chunk_num = [], 0, 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_text = " ".join(words[start:end])

            chunks.append(
                {
                    "id": f"{doc_id}_{chunk_num}",
                    "text": chunk_text,
                    "metadata": {"chunk": chunk_num, "doc_id": doc_id},
                }
            )
            start = end - self.overlap
            chunk_num += 1
        return chunks


class DocumentLoader:
    """
    Load and parse documents from the file system.
    """

    def __init__(self, chunker: DocumentChunker = None):
        """Initialize loader with optional chunker"""
        self.chunker = chunker

    def load_documents(self, directory: str) -> list[dict]:
        """
        Load all text documents from a directory.

        Args:
            directory: Path to a directory containing documents

        Returns:
            List of documents, each with 'id', 'text' and 'metadata'.
        """
        documents = []
        path = Path(directory)
        if not path.is_dir() or not path.exists():
            raise ValueError(f"Directory '{directory}' does not exist.")
        for filepath in path.glob("*.txt"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    documents.append(
                        {"id": filepath.stem, "text": text, "metadata": {"filename": filepath.name}}
                    )
            except Exception as e:
                logger.warning(f"Warning: Failed to load document {filepath} : {e}")
        return documents

    def _load_text_file(self, filepath: Path) -> list[dict]:
        """Load a single txt file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                return []

            doc_id = filepath.stem
            metadata = {"filename": filepath.stem, "type": "txt"}

            if self.chunker:
                chunks = self.chunker.chunk_text(text, doc_id)
                # add filename to each chunk's metadata
                for chunk in chunks:
                    chunk["metadata"].update(metadata)
                return chunks
            else:
                return [{"id": doc_id, "text": text, "metadata": metadata}]

        except Exception as e:
            logger.warning(f"Warning: Failed to load document {filepath} : {e}")
            return []
