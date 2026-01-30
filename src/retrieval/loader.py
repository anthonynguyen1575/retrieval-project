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


class DocumentLoader:
    """
    Load and parse documents from the file system.
    """

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
