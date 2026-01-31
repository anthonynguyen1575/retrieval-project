"""
Document retrieval system.

@author: Sebastian Silva & Anthony Nguyen
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 1.0.0+w26
"""

from .embeddings import DocumentEmbedder
from .loader import DocumentLoader
from .store import VectorStore


class DocumentRetriever:
    """High-level interface for document retrieval."""

    def __init__(self):
        """Initialize retriever with default components."""
        self.loader = DocumentLoader()
        self.store = VectorStore(DocumentEmbedder())
        self._indexed = False  # flag to indicate we've done some indexing

    def index_documents(self, directory: str):
        """
        Load and index documents from a directory.

        Args:
            directory: Path to the directory containing documents

        Returns:
            Number of documents indexed
        """
        # use our components to load and add documents
        before = self.document_count
        documents = self.loader.load_documents(directory)
        self.store.add_documents(documents)
        self._indexed = True
        return self.document_count - before

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Search for documents relevant to the query.

        Args:
            query: Search query text
            n_results: Number of results to return

        Returns:
            List of result dicts with document information
        """
        # use our vector store to query
        if not self._indexed:
            raise ValueError("No documents indexed. Call index_documents() first.")

        return self.store.search(query, n_results)

    # define a property that fetches the number of indexed documents
    @property
    def document_count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()
