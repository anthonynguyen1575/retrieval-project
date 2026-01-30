"""
Lab 3 FastAPI API.
@authors: Anthony Nguyen and Sebastian Silva
Seattle University, ARIN 5360
@see: https://catalog.seattleu.edu/preview_course_nopop.php?catoid=55&coid
=190380
@version: 0.1.0+w26
"""

import pytest

from retrieval.loader import DocumentLoader


def test_loader_loads_documents(tmp_path):
    """
    Test loading documents from a directory.
    """
    # Create some test files
    (tmp_path / "file1.txt").write_text("This is a test file.")
    (tmp_path / "file2.txt").write_text("This is another test file.")

    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))

    assert len(documents) == 2
    assert all("id" in doc for doc in documents)
    assert all("text" in doc for doc in documents)
    assert all("metadata" in doc for doc in documents)


def test_loader_skips_empty_files(tmp_path):
    """
    Test that empty files are skipped.
    """

    (tmp_path / "not_empty.txt").write_text("This is a test file.")
    (tmp_path / "empty.txt").write_text("")

    loader = DocumentLoader()
    documents = loader.load_documents(str(tmp_path))

    assert len(documents) == 1
    assert documents[0]["text"] == "This is a test file."
    assert documents[0]["metadata"]["filename"] == "not_empty.txt"


def test_loader_skips_nonexistent_directory(tmp_path):
    """
    Test that non-existent directories are skipped.
    """

    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Directory 'garbage' does not exist"):
        loader.load_documents("garbage")
