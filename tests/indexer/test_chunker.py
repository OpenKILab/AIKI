import pytest
from aiki.indexer.chunker import FixedSizeChunker

def test_large_text_chunk():
    chunker = FixedSizeChunker(chunk_size=1024)

    data = (
            "A" * 1024 +
            "B" * 1024
        )

    expected_chunks = ["A" * 1024, "B" * 1024]

    actual_chunks = chunker.chunk(data)

    assert actual_chunks == expected_chunks