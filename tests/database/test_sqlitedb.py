# test_sqlite_db.py
import pytest
from tortoise import Tortoise
from bson import ObjectId
from aiki.database.sqlite import SQLiteDB
from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData

@pytest.fixture(scope="module")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)     # Set the new event loop as the current event loop
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def db():
    db_url = "sqlite://:memory:"
    db_instance = SQLiteDB(db_url)  # Renamed to db_instance for clarity
    return db_instance

@pytest.mark.asyncio
async def test_mset_and_mget_image(db):
    # Create a sample ImageModalityData
    image_data = ImageModalityData(
        _id=ObjectId(),
        modality=ModalityType.IMAGE,
        content="sample_image_content",
        url="https://picsum.photos/200/300"
    )
    # Insert the data into the database
    await db.mset([image_data])

    # Retrieve the data from the database
    retrieved_data = await db.mget([image_data._id])

    # Verify the retrieved data
    assert len(retrieved_data) == 1
    assert retrieved_data[0] is not None
    assert retrieved_data[0].modality == "image"
    assert retrieved_data[0].content == "sample_image_content"
    assert retrieved_data[0].url == "http://example.com/image.png"

@pytest.mark.skip(reason="Skipping this test")
@pytest.mark.asyncio
async def test_mset_and_mget_text(db):
    # Create a sample TextModalityData
    text_data = TextModalityData(
        _id=ObjectId(),
        modality=ModalityType.TEXT,
        content="sample_text_content"
    )

    # Insert the data into the database
    await db.mset([text_data])

    # Retrieve the data from the database
    retrieved_data = await db.mget([text_data._id])

    # Verify the retrieved data
    assert len(retrieved_data) == 1
    assert retrieved_data[0] is not None
    assert retrieved_data[0].modality == ModalityType.TEXT
    assert retrieved_data[0].content == "sample_text_content"

@pytest.mark.skip(reason="Skipping this test")
@pytest.mark.asyncio
async def test_mdelete(db):
    # Create a sample TextModalityData
    text_data = TextModalityData(
        _id=ObjectId(),
        modality=ModalityType.TEXT,
        content="sample_text_content"
    )

    # Insert the data into the database
    await db.mset([text_data])

    # Delete the data from the database
    await db.mdelete([text_data._id])

    # Attempt to retrieve the deleted data
    retrieved_data = await db.mget([text_data._id])

    # Verify the data has been deleted
    assert len(retrieved_data) == 1
    assert retrieved_data[0] is None