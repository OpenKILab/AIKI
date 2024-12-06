# test_sqlite_db.py
import asyncio
import logging
import pytest
from tortoise import Tortoise
from bson import ObjectId
from aiki.database.sqlite import SQLiteDB
from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData

@pytest.fixture(scope="module")
def db():
    db_url = "sqlite://:memory:"
    db_instance = SQLiteDB(db_url)
    yield db_instance
    asyncio.run(db_instance.close())

@pytest.mark.asyncio
async def test_mset_and_mget_image(db):
    image_data = ImageModalityData(
        _id=ObjectId(),
        modality=ModalityType.IMAGE,
        content="sample_image_content",
        url="https://picsum.photos/200/300"
    )
    await db.mset([image_data])

    retrieved_data = await db.mget([image_data._id])

    assert len(retrieved_data) == 1
    assert retrieved_data[0] is not None
    assert retrieved_data[0].modality == ModalityType.IMAGE
    assert retrieved_data[0].url == "https://picsum.photos/200/300"

@pytest.mark.asyncio
async def test_mset_and_mget_text(db):
    text_data = TextModalityData(
        _id=ObjectId(),
        modality=ModalityType.TEXT,
        content="sample_text_content"
    )

    await db.mset([text_data])

    res = await db.mget([text_data._id])

    assert len(res) == 1
    assert res[0] is not None
    assert res[0].modality == ModalityType.TEXT
    assert res[0].content == "sample_text_content"

@pytest.mark.asyncio
async def test_mdelete(db):
    text_data = TextModalityData(
        _id=ObjectId(),
        modality=ModalityType.TEXT,
        content="sample_text_content"
    )

    await db.mset([text_data])

    await db.mdelete([text_data._id])

    res = await db.mget([text_data._id])

    assert len(res) == 1
    assert res[0] is None