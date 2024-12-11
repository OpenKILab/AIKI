import asyncio
import pytest
from aiki.multimodal.base import MultiModalProcessor, ModalityType
from aiki.multimodal.text import TextHandler, TextModalityData, TextHandlerOP
from aiki.multimodal.image import ImageHandler, ImageModalityData, ImageHandlerOP
from aiki.database.sqlite import SQLiteDB
from bson import ObjectId

@pytest.fixture(scope="module")
def sqlite_db():
    db = SQLiteDB(db_url="sqlite://:memory:")
    yield db
    asyncio.run(db.close())

@pytest.fixture
def processor(sqlite_db):
    processor = MultiModalProcessor()
    processor.register_handler(ModalityType.TEXT, TextHandler(sqlite_db))
    processor.register_handler(ModalityType.IMAGE, ImageHandler(sqlite_db))
    return processor

def test_register_handler(processor):
    assert ModalityType.TEXT.value in processor.handlers
    assert ModalityType.IMAGE.value in processor.handlers

def test_execute_text_operation(processor):
    text_data = TextModalityData(_id=ObjectId(), content="Sample text")
    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [text_data])
    result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, [text_data._id])
    assert result[0].content == "Sample text"

def test_execute_image_operation(processor):
    image_data = ImageModalityData(_id=ObjectId(), url="https://picsum.photos/200/300")
    processor.execute_operation(ModalityType.IMAGE, ImageHandlerOP.MSET, [image_data])
    result = processor.execute_operation(ModalityType.IMAGE, ImageHandlerOP.MGET, [image_data._id])
    assert result[0].url == "https://picsum.photos/200/300"

