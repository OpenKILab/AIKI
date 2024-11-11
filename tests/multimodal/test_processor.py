from aiki.multimodal import MultiModalProcessor, TextHandler, TextModalityData, TextHandlerOP

from aiki.multimodal import ModalityType

from bson import ObjectId

from aiki.database import JSONFileDB

def test_processor_text():

    processor = MultiModalProcessor()
    processor.register_handler(ModalityType.TEXT, TextHandler(database=JSONFileDB("data.json")))

    # Test mset, mget, mdelete
    _id = ObjectId()
    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id=_id,
                                                                                         content="hello")])
    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, [_id])

    print(mget_result[0])

    assert mget_result[0].content == "hello"

    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, [_id, ObjectId(), _id])

    assert len(mget_result) == 3 and mget_result[1] is None

    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id=_id,
                                                                                         content="hello_again")])

    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, [_id])

    assert mget_result[0].content == "hello_again"

    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MDELETE, [_id])

    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, [_id])

    assert mget_result[0] is None

test_processor_text()