from aiki.multimodal import MultiModalProcessor, TextHandler, TextModalityData, TextHandlerOP

from aiki.multimodal import ModalityType


from aiki.database import JSONFileDB

def test_processor_text():

    processor = MultiModalProcessor()
    processor.register_handler(ModalityType.TEXT, TextHandler(database=JSONFileDB("data.json")))

    # Test mset, mget, mdelete
    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id="1",
                                                          modality=ModalityType.TEXT,
                                                          text="hello")])
    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, ["1"])

    assert  mget_result[0].text == "hello"

    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MSET, [TextModalityData(_id="1",
                                                                         modality=ModalityType.TEXT,
                                                                         text="hello_again")])

    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, ["1"])

    assert  mget_result[0].text == "hello_again"

    processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MDELETE, ["1"])

    mget_result = processor.execute_operation(ModalityType.TEXT, TextHandlerOP.MGET, ["1"])

    assert mget_result[0] is None

test_processor_text()

