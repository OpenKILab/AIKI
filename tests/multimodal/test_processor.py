from aiki.multimodal import MultiModalProcessor, TextHandler, TextModality, ModalityType

from aiki.database import JSONFileDB

processor = MultiModalProcessor()
processor.register_handler("text", TextHandler(database=JSONFileDB("data.json")))

processor.execute_operation("text", "mset", [TextModality(_id="1",
                                                          modality=ModalityType.TEXT,
                                                          text="hello")])

processor.execute_operation("text", "mget", ["1"])