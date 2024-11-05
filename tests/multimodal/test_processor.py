from aiki.multimodal import MultiModalProcessor, TextHandler, TextModality, ModalityType

from aiki.multimodal import ModalityType
from aiki.database import JSONFileDB

processor = MultiModalProcessor()
processor.register_handler(ModalityType.TEXT, TextHandler(database=JSONFileDB("data.json")))

processor.execute_operation(ModalityType.TEXT, "mset", [TextModality(_id="1",
                                                          modality=ModalityType.TEXT,
                                                          text="hello")])

processor.execute_operation(ModalityType.TEXT, "mget", ["1"])