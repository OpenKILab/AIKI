from aiki.multimodal import MultiModalProcessor, ImageHandler, ImageModalityData, ImageHandlerOP
from aiki.multimodal import ModalityType
from aiki.database import JSONFileDB

from bson import ObjectId
import base64
from unittest.mock import patch

def test_processor_image():

    processor = MultiModalProcessor()
    processor.register_handler(ModalityType.IMAGE, ImageHandler(database=JSONFileDB("image_db.json")))

    # Test mset, mget, mdelete
    data =  b"image data"
    encoded_data = base64.b64encode(data).decode("utf-8")
    with patch("aiki.multimodal.image.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value.read.return_value = data

        _id = ObjectId()
        processor.execute_operation(ModalityType.IMAGE,
                                    ImageHandlerOP.MSET,
                                    [ImageModalityData(_id=_id,
                                                       url="http://example.com/example.jpg")])
        mget_result = processor.execute_operation(ModalityType.IMAGE, ImageHandlerOP.MGET, [_id])

        assert mget_result[0].content == encoded_data