from dataclasses import dataclass
from aiki.multimodal.base import BaseModalityData, BaseModalityHandler, ModalityType, BaseModalityHandlerOP

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId
from urllib.request import urlopen
import base64

class ImageHandlerOP(BaseModalityHandlerOP):
    MSET = "mset"
    MGET = "mget"
    MDELETE = "mdelete"

@dataclass()
class ImageModalityData(BaseModalityData):
    modality: ModalityType = ModalityType.IMAGE
    # url can be a http url or a data uri
    url: str = ""
    _content: str = None

    # content is a base64 encoded string
    @property
    def content(self):
        if self._content:
            return self._content
        elif self.url:
            try:
                with urlopen(self.url) as response:
                    data = response.read()
                    self._content = base64.b64encode(data).decode("utf-8")
                    return self._content
            except ValueError:
                raise ValueError(f"Invalid url: {self.url}")
        else:
            raise ValueError("No content or url provided")

    @content.setter
    def content(self, content):
        self._content = content

class ImageHandler(BaseModalityHandler):
    def __init__(self, database):
        super().__init__(database)
        
    def mget(self, ids: List[ObjectId]) -> List[ImageModalityData]:
        return self.database.mget(ids)

    def mset(self, data_list: List[ImageModalityData]):
        self.database.mset(data_list)

    def mdelete(self, ids):
        self.database.mdelete(ids)