import asyncio
from dataclasses import dataclass
import os
from aiki.multimodal.base import BaseModalityData, BaseModalityHandler, ModalityType, BaseModalityHandlerOP

from typing import Generic, TypeVar, Union, Dict, Any, List, TypedDict, Literal, Optional
from bson import ObjectId
from urllib.request import urlopen
import base64

from petrel_client.client import Client


class ImageHandlerOP(BaseModalityHandlerOP):
    MSET = "mset"
    MGET = "mget"
    MDELETE = "mdelete"

@dataclass()
class ImageModalityData(BaseModalityData):
    modality: ModalityType = ModalityType.IMAGE
    # url can be a http url or a data uri
    _content: Union[str, bytes] = None
    url: str = ""

    # content is a base64 encoded string
    @property
    def content(self):
        if self._content:
            return self._content
        elif self.url:
            try:
                self.load_content(self.url)
                return self._content
            except ValueError:
                raise ValueError(f"Invalid url: {self.url}")
        else:
            raise ValueError("No content or url provided")

    @content.setter
    def content(self, content):
        self._content = content
        
    def load_content(self, path_or_url: str):
        if path_or_url.startswith("s3://"):
            client = Client('~/petreloss.conf')
            data = client.get(path_or_url)
            self._content = data
        elif os.path.isfile(path_or_url):
            with open(path_or_url, "rb") as file:
                data = file.read()
                self._content = base64.b64encode(data).decode("utf-8")
        else:
            try:
                with urlopen(path_or_url) as response:
                    data = response.read()
                    self._content = base64.b64encode(data).decode("utf-8")
            except ValueError:
                raise ValueError(f"Invalid path or url: {path_or_url}")
    
    '''
    # content is a base64 encoded string
    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, url):
        try:
            self._url = url
            with urlopen(self.url) as response:
                data = response.read()
                self.content = base64.b64encode(data).decode("utf-8")
                return self._url
        except ValueError:
            raise ValueError(f"Invalid url: {self.url}")
    '''
class ImageHandler(BaseModalityHandler):
    def __init__(self, database):
        super().__init__(database)
        
    def mget(self, ids: List[ObjectId]) -> List[ImageModalityData]:
        return asyncio.run(self.database.mget(ids))

    def mset(self, data_list: List[ImageModalityData]):
        asyncio.run(self.database.mset(data_list))

    def mdelete(self, ids):
        asyncio.run(self.database.mdelete(ids))