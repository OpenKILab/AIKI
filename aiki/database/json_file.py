from aiki.database import BaseKVDatabase
from aiki.multimodal import BaseModalityData
from aiki.serialization import JsonEncoder, JsonDecoder

import os
import json
from typing import List

from bson import ObjectId

class JSONFileDB(BaseKVDatabase):

    def mdelete(self, ids: List[ObjectId]):
        file_data = self._read_file()
        for _id in ids:
            if str(_id) in file_data:
                del file_data[str(_id)]
        self._write_file(file_data)


    def mset(self, data_list: List[BaseModalityData]):
        file_data = self._read_file()
        for data in data_list:
            file_data[str(data._id)] = data
        self._write_file(file_data)

    def mget(self, ids: List[ObjectId]):
        """
        Retrieve multiple modality data by their IDs.
        If a _id is not found, the corresponding data will be None.
        """
        file_data = self._read_file()
        return [file_data.get(str(_id)) for _id in ids]


    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        # Initialize empty JSON file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump({}, f)

    def connect(self):
        pass

    def close(self):
        pass

    def _read_file(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f, cls=JsonDecoder)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON file: {e}")

    def _write_file(self, data):
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4, cls=JsonEncoder)

