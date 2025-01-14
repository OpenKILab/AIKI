import asyncio
import os
import sqlite3
from tortoise import Tortoise, fields, models
from tortoise.transactions import in_transaction
from aiki.database import BaseRelDatabase
from aiki.multimodal import BaseModalityData
from bson import ObjectId
from typing import List
import json
import logging
import time

from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData

class ModalityData(models.Model):
    id = fields.BinaryField(primary_key=True)
    modality = fields.CharField(max_length=255)
    content = fields.BinaryField(null=True)
    url = fields.CharField(max_length=255, null=True)
    metadata = fields.JSONField(null=True)
    colbert_tensor = fields.BinaryField(null=True)
    
    class Meta:
        table = "modality_data"
    
class SQLiteDB(BaseRelDatabase):
    def __init__(self, db_url: str):
        self.db_url = f"sqlite://{db_url}"
        asyncio.run(self._init())
        
    async def _init(self):
        await Tortoise.init(
            db_url = self.db_url,
            # TODO: why '__main__' error
            modules = {'models': ['aiki.database.sqlite']}
        )
        await Tortoise.generate_schemas()
        
    async def mset(self, data_list: List[BaseModalityData]):
        start_time = time.time()  # 记录开始时间
        modality_data_objects = []
        for data in data_list:
            _id = data._id
            content = data.content
            if isinstance(content, str):
                content = content.encode('utf-8')
            modality_data_objects.append(ModalityData(
                id=sqlite3.Binary(_id.binary),
                modality=data.modality.value,
                content=content,
                url=getattr(data, 'url', None),
                metadata=data.metadata if hasattr(data, 'metadata') else None,
                colbert_tensor=data.colbert_tensor.tobytes() if hasattr(data, 'colbert_tensor') else None
            ))
        await ModalityData.bulk_create(modality_data_objects)
        end_time = time.time()  # 记录结束时间
        logging.info(f"SQLite mset operation took {end_time - start_time} seconds")
    
    async def mget(self, ids: List[ObjectId]) -> List[BaseModalityData]:
        results = []
        for _id in ids:
            if isinstance(_id, str):
                _id = ObjectId(_id)
            row = await ModalityData.filter(id=_id.binary).first()
            if row:
                results.append(self._row_to_data(row))
            else:
                results.append(None)
        return results
    
    def _row_to_data(self, row):
        modality_type = ModalityType(row.modality)
        if modality_type == ModalityType.IMAGE:
            return ImageModalityData(
                _id=ObjectId(row.id),
                modality=modality_type,
                content=row.content.decode('utf-8') if isinstance(row.content, str) else row.content,
                url=row.url,
                metadata=row.metadata
            )
        elif modality_type == ModalityType.TEXT:
            return TextModalityData(
                _id=ObjectId(row.id),
                modality=modality_type,
                content=row.content.decode('utf-8') if row.content else None,
                metadata=row.metadata
            )
        else:
            raise ValueError(f"Unsupported modality type: {modality_type}")
        
    async def mdelete(self, ids: List[ObjectId]):
        async with in_transaction() as conn:
            for _id in ids:
                await conn.execute_query(
                    'DELETE FROM "modality_data" WHERE "id"=?',
                    (sqlite3.Binary(_id.binary),)  # 使用参数化查询传递二进制数据
                )
    
    async def close(self):
        await Tortoise.close_connections()