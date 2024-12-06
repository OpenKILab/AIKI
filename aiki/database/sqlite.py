import asyncio
import sqlite3
from tortoise import Tortoise, fields, models
from tortoise.transactions import in_transaction
from aiki.database import BaseRelDatabase
from aiki.multimodal import BaseModalityData
from bson import ObjectId
from typing import List
import json

from aiki.multimodal.base import ModalityType
from aiki.multimodal.image import ImageModalityData
from aiki.multimodal.text import TextModalityData

class ModalityData(models.Model):
    id = fields.BinaryField(pk=True)
    modality = fields.CharField(max_length=255)
    content = fields.BinaryField(null=True)
    url = fields.CharField(max_length=255, null=True)
    metadata = fields.JSONField(null=True)
    colbert_tensor = fields.BinaryField(null=True)
    
    class Meta:
        table = "modality_data"
    
class SQLiteDB(BaseRelDatabase):
    def __init__(self, db_url: str):
        self.db_url = db_url
        asyncio.run(self._init())
        
    async def _init(self):
        await Tortoise.init(
            db_url = self.db_url,
            # TODO: why '__main__' error
            modules = {'models': ['aiki.database.sqlite']}
        )
        await Tortoise.generate_schemas()
        
    async def mset(self, data_list: List[BaseModalityData]):
        async with in_transaction() as conn:
            for data in data_list:
                await ModalityData.create(
                    id=data._id.binary,
                    modality=data.modality.name,
                    content=data.content.encode('utf-8') if data.content else None,
                    url=getattr(data, 'url', None),
                    metadata=data.metadata if hasattr(data, 'metadata') else None,
                    colbert_tensor=data.colbert_tensor.tobytes() if hasattr(data, 'colbert_tensor') else None
                )
    
    async def mget(self, ids: List[ObjectId]):
        results = []
        for _id in ids:
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
                content=row.content.decode('utf-8') if row.content else None,
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
                await ModalityData.filter(id=_id.binary).delete()
    
    async def close(self):
        await Tortoise.close_connections()