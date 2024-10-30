from abc import ABC, abstractmethod
from typing import List, str

class BaseChunker(ABC):
    '''
    1. Fixed size chunking
        根据指定的字符数、单词数量或者token数量将文本均匀分割; 同时可以在chunk之间保留重叠;
    2. Semantic chunking
        块间语义相似度最小，合并语义上相似的块
    3. Recursive chunking
        首先基于内在分隔符切分，比如自然段落、章节，接着对自然段落和章节递归切分；
    4. Document structure-based chunking
        doc拆成title、introduction、section1、section2、conclusion这几个chunk;
    5. LLM-based chunking
        使用LLM来根据上下文确定块中应包含多少文本以及哪些文本的可能性。受限于llm上下文长度
    '''
    def chunk(self, data: str) -> List[str]:
        raise NotImplementedError(f"{self.__class__.__name__}.chunk() must be implemented in subclasses.")

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def chunk(self, data: str):
        chunks = []
        for i in range(0, len(data), self.char_limit):
            chunks.append(data[i:i + self.char_limit])
        return chunks