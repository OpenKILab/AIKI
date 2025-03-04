from abc import ABC, abstractmethod
from typing import List
from chonkie import TokenChunker, SentenceChunker, SemanticChunker
import tiktoken

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
    def __init__(self, chunk_size: int = 512):
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')
        # self.chunker = SentenceChunker(self.tokenizer, chunk_size=24, chunk_overlap=1, min_chunk_size=chunk_size)
        self.chunk_size = chunk_size
        
    def chunk(self, data: str) -> List[str]:
        raise NotImplementedError(f"{self.__class__.__name__}.chunk() must be implemented in subclasses.")

class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512):
        super().__init__(chunk_size)
        self.chunker = TokenChunker(self.tokenizer, chunk_size=chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def chunk(self, data: str):
        if not isinstance(data, str):
            raise TypeError("data must be a string")
        if not data:
            return []

        results = []
        
        chunks = self.chunker(data)

        for chunk in chunks:
            results.append(chunk.text)
        return results

class CurSentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 6):
        super().__init__(chunk_size)
        self.chunker = SentenceChunker(self.tokenizer, chunk_size=24, chunk_overlap=1, min_chunk_size=chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def chunk(self, data: str):
        if not isinstance(data, str):
            raise TypeError("data must be a string")
        if not data:
            return []

        results = []
        
        chunks = self.chunker(data)

        for chunk in chunks:
            results.append(chunk.text)
        return results

class CurSemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 6):
        super().__init__(chunk_size)
        self.chunker = SemanticChunker()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

    def chunk(self, data: str):
        if not isinstance(data, str):
            raise TypeError("data must be a string")
        if not data:
            return []

        results = []
        
        chunks = self.chunker(data)

        for chunk in chunks:
            results.append(chunk.text)
        return results

if __name__ == "__main__":
    from bs4 import BeautifulSoup
    import time
    start_time = time.time()

    for _ in range(0, 1024):
        text = """<p style=\"text-indent: 2em;\">晶体硬度与晶体的结构有关，如金刚石的晶体结构与石墨有很大不同，其硬度也远大于石墨。晶体硬度还与构成晶体的元素和元素之间的键能有关，如原子晶体＞离子晶体＞分子晶体。</p><p style=\"text-indent: 2em;\">离子晶体由阴离子和阳离子组成，阴、阳离子交替排列在晶格结点上，它们之间以静电引力相结合，这种结合力所形成的键称为离子键。晶格断裂时，沿离子界面断开，断裂后表面露出的是不饱和的离子键。由于阴、阳离子的电子云可以近似地看成球形对称，故离子键没有方向性。一般配位数较高，硬度较大，极性较强。</p><p style=\"text-indent: 2em;\">原子晶体由原子组成，晶格结点上排列的是中性原子。它们靠共用电子对结合在一起，这种键称为共价键。共价键具有方向性和饱和性，一般配位数很小，因此，晶体结构的紧密程度远比离子晶格低。原子晶体中没有自由电子，故晶体是不良导体。矿物晶格断裂时，必须破坏共价键，其极性较强，共价键的键能强度比离子键的高，因此原子晶体的硬度比离子晶体高。</p><p style=\"text-indent: 2em;\">分子晶体的晶格中，分子是结构的基本单元，分子间由极弱的范德瓦耳斯力联结。晶格断裂时暴露出来的是分子键，为弱键。分子间的引力与分子间距离的7次方成反比。分子晶体的特点是：分子间无自由电子运动，故而分子晶体为不良导体；组成晶体的分子键很弱，因此分子晶体硬度小，对水的亲和力弱。</p><p style=\"text-indent: 2em;\">金属晶格的结点上为金属阳离子，阳离子的周围有自由运动的电子。阳离子与电子相互作用，结合成金属键。金属键无方向性和饱和性，因此具有最大的配位数和最紧密的堆积。由于自由电子可以在整个晶体中自由运动，晶格的导电性、导热性均较好。晶格断裂后，其断裂面上为强不饱和键。自然金属，如自然金、自然铜属于该类结构。</p>"""
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        # text = """test"""
        tokenizer = tiktoken.encoding_for_model('gpt-4o')
        chunker = SentenceChunker(tokenizer, chunk_size=24, chunk_overlap=1, min_chunk_size=16)
        res = chunker.chunk(text)
        # for item in res:
            # print(item)
            # print(len(item))

    elapsed_time = time.time() - start_time
    print(f"Processed 1024 iterations in {elapsed_time:.2f} seconds")
    print(f"Average time per iteration: {(elapsed_time/1024):.3f} seconds")