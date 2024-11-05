from typing import List
from openai import OpenAI
from aiki.config.config import Config
from aiki.corpus.mockdatabase import DatabaseConnectionFactory, DatabaseConnection, KVSchema
from aiki.corpus.storage import StorageBase
from bson import ObjectId
from datetime import datetime
from abc import ABC, abstractmethod

from aiki.indexer.chunker import BaseChunker, FixedSizeChunker
from aiki.modal.retrieval_data import RetrievalData, RetrievalItem, RetrievalType

import os

# 多模态数据生成文本摘要
class BaseSummaryGenerator(ABC):
    def __init__(self, model_path):
        self.model_path = model_path
        
    def generate_summary(self, data: RetrievalItem):
        ...
        
class ModelSummaryGenerator(BaseSummaryGenerator):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = self.load_model(self.model_path) 
    
    def load_model(self, model_path):
        # load tokenizer and model
        ...
        
    def generate_summary(self, data: RetrievalItem):
        ...

class APISummaryGenerator(BaseSummaryGenerator):
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, '..', '..', 'aiki', 'config', 'config.yaml')
        config = Config(config_path)
        
        super().__init__(config.get('model_path', 'default_model_path'))
        self.client = OpenAI(
            base_url=config.get('base_url', "https://api.claudeshop.top/v1")
        )
        self.model = config.get('model', "gpt-4o-mini")
        
    def generate_summary(self, data: RetrievalItem) -> str:
        item = data
        if item.type not in [RetrievalType.IMAGE, RetrievalType.TEXT]:
            raise ValueError(f"{self.__class__.__name__}.genearte_summary(). There is no such modal data processing method")
        
        content_type = "image_url" if item.type == RetrievalType.IMAGE else "text"
        content_value = {
            "url": f"data:image/jpeg;base64,{item.content}"
        } if item.type == RetrievalType.IMAGE else item.content
        
        prompt_text = "What is in this image?" if item.type == RetrievalType.IMAGE else "Please summarize this text."
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text,
                        },
                        {
                            "type": content_type,
                            content_type: content_value,
                        },
                    ],
                }
            ],
        )
        summary = response.choices[0].message.content

        return summary

class BaseIndexer(ABC):
    def __init__(self, model_path, sourcedb: DatabaseConnection, vectordb: DatabaseConnection, chunker: BaseChunker = FixedSizeChunker()):
        self.model_path = model_path
        self.sourcedb = sourcedb  # Source database storage
        self.vectordb = vectordb  # Vector database storage
        self.chunker = chunker
        
    def index(self, data):
        raise NotImplementedError(f"{self.__class__.__name__}.index() must be implemented in subclasses.")

class TextIndexer(BaseIndexer):
    def index(self, data: RetrievalData):
        for retreval_data in data.items:
            if retreval_data.type != RetrievalType.TEXT:
                raise ValueError(f"{self.__class__.__name__}.index(). Unsupported data type: {retreval_data.type}")
            id = ObjectId()
            dataSchema = KVSchema(
                _id=id,
                modality="text",
                summary="",
                source_encoded_data=retreval_data.content,
                inserted_timestamp=datetime.now(),
                parent=[],
                children=[]
            )
            self.sourcedb.create(dataSchema)
            chunks = self.chunker.chunk(retreval_data.content)
            for data in chunks:
                cur_id = ObjectId()
                dataSchema = KVSchema(
                    _id=cur_id,
                    modality="text",
                    summary="",
                    source_encoded_data=data,
                    inserted_timestamp=datetime.now(),
                    parent=[id],
                    children=[]
                )
                self.sourcedb.create(dataSchema)
                self.vectordb.upsert(docs=[data], ids=[str(cur_id)])
                
class ImageIndexer(BaseIndexer):
    def __init__(self, model_path, sourcedb: DatabaseConnection, vectordb: DatabaseConnection, chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator()):
        super().__init__(model_path, sourcedb, vectordb, chunker)
        self.summary_generator = summary_generator
        
    def index(self, data: RetrievalData):
        for retreval_data in data.items:
            if retreval_data.type != RetrievalType.IMAGE:
                raise ValueError(f"{self.__class__.__name__}.index(). Unsupported data type: {retreval_data.type}")
            id = ObjectId()
            dataSchema = KVSchema(
                _id=id,
                modality="image",
                summary=self.summary_generator.generate_summary(),
                source_encoded_data=retreval_data.content,
                inserted_timestamp=datetime.now(),
                parent=[],
                children=[]
            )
            self.sourcedb.create(dataSchema)
            self.vectordb.upsert(docs=[dataSchema['summary']], ids=[str(id)])
        
class MultimodalIndexer(BaseIndexer):
    def __init__(self, model_path, sourcedb: DatabaseConnection, vectordb: DatabaseConnection, chunker: BaseChunker = FixedSizeChunker(), summary_generator: BaseSummaryGenerator = APISummaryGenerator()):
        super().__init__(model_path, sourcedb, vectordb)
        self.text_indexer = TextIndexer(model_path, sourcedb, vectordb, chunker)
        self.image_indexer = ImageIndexer(model_path, sourcedb, vectordb, chunker, summary_generator)
    
    def index(self, data: RetrievalData):
        text_retrieval_data = RetrievalData(items=[])
        image_retrieval_data = RetrievalData(items=[])
        # slice data with type
        for retrieval_data in data.items:
            if retrieval_data.type == RetrievalType.TEXT:
                text_retrieval_data.items.append(
                    retrieval_data
                )
            elif retrieval_data.type == RetrievalType.IMAGE:
                image_retrieval_data.items.append(
                    retrieval_data
                )
            else:
                raise ValueError(f"Unsupported data type: {retrieval_data.type}")
        self.text_indexer.index(text_retrieval_data)
        self.image_indexer.index(image_retrieval_data)
        
class KnowledgeGraphIndexer(BaseIndexer):
    ...
    
# Example usage
if __name__ == "__main__":
    source_db_connection = DatabaseConnectionFactory.create_connection('json_file', file_name="json_file")
    
    chroma_connection = DatabaseConnectionFactory.create_connection('chroma', index_file='chroma_index')

    text_indexer = TextIndexer(model_path='path/to/model', sourcedb=source_db_connection, vectordb=chroma_connection)

    retrieval_data = RetrievalData(
        items=[
            {"type": RetrievalType.TEXT, "content": f"""MARLEY'S GHOST


Marley was dead, to begin with. There is no doubt whatever about that.
The register of his burial was signed by the clergyman, the clerk, the
undertaker, and the chief mourner. Scrooge signed it. And Scrooge's name
was good upon 'Change for anything he chose to put his hand to. Old
Marley was as dead as a door-nail.

Mind! I don't mean to say that I know of my own knowledge, what there is
particularly dead about a door-nail. I might have been inclined, myself,
to regard a coffin-nail as the deadest piece of ironmongery in the
trade. But the wisdom of our ancestors is in the simile; and my
unhallowed hands shall not disturb it, or the country's done for. You
will, therefore, permit me to repeat, emphatically, that Marley was as
dead as a door-nail.

Scrooge knew he was dead? Of course he did. How could it be otherwise?
Scrooge and he were partners for I don't know how many years. Scrooge
was his sole executor, his sole administrator, his sole assign, his sole
residuary legatee, his sole friend, and sole mourner. And even Scrooge
was not so dreadfully cut up by the sad event but that he was an
excellent man of business on the very day of the funeral, and solemnised
it with an undoubted bargain.

The mention of Marley's funeral brings me back to the point I started
from. There is no doubt that Marley was dead. This must be distinctly
understood, or nothing wonderful can come of the story I am going to
relate. If we were not perfectly convinced that Hamlet's father died
before the play began, there would be nothing more remarkable in his
taking a stroll at night, in an easterly wind, upon his own ramparts,
than there would be in any other middle-aged gentleman rashly turning
out after dark in a breezy spot--say St. Paul's Churchyard, for
instance--literally to astonish his son's weak mind.

Scrooge never painted out Old Marley's name. There it stood, years
afterwards, above the warehouse door: Scrooge and Marley. The firm was
known as Scrooge and Marley. Sometimes people new to the business called
Scrooge Scrooge, and sometimes Marley, but he answered to both names. It
was all the same to him.

Oh! but he was a tight-fisted hand at the grindstone, Scrooge! a
squeezing, wrenching, grasping, scraping, clutching, covetous old
sinner! Hard and sharp as flint, from which no steel had ever struck out
generous fire; secret, and self-contained, and solitary as an oyster.
The cold within him froze his old features, nipped his pointed nose,
shrivelled his cheek, stiffened his gait; made his eyes red, his thin
lips blue; and spoke out shrewdly in his grating voice. A frosty rime
was on his head, and on his eyebrows, and his wiry chin. He carried his
own low temperature always about with him; he iced his office in the
dog-days, and didn't thaw it one degree at Christmas.

External heat and cold had little influence on Scrooge. No warmth could
warm, no wintry weather chill him. No wind that blew was bitterer than
he, no falling snow was more intent upon its purpose, no pelting rain
less open to entreaty. Foul weather didn't know where to have him. The
heaviest rain, and snow, and hail, and sleet could boast of the
advantage over him in only one respect. They often 'came down'
handsomely, and Scrooge never did.

Nobody ever stopped him in the street to say, with gladsome looks, 'My
dear Scrooge, how are you? When will you come to see me?' No beggars
implored him to bestow a trifle, no children asked him what it was
o'clock, no man or woman ever once in all his life inquired the way to
such and such a place, of Scrooge. Even the blind men's dogs appeared to
know him; and, when they saw him coming on, would tug their owners into
doorways and up courts; and then would wag their tails as though they
said, 'No eye at all is better than an evil eye, dark master!'

But what did Scrooge care? It was the very thing he liked. To edge his
way along the crowded paths of life, warning all human sympathy to keep
its distance, was what the knowing ones call 'nuts' to Scrooge.

Once upon a time--of all the good days in the year, on Christmas
Eve--old Scrooge sat busy in his counting-house. It was cold, bleak,
biting weather; foggy withal; and he could hear the people in the court
outside go wheezing up and down, beating their hands upon their breasts,
and stamping their feet upon the pavement stones to warm them. The City
clocks had only just gone three, but it was quite dark already--it had
not been light all day--and candles were flaring in the windows of the
neighbouring offices, like ruddy smears upon the palpable brown air. The
fog came pouring in at every chink and keyhole, and was so dense
without, that, although the court was of the narrowest, the houses
opposite were mere phantoms. To see the dingy cloud come drooping down,
obscuring everything, one might have thought that nature lived hard by,
and was brewing on a large scale."""}
        ]
    )

    # Index the data
    text_indexer.index(retrieval_data)