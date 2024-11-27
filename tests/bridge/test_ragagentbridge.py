import pytest
from unittest.mock import MagicMock, patch
from aiki.bridge.rag_agent_bridge import RAGAgentBridge, RetrievalData
from aiki.multimodal.text import TextModalityData
from bson import ObjectId
from datetime import datetime, timedelta

@pytest.fixture
def setup_rag_agent():
    with patch('aiki.bridge.RAGAgentBridge.JSONFileDB') as MockJSONFileDB, \
            patch('aiki.bridge.RAGAgentBridge.ChromaDB') as MockChromaDB, \
            patch('aiki.bridge.RAGAgentBridge.MultimodalIndexer') as MockMultimodalIndexer:

        rag_agent = RAGAgentBridge(name="test_agent")
        
        mocked_retrieval_data = RetrievalData(items=[])
        
        rag_agent.multimodal_retriever.search = MagicMock(return_value=mocked_retrieval_data)
        
        rag_agent.multimodal_indexer = MockMultimodalIndexer.return_value
        rag_agent.multimodal_indexer.index = MagicMock()
        
        return rag_agent

def test_add(setup_rag_agent):
    rag_agent = setup_rag_agent
    
    retrieval_data = RetrievalData(
        items=[
            TextModalityData(
                content="Test content",
                _id=ObjectId(),
                metadata={"timestamp": int(datetime.now().timestamp())}
            )
        ]
    )
    
    result = rag_agent.add(retrieval_data)
    
    rag_agent.multimodal_indexer.index.assert_called_once()
    
    assert isinstance(result, RetrievalData)
    assert len(result.items) == 1
    assert result.items[0].content == "Test content"


def test_query(setup_rag_agent):
    rag_agent = setup_rag_agent
    
    retrieval_data = RetrievalData(items=[
        TextModalityData(
            content= "棕色狗狗",
            _id = ObjectId(),
            metadata={
                "start_time": 1324235,
                "end_time": int(datetime.now().timestamp())
            }
        )
    ])
    
    result = rag_agent.query(retrieval_data)
    
    rag_agent.multimodal_retriever.search.assert_called_once()
    
    assert isinstance(result, RetrievalData)
    assert result.items == []