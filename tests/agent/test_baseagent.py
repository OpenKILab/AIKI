from datetime import datetime
import unittest
from unittest.mock import MagicMock, Mock

from bson import ObjectId
from aiki.agent.baseagent import InfoExtractAgent, MemoryEditAgent, Message, AgentAction
from aiki.modal.retrieval_data import RetrievalData
from aiki.multimodal.text import TextModalityData
from aiki.proxy.query import query_llm

class TestInfoExtractAgent(unittest.TestCase):
    def setUp(self):
        self.agent = InfoExtractAgent(config={}, extract_model=query_llm)

    @unittest.skip("Skipping test_talk_with_text_message_query_message")
    def test_talk_with_text_message_query_message(self):
        test_message = [
            Message(role='user', type='text', content='上周忘了吃了些啥', action=AgentAction.QUERY, metadata={}, new_memory=None)
        ]
        
        result = self.agent.talk(test_message)
        
        print(result.metadata['start_time'])
        print(result.metadata['end_time'])
        self.assertIsInstance(result, Message)
        self.assertEqual(result.action, AgentAction.QUERY)
        self.assertIn('start_time', result.metadata)
        self.assertIn('end_time', result.metadata)
    
class TestMemoryEditAgent(unittest.TestCase):
    def setUp(self):
        self.mock_rag = MagicMock()
        self.mock_process_model = MagicMock()

        self.agent = MemoryEditAgent(config={}, process_model=query_llm)
        self.agent.rag = self.mock_rag

    def test_search(self):
        mock_retrieval_data = RetrievalData(items=[
            TextModalityData(
                content="在徐汇吃了些东西",
                _id=ObjectId(),
                metadata={
                        "timestamp": int(datetime.now().timestamp())
                },
            )
        ])
        self.mock_rag.query.return_value = mock_retrieval_data

        self.mock_process_model.return_value = '{"selected_ids": ["some_id"]}'

        test_message = Message(
            content="我吃了啥",
            metadata={"timestamp": int(datetime.now().timestamp())},
            action=AgentAction.QUERY
        )

        result_message = self.agent.search(test_message)

        # Assertions
        self.mock_rag.query.assert_called_once_with(mock_retrieval_data)
        self.assertEqual(result_message.content, mock_retrieval_data.to_json())
        self.assertIsInstance(result_message, Message)
        self.assertIsInstance(RetrievalData.from_json(result_message.content), RetrievalData)

    def test_talk_with_query_action(self):
        mock_retrieval_data = RetrievalData(items=[
            TextModalityData(
                content="在徐汇吃了些东西",
                _id=ObjectId(),
                metadata={"timestamp": int(datetime.now().timestamp())},
            )
        ])
        self.mock_rag.query.return_value = mock_retrieval_data

        test_message = Message(
            content="我吃了啥",
            metadata={"timestamp": int(datetime.now().timestamp())},
            action=AgentAction.QUERY
        )

        result_message = self.agent.talk(test_message)

        self.mock_rag.query.assert_called_once_with(mock_retrieval_data)
        self.assertEqual(result_message.content, mock_retrieval_data.to_json())
        self.assertIsInstance(result_message, Message)
        self.assertIsInstance(RetrievalData.from_json(result_message.content), RetrievalData)


if __name__ == '__main__':
    unittest.main()