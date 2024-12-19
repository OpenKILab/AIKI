import pytest
from aiki.externalsearch.external_search import ExternalSearch

FAKE_API_KEY = "fake_api_key"

@pytest.fixture
def search_instance():
    """
    Fixture 用于初始化 ExternalSearch 实例。
    """
    return ExternalSearch(api_key=FAKE_API_KEY)

def test_search_text(search_instance, mocker):
    """
    测试 ExternalSearch.search 方法的文本搜索功能。
    """
    # 模拟返回的搜索结果
    mock_text_results = [
        {"type": "text", "title": "Sample Title", "snippet": "Sample Snippet", "link": "http://example.com"}
    ]

    # 使用 mocker 模拟 _execute_search 方法的行为
    mocker.patch.object(
        search_instance,
        "_execute_search",
        return_value=mock_text_results
    )

    query = "What is AI?"
    result = search_instance.search(query, "text")

    # 断言返回结果是否符合预期
    assert result == [
        {"type": "text", "title": "Sample Title", "snippet": "Sample Snippet", "link": "http://example.com"}
    ]
    assert len(result) == 1
    assert result[0]["type"] == "text"
    assert result[0]["title"] == "Sample Title"

def test_search_invalid_type(search_instance):
    """
    测试 ExternalSearch.search 方法对无效搜索类型的处理。
    """
    with pytest.raises(ValueError, match="Unsupported search type: invalid"):
        search_instance.search("test", "invalid")
