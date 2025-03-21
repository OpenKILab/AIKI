def convert_search_results(query, input_data):
    # 解析输入数据
    results = []
    for item in input_data:
        result = {
            'id': f"https://api.bing.microsoft.com/api/v7/#WebPages.{item['position'] - 1}",
            'name': item['title'],
            'url': item['link'],
            'isFamilyFriendly': True,
            'displayUrl': item['displayed_link'],
            'snippet': item['snippet'],
            'dateLastCrawled': '2025-02-20T05:17:00.0000000Z',
            'language': 'en',
            'isNavigational': False,
            'noCache': False,
            'siteName': item['source']
        }
        results.append(result)

    # 构建输出格式
    output = {
        '_type': 'SearchResponse',
        'queryContext': {
            'originalQuery': query
        },
        'webPages': {
            'webSearchUrl': 'https://www.bing.com/search?q='+query,
            'totalEstimatedMatches': len(results),
            'value': results
        }
    }
    
    return output

# 示例输入数据
input_data = [
    {
        'position': 1,
        'title': '火龙果-维基百科，自由的百科全书',
        'link': 'https://zh.wikipedia.org/zh-hans/%E7%81%AB%E9%BE%99%E6%9E%9C',
        'displayed_link': 'https://zh.wikipedia.org›zh-hans',
        'snippet': '火龙果又称红龙果、龙珠果、仙人掌果或量天尺果，是多种仙人掌科蛇鞭柱属的植物果实的总称（过去分类为量天尺属）。水果呈椭圆形，直径10－12cm，表皮多为红色或黄色，覆有...',
        'source': '维基百科'
    },
    # 其他条目...
]

# 调用转换函数
output_data = convert_search_results(input_data)
print(output_data)