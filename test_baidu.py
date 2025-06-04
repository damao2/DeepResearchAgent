import asyncio
from src.tools.search.baidu_search import BaiduSearchEngine

async def test_baidu_search():
    baidu_engine = BaiduSearchEngine()
    query = "最新人工智能新闻" # Use a Chinese query for Baidu
    print(f"Testing Baidu search for query: '{query}'")
    try:
        results = await baidu_engine.perform_search(query, num_results=5)
        if results:
            print("Baidu search successful! Results:")
            for i, item in enumerate(results):
                print(f"{i+1}. Title: {item.title}, URL: {item.url}")
        else:
            print("Baidu search returned no results.")
    except Exception as e:
        print(f"Baidu search failed with an error: {e}")

if __name__ == "__main__":
    asyncio.run(test_baidu_search())
