import asyncio
from src.tools.search.bing_search import BingSearchEngine

async def test_bing_search():
    bing_engine = BingSearchEngine()
    query = "latest AI news"
    print(f"Testing Bing search for query: '{query}'")
    try:
        results = await bing_engine.perform_search(query, num_results=5)
        if results:
            print("Bing search successful! Results:")
            for i, item in enumerate(results):
                print(f"{i+1}. Title: {item.title}, URL: {item.url}")
        else:
            print("Bing search returned no results.")
    except Exception as e:
        print(f"Bing search failed with an error: {e}")

if __name__ == "__main__":
    asyncio.run(test_bing_search())
