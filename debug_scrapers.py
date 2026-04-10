import asyncio
from kimi_service import KimiService

async def test():
    kms = KimiService()
    print("Testing Amazon...")
    res = await kms.search_amazon_products("blue shirts", limit=5)
    print(f"Amazon: {len(res)} results")
    for r in res:
        print(f" - {r['name']} ({r['price']})")
        
    print("\nTesting Fast Data for 'mens hat'...")
    res = await kms.get_fast_bing_data("mens hat")
    print(f"Fast Data: {len(res)} results")
    for r in res[:5]:
        print(f" - {r['name']} ({r['price']}) -> {r['image_url']}")

if __name__ == "__main__":
    asyncio.run(test())
