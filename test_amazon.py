import asyncio
from kimi_service import kimi_service
from asset_processor import AssetProcessor
import httpx

async def test():
    # 1. Search Amazon
    print("Searching Amazon India for Baby Clothes...")
    products = await kimi_service.search_amazon_products("baby clothes", limit=5)
    
    for i, p in enumerate(products):
        print(f"\nProduct {i+1}: {p['name']}")
        print(f"Extracted img_url: {p['image_url']}")
        
        # 2. Test downloading it
        url = p['image_url']
        if not url:
            print("No URL to test.")
            continue
            
        processor = AssetProcessor()
        headers = processor._get_headers(url)
        print(f"Using headers: {headers}")
        
        try:
            with httpx.Client(timeout=10.0, verify=False) as client:
                resp = client.get(url, headers=headers)
                print(f"Download status: {resp.status_code}")
                print(f"Content length: {len(resp.content)} bytes")
                if resp.status_code == 200:
                    if len(resp.content) < 1000:
                        print("FAILED: Image is too small (<1000 bytes)")
                    else:
                        print("SUCCESS: Image downloaded correctly!")
                else:
                    print(f"FAILED: HTTP {resp.status_code}")
        except Exception as e:
            print(f"FAILED: Exception {e}")

if __name__ == "__main__":
    asyncio.run(test())
