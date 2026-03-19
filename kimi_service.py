import os
import json
import asyncio
import re
import aiohttp
from urllib.parse import urljoin, urlparse
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler

load_dotenv()

SYSTEM_PROMPT = """
You are a real-time intelligent assistant.
Always give accurate and helpful answers.
CRITICAL: NEVER use markdown image syntax like ![alt](url) in your text responses.
"""

EXTRACTION_SYSTEM_PROMPT = """
Return ONLY valid JSON.
No explanation.
If missing → null.
"""

class KimiService:
    def __init__(self):
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"
        # Increase semaphore to allow more parallel extraction
        self.semaphore = asyncio.Semaphore(4)
        self.base_retail_domains = [
            "amazon.com", "amazon.in", "flipkart.com", "ebay.com"
        ]

    def detect_intent(self, query):
        q = query.lower()
        # 🚗 Vehicle / Mobility
        if any(x in q for x in ["car", "bike", "vehicle", "mileage", "scooter", "truck"]):
            return "vehicle"
        
        # 🛒 Shopping / Products
        shopping_keywords = [
            "buy", "price", "shop", "laptop", "phone", "macbook", "iphone", 
            "toy", "gift", "tshirt", "t-shirt", "shirt", "shoes", "shoe", 
            "cloth", "clothing", "wear", "jean", "pant", "fashion", "brand",
            "electronics", "gadget", "watch", "accessory"
        ]
        if any(x in q for x in shopping_keywords):
            return "shopping"
            
        return "general"

    async def get_vehicle_data(self, query):
        q = query.lower()
        if "image" in q:
            return await self.search_images(q)

        prompt = f"Give details for: {query}\nReturn JSON: name, price, mileage, fuel"
        try:
            print(f"DEBUG: Vehicle LLM start for {query}")
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if not response: return f"No data found for {query}"
            text = response.content[0].text
            return self._safe_json_parse(text, "vehicle")
        except Exception as e:
            print("Vehicle error:", e)
            return f"No data available for {query}"

    async def search_images(self, query):
        # 1. Smarter query cleaning: Remove filler words
        fillers = ["show me", "some", "images", "image", "of", "find", "search", "get", "pics", "pictures", "photos"]
        clean_query = query.lower()
        for f in fillers:
            clean_query = clean_query.replace(f, "")
        clean_query = clean_query.strip()
        
        if not clean_query: clean_query = query # Fallback

        print(f"DEBUG: Starting image search for: {clean_query}")
        # Try to find real images using the crawler
        try:
            # Bing search often has easier to scrape image URLs
            search_url = f"https://www.bing.com/images/search?q={clean_query.replace(' ', '+')}"
            
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=search_url)
                if result.success:
                    import re
                    html_content = result.html
                    
                    # Extract both image URL (murl) and page URL (purl)
                    # Bing encodes JSON in data-m attribute
                    blocks = re.findall(r'murl&quot;:&quot;(https?://[^&]+)&quot;,&quot;turl&quot;:&quot;[^&]+&quot;,&quot;contentUrl&quot;:&quot;[^&]+&quot;,&quot;purl&quot;:&quot;(https?://[^&]+)&quot;', html_content)
                    
                    if not blocks:
                        # Fallback: try simpler separate extraction if the complex one fails
                        murls = re.findall(r'murl&quot;:&quot;(https?://[^&]+)&quot;', html_content)
                        purls = re.findall(r'purl&quot;:&quot;(https?://[^&]+)&quot;', html_content)
                        blocks = list(zip(murls, purls))

                    # Deduplicate and filter
                    real_results = []
                    seen = set()
                    for img_url, pg_url in blocks:
                        if img_url.startswith("//"): img_url = "https:" + img_url
                        if pg_url.startswith("//"): pg_url = "https:" + pg_url
                        
                        if not img_url.startswith("http"): continue
                        # Filter out potential internal/junk URLs
                        if any(x in img_url for x in ["bing.com", "google.com", "gstatic.com", "microsoft.com"]): continue
                        if img_url in seen: continue
                        seen.add(img_url)
                        
                        real_results.append({
                            "name": f"{clean_query} {len(real_results) + 1}",
                            "image_url": img_url,
                            "source_url": pg_url
                        })
                        if len(real_results) >= 10: break
                    
                    if real_results:
                        print(f"DEBUG: Found {len(real_results)} real images with source URLs from Bing.")
                        return {
                            "type": "images",
                            "query": clean_query,
                            "results": real_results
                        }
                    else:
                        print("DEBUG: No real images found in Bing search result.")
        except Exception as e:
            print(f"Image search error: {e}")

        # Final fallback to working placeholder
        return {
            "type": "images",
            "query": clean_query,
            "results": [
                {
                    "name": f"{clean_query} 1",
                    "image_url": f"https://placehold.co/800x600?text={clean_query.replace(' ', '+')}+1",
                    "source_url": f"https://www.bing.com/images/search?q={clean_query.replace(' ', '+')}"
                },
                {
                    "name": f"{clean_query} 2",
                    "image_url": f"https://placehold.co/800x600?text={clean_query.replace(' ', '+')}+2",
                    "source_url": f"https://www.bing.com/images/search?q={clean_query.replace(' ', '+')}"
                }
            ]
        }

    async def get_fast_bing_data(self, query, num_results=10):
        print(f"DEBUG: Starting get_fast_bing_data for {query}")
        # 1. Parallel Search and Image Lookup
        urls_task = self.search_sources(query, intent="shopping", limit=num_results)
        images_task = self.search_images(query) # Proactive image lookup as fallback
        
        urls, images_res = await asyncio.gather(urls_task, images_task)
        
        fast_results = []
        bing_images = images_res.get("results", []) if images_res else []
        
        # Combine them
        for idx, url in enumerate(urls):
            img_url = bing_images[idx].get("image_url") if idx < len(bing_images) else None
            fast_results.append({
                "name": f"Product Option {idx+1}",
                "url": url,
                "source_url": url,
                "image_url": img_url,
                "price": "Pending Background Check...",
                "brand": "Search"
            })
            
        # Pad with bing images if we need more to reach num_results
        if len(fast_results) < num_results:
            for img in bing_images[len(fast_results):num_results]:
                if not any(r.get("source_url") == img.get("source_url") for r in fast_results):
                    fast_results.append({
                        "name": img.get("name"),
                        "url": img.get("source_url"),
                        "source_url": img.get("source_url"),
                        "image_url": img.get("image_url"),
                        "price": "Pending Background Check...",
                        "brand": "Search"
                    })
        return fast_results

    async def run_deep_crawl_process(self, query, fast_bing_products):
        print(f"DEBUG: Starting background run_deep_crawl_process for {query}")
        urls = [p["source_url"] for p in fast_bing_products if p.get("source_url")]
        
        results = []
        if urls:
            print(f"DEBUG: Found {len(urls)} URLs. Starting advanced crawl for visibility...")
            pages = []
            from crawler import crawl_site
            async with AsyncWebCrawler() as crawler:
                for idx, url in enumerate(urls):
                    print(f"🚀 [CRAWL] ({idx+1}/{len(urls)}) -> {url}")
                    content, _ = await crawl_site(url, crawler=crawler)
                    if content:
                        pages.append(content)
            
            print(f"✅ [COMPLETE] Crawled {len(pages)} product pages successfully.")

            print(f"DEBUG: Fetched {len(pages)} pages. Starting parallel extraction...")
            extraction_tasks = []
            for idx, content in enumerate(pages):
                if not content or len(content) < 200: continue
                source_url = urls[idx] if idx < len(urls) else query
                
                clean = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL)
                clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL)
                clean = re.sub(r"<[^>]+>", " ", clean)
                extraction_tasks.append(self.extract_product_data(clean, query, base_url=source_url))
            
            if extraction_tasks:
                extracted_batches = await asyncio.gather(*extraction_tasks)
                for batch in extracted_batches:
                    results.extend(batch)

        # Fallback to the fast_bing_products for any URLs that failed to extract
        extracted_source_urls = [p.get("source_url") or p.get("url") for p in results]
        extracted_source_urls = [self._normalize_url(u) for u in extracted_source_urls if u]
        
        for fast_p in fast_bing_products:
            fast_url = self._normalize_url(fast_p.get("source_url"))
            if fast_url and fast_url not in extracted_source_urls:
                results.append(fast_p)

        # 2. Heuristic: If we still have too few results with images, pad with general images
        products_with_images = [p for p in results if p.get("image_url")]
        # 3. Process Images (S3 Upload & Filtering)
        from asset_processor import asset_processor
        if results:
            print(f"DEBUG: Processing {len(results)} extracted products for S3 upload and filtering...")
            results = asset_processor.process_product_images(results, category="retail", subcategory="live_search")
            # process_product_images modifies dictionaries in place and adds s3_image_url
            for p in results:
                if p.get("s3_image_url"):
                    p["image_url"] = p["s3_image_url"] # Ensure the primary image_url is the S3 one

        print(f"DEBUG: Finished get_product_data. Total combined items: {len(results)}")
        return results[:10]

    async def extract_product_data(self, content, target_category="relevant", base_url=None):
        truncated_content = content[:15000]
        prompt = (
            f"Extract products matching '{target_category}' from the text.\n"
            f"Return a JSON list of objects with: name, price, brand, image_url, url.\n"
            f"Text:\n{truncated_content}"
        )
        try:
            print(f"DEBUG: Extraction LLM call start (content length: {len(truncated_content)})")
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system="Return ONLY valid JSON list named 'products'.",
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if not response: return []
            data = self._safe_json_parse(response.content[0].text, "products")
            extracted = data if isinstance(data, list) else data.get("products", [])
            
            # NORMALIZE URLs using base_url
            if base_url:
                for p in extracted:
                    if p.get("image_url"):
                        p["image_url"] = urljoin(base_url, p["image_url"])
                    if p.get("url"):
                        p["url"] = urljoin(base_url, p["url"])
                    elif p.get("source_url"):
                        p["source_url"] = urljoin(base_url, p["source_url"])
            
            return extracted
        except Exception as e:
            print(f"Extraction error: {e}")
            return []

    async def live_search(self, query):
        prompt = f"Give a helpful answer for: {query}"
        try:
            print(f"DEBUG: Live search LLM start for {query}")
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            return response.content[0].text if response else "No result found."
        except Exception as e:
            print("Live search error:", e)
            return "Error fetching result."

    async def search_sources(self, query, intent="shopping", limit=10):
        system_msg = "You are a shopping expert." if intent == "shopping" else "You are a research expert."
        prompt = f"Find {limit} useful DIRECT product listing URLs for: {query}. Return ONLY a JSON list of strings."
        try:
            print(f"DEBUG: Search sources LLM start for {query}")
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if not response: return []
            text = response.content[0].text
            data = self._safe_json_parse(text, "urls")
            urls = data if isinstance(data, list) else data.get("urls", [])
            return [self._normalize_url(u) for u in urls if isinstance(u, str)]
        except Exception as e:
            print("Search error:", e)
            return []

    async def _fetch_page(self, session, url):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            print(f"DEBUG: Fetching URL: {url}")
            async with session.get(url, timeout=8, headers=headers) as res:
                if res.status == 200:
                    return await res.text()
                print(f"DEBUG: Fetch failed with status {res.status} for {url}")
                return ""
        except Exception as e:
            print(f"DEBUG: Fetch error for {url}: {e}")
            return ""

    def _normalize_url(self, url):
        if not url or not isinstance(url, str): return None
        url = url.strip()
        if url.startswith("//"): return "https:" + url
        if not url.startswith("http"): return "https://" + url.lstrip("/")
        return url

    def _safe_json_parse(self, text, key):
        try:
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", text, flags=re.DOTALL)
            return json.loads(text)
        except:
            try:
                match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
                if match: return json.loads(match.group(1))
            except:
                pass
            return {key: []}

    async def _call_with_retry(self, func_factory, retries=3):
        for i in range(retries):
            try:
                async with self.semaphore:
                    return await func_factory()
            except Exception as e:
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                if is_rate_limit and i < retries - 1:
                    wait_time = (5 ** i) + 2
                    print(f"Rate limited. Waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif i == retries - 1:
                    print(f"LLM call failed after {retries} retries: {e}")
                    return None
                else:
                    print(f"LLM error: {e}. Retrying...")
                    await asyncio.sleep(1)

    async def cache_and_store_products(self, products, crawler, query):
        """
        Background task to ingest live product data into the local vector store.
        """
        if not products:
            return

        print(f"\n🚀 [BACKGROUND] Starting caching for: {query}")
        print(f"📦 [BACKGROUND] Processing {len(products)} products...")
        
        try:
            from ingest import add_multiple_contents_to_store
            
            ingest_items = []
            for product in products:
                # Basic description formatting for RAG
                # We normalize keys to ensure compatibility with ingest.py
                source_url = product.get('source_url') or product.get('url') or "unknown"
                image_url = product.get('image_url')
                
                description = (
                    f"Product: {product.get('name')}\n"
                    f"Brand: {product.get('brand', 'Product')}\n"
                    f"Price: {product.get('price', 'Check Site')}\n"
                    f"Category: {product.get('category', 'retail')} / {product.get('subcategory', 'general')}\n"
                    f"Details: {product.get('details', 'No details available')}\n"
                    f"Image URL: {image_url}\n"
                    f"Source URL: {source_url}"
                )
                
                # Metadata for ChromaDB
                metadata = {
                    "source": source_url,
                    "type": "live_cache",
                    "image_url": image_url,
                    "s3_image_url": image_url, # Fallback if already uploaded
                    "name": product.get("name"),
                    "price": str(product.get("price") or "Check Site")
                }
                
                ingest_items.append({
                    "content": description,
                    "url": source_url,
                    "metadata": metadata
                })
            
            if ingest_items:
                await add_multiple_contents_to_store(ingest_items)
                print(f"✅ [BACKGROUND] Successfully cached {len(ingest_items)} products for '{query}'\n")
            
        except Exception as e:
            print(f"❌ [BACKGROUND] Error during caching: {e}")

kimi_service = KimiService()