import os
import json
import asyncio
import re
import aiohttp
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
        if any(x in q for x in ["car", "bike", "vehicle", "mileage"]):
            return "vehicle"
        if any(x in q for x in ["buy", "price", "shop", "laptop", "phone", "macbook", "iphone", "toy", "gift"]):
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

    async def get_product_data(self, query):
        print(f"DEBUG: Starting get_product_data for {query}")
        urls = await self.search_sources(query, intent="shopping")
        if not urls: 
            print("DEBUG: No URLs found via search_sources")
            return []

        print(f"DEBUG: Found {len(urls)} URLs. Fetching pages...")
        results = []
        async with aiohttp.ClientSession() as session:
            # Parallel Fetch
            fetch_tasks = [self._fetch_page(session, url) for url in urls]
            pages = await asyncio.gather(*fetch_tasks)

            print(f"DEBUG: Fetched {len(pages)} pages. Starting parallel extraction...")
            # Parallel Extraction
            extraction_tasks = []
            for html in pages:
                if not html or len(html) < 200: continue
                # Simple cleaning for token efficiency
                clean = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL)
                clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL)
                clean = re.sub(r"<[^>]+>", " ", clean)
                extraction_tasks.append(self.extract_product_data(clean, query))
            
            if extraction_tasks:
                extracted_batches = await asyncio.gather(*extraction_tasks)
                for batch in extracted_batches:
                    results.extend(batch)

        print(f"DEBUG: Finished get_product_data. Total products: {len(results)}")
        return results[:10]

    async def extract_product_data(self, content, target_category="relevant"):
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
            return data if isinstance(data, list) else data.get("products", [])
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

    async def search_sources(self, query, intent="shopping"):
        system_msg = "You are a shopping expert." if intent == "shopping" else "You are a research expert."
        prompt = f"Find 5 useful DIRECT product listing URLs for: {query}. Return ONLY a JSON list of strings."
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
        print(f"BACKGROUND: caching products for {query}")
        pass

kimi_service = KimiService()