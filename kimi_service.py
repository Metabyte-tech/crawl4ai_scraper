import os
import json
import asyncio
import re
import aiohttp
from bs4 import BeautifulSoup
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
    async def generate_execution_plan(self, query, template_id=None):
        """
        Breaks down a complex business request into a series of actionable steps.
        Matches the 'Agent task' UI seen in Accio.com.
        """
        system_prompt = (
            "You are an expert AI business architect. Your task is to break down a "
            "complex e-commerce request into a logical, multi-step execution plan. "
            "Provide exactly 3-5 sub-tasks that are clear, actionable, and cover research, "
            "analysis, and synthesis. Return a JSON list of strings."
        )
        
        user_prompt = f"Request: {query}\nTemplate ID: {template_id or 'general'}\n\nGenerate the execution plan."
        
        try:
            # Correctly use _call_with_retry instead of non-existent sem_call_llm
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            )
            if not response: return ["Analyze request", "Gather data from web", "Generate final report"]
            text = response.content[0].text
            # Extract list from response if LLM adds preamble
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return ["Analyze request", "Gather data from web", "Generate final report"]
        except Exception as e:
            print(f"Error generating plan: {e}", flush=True)
            return [f"Plan Error: {str(e)}"]

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
        words = set(re.findall(r'\b\w+\b', q))  # Use word boundaries for exact word matching
        
        # 🤖 Agent Task / Template execution detection
        # Templates usually start with specific "Professional" verbs or keywords
        agent_keywords = [
            "commercial feasibility", "market segment", "qualified suppliers", 
            "viral hits", "marketing strategy", "business architect", "report for"
        ]
        # Agent keywords can be multi-word phrases, so use substring match for them
        if any(x in q for x in agent_keywords):
            return "agent_task"

        # 🧸 Shopping / Products (Check this BEFORE vehicle to catch "car toys")
        shopping_keywords_exact = {
            "toy", "gift", "miniature", "lego", "puzzle", "doll"
        }
        if "remote control" in q or "rc car" in q or any(x in words for x in shopping_keywords_exact):
            return "shopping"

        # 🚗 Vehicle / Mobility
        vehicle_keywords = {
            "car", "bike", "vehicle", "mileage", "scooter", "truck",
            "suv", "sedan", "hatchback", "coupe", "ev", "thar", "mahindra", 
            "toyota", "honda", "hyundai", "kia", "maruti", "suzuki", "ford", 
            "chevrolet", "bmw", "mercedes", "audi", "volkswagen", "jeep", 
            "defender", "porsche", "ferrari", "lamborghini", "tata",
            "nexon", "creta", "innova", "fortuner", "scorpio", "bolero",
            "swift", "brezza", "ertiga", "baleno", "i20", "venue",
            "xuv", "compass", "duster", "kwid", "redi-go", "harrier"
        }
        if "electric car" in q or "range rover" in q or "land rover" in q or any(x in words for x in vehicle_keywords):
            return "vehicle"
        
        # ℹ️ Informational / General
        # If it contains informational words, it should be "general" even if it has product keywords
        info_words = {"how", "why", "who", "what", "where", "tell", "explain", "list", "history", "about", "meaning", "definition"}
        if any(x in words for x in info_words):
            return "general"

        # 🛒 Shopping / Products
        shopping_keywords = {
            "buy", "price", "shop", "laptop", "phone", "macbook", "iphone", 
            "toy", "gift", "tshirt", "t-shirt", "shirt", "shoes", "shoe", 
            "cloth", "clothing", "wear", "jean", "pant", "fashion", "brand",
            "electronics", "gadget", "watch", "accessory", "bottle", "glass",
            "box", "bag", "lunch", "home", "kitchen", "furniture", "book", 
            "tool", "beauty", "care", "health", "product", "item", "unit", "set",
            "chair", "desk", "lamp", "lighting", "find",
            "certificat", "customizable", "logo", "hires"
        }
        
        if "new hires" in q or "under $" in q or "under ₹" in q or any(x in words for x in shopping_keywords):
            return "shopping"
        
        # 🔍 Heuristic for short product-like queries (e.g. "milk glass bottle 90ml")
        # If it's a short query with no info words (already checked above), it's likely a product search
        q_words = q.split()
        if 1 <= len(q_words) <= 10:  # Increased from 5 to 10 to catch longer descriptive product queries
            # Avoid misclassifying greetings
            greetings = {"hi", "hello", "hey", "hola", "namaste", "thanks", "ok"}
            if len(q_words) == 1 and q_words[0] in greetings:
                return "general"
            return "shopping"
            
        return "general"

    async def get_vehicle_data(self, query):
        q = query.lower()
        if "image" in q:
            return await self.search_images(q)

        prompt = f"""You are an automotive expert. Give key specs for the vehicle: "{query}"
Use your best knowledge — even for newer Indian or regional models like Thar Rox, Nexon, Creta etc.
Return ONLY valid JSON with these fields (never return null — use "N/A" if unknown):
{{"name": "full model name", "price": "price range e.g. ₹15-18 Lakh", "mileage": "e.g. 18 kmpl", "fuel": "Petrol/Diesel/Electric"}}"""
        try:
            print(f"DEBUG: Vehicle LLM start for {query}", flush=True)
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    system=EXTRACTION_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if not response:
                return await self.search_images(query)
            text = response.content[0].text
            result = self._safe_json_parse(text, "vehicle")
            # If all key fields are None or "N/A", fall back to image search
            def is_empty(val):
                return not val or str(val).strip().upper() in ("N/A", "NONE", "NULL", "UNKNOWN", "-")
            if isinstance(result, dict) and all(is_empty(result.get(f)) for f in ["price", "mileage", "fuel"]):
                print(f"DEBUG: Vehicle LLM returned all N/A for {query}. Falling back to images.", flush=True)
                return await self.search_images(query)
            return result
        except Exception as e:
            print("Vehicle error:", e, flush=True)
            return await self.search_images(query)

    async def search_images(self, query):
        # 1. Smarter query cleaning: Remove filler words
        fillers = ["show me", "some", "images", "image", "of", "find", "search", "get", "pics", "pictures", "photos"]
        clean_query = query.lower()
        for f in fillers:                                                                                                                                                                                                                                                                                                                                                   
            clean_query = clean_query.replace(f, "")
        clean_query = clean_query.strip()
        
        if not clean_query: clean_query = query # Fallback

        print(f"DEBUG: Starting lightweight image search for: {clean_query}", flush=True)
        try:
            search_url = f"https://www.bing.com/images/search?q={clean_query.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    html_content = ""
                    if response.status == 200:
                        html_content = await response.text()
                    
                    # Extract both thumbnail URL (turl) and page URL (purl)
                    # Bing encodes JSON in data-m attribute - we want TURL (Bing Proxy) not MURL (Source blockable CDN)
                    # Use flexible independent extraction as order can vary
                    turls = re.findall(r'turl&quot;:&quot;(https?://.*?)&quot;', html_content)
                    purls = re.findall(r'purl&quot;:&quot;(https?://.*?)&quot;', html_content)
                    
                    blocks = []
                    for t, p in zip(turls, purls):
                        # Decode HTML entities like &amp; in URLs
                        t = t.replace("&amp;", "&")
                        p = p.replace("&amp;", "&")
                        blocks.append((t, p))

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
                        print(f"DEBUG: Found {len(real_results)} real images with source URLs from Bing.", flush=True)
                        return {
                            "type": "images",
                            "query": clean_query,
                            "results": real_results
                        }
                    else:
                        print("DEBUG: No real images found in Bing search result.", flush=True)
        except Exception as e:
            print(f"Image search error: {e}", flush=True)

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

    async def search_amazon_products(self, query, limit=10):
        """
        Scrape Amazon India search for immediate price, rating, and image data.
        This bypasses all per-page retailer blocking.
        """
        print(f"DEBUG: Searching Amazon for: {query}", flush=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        print(f"DEBUG: Amazon returned status {response.status}", flush=True)
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    products = []
                    for item in soup.select('[data-component-type="s-search-result"]')[:limit]:
                        name = item.select_one('h2 span')
                        price_w = item.select_one('.a-price-whole')
                        price_f = item.select_one('.a-price-fraction')
                        rating_el = item.select_one('.a-icon-alt')
                        reviews_el = item.select_one('.a-size-base.s-underline-text')
                        img_el = item.select_one('img.s-image')
                        link_el = item.select_one('h2 a')
                        
                        if not name: continue
                        
                        price_str = None
                        if price_w:
                            price_str = f"₹{price_w.get_text(strip=True)}"
                            if price_f:
                                fraction = price_f.get_text(strip=True)
                                if fraction and fraction != "00":
                                    price_str += f".{fraction}"
                        
                        rating_val = None
                        if rating_el:
                            m = re.search(r'(\d+\.\d+)', rating_el.get_text())
                            if m: rating_val = float(m.group(1))
                        
                        reviews_count = None
                        if reviews_el:
                            m = re.search(r'([\d,]+)', reviews_el.get_text())
                            if m: reviews_count = m.group(1)
                        
                        product_url = f"https://www.amazon.in{link_el.get('href', '')}" if link_el else url
                        
                        products.append({
                            "name": name.get_text(strip=True),
                            "price": price_str or "Check Site",
                            "rating_avg": rating_val,
                            "rating_count": reviews_count,
                            "image_url": img_el.get("src") if img_el else None,
                            "url": product_url,
                            "source_url": product_url,
                            "source": "Amazon India",
                            "brand": "Amazon India"
                        })
                    
                    print(f"DEBUG: Amazon returned {len(products)} products with prices.", flush=True)
                    return products
        except Exception as e:
            print(f"DEBUG: Amazon scrape error: {e}", flush=True)
            return []

    async def search_ebay_products(self, query, limit=6):
        """
        Scrape eBay global search for products with USD prices.
        eBay is globally available and does not geo-block, making it an
        excellent complement to Amazon India for worldwide coverage.
        """
        print(f"DEBUG: Searching eBay for: {query}", flush=True)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/"
        }
        url = f"https://www.ebay.com/sch/i.html?_nkw={query.replace(' ', '+')}&_sacat=0"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        print(f"DEBUG: eBay returned status {response.status}", flush=True)
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    products = []
                    # eBay uses li.s-card for its product cards
                    for item in soup.find_all('li', class_=lambda c: c and 's-card' in c)[:limit + 10]:
                        title_elem = item.find(['h3', 'div'], class_=re.compile(r'title|name'))
                        if not title_elem: continue
                        
                        name_text = title_elem.get_text(strip=True)
                        if name_text in ("Shop on eBay", "", "New Listing") or "Opens in a new window" in name_text and len(name_text) < 30: 
                            continue
                            
                        # Clean up eBay's screen reader text
                        name_text = name_text.replace("Opens in a new window or tab", "").strip()
                        
                        text = item.get_text()
                        prices = re.findall(r'\$[\d,.]+', text)
                        img = item.find('img')
                        link = item.find('a')
                        
                        if prices and len(prices) > 0:
                            products.append({
                                "name": name_text[:70],
                                "price": prices[0],  # Take the first price found
                                "rating_avg": None,
                                "image_url": img.get("src") if img else None,
                                "url": link.get("href") if link else url,
                                "source_url": link.get("href") if link else url,
                                "source": "eBay",
                                "brand": "eBay",
                                "details": ""
                            })
                        if len(products) >= limit: break
                    
                    print(f"DEBUG: eBay returned {len(products)} products with prices.", flush=True)
                    return products
        except Exception as e:
            print(f"DEBUG: eBay scrape error: {e}", flush=True)
            return []
    
    async def get_fast_bing_data(self, query, num_results=10):
        print(f"DEBUG: Starting get_fast_bing_data for {query}", flush=True)
        # 1. Run Amazon India + eBay (global) + Image Lookup in parallel
        amazon_task = self.search_amazon_products(query, limit=num_results)
        ebay_task = self.search_ebay_products(query, limit=5)  # global USD prices
        images_task = self.search_images(query)
        ddg_task = self.search_sources(query, limit=5)  # supplementary
        
        amazon_products, ebay_products, images_res, ddg_results = await asyncio.gather(
            amazon_task, ebay_task, images_task, ddg_task
        )
        bing_images = images_res.get("results", []) if images_res else []
        print(f"DEBUG: Combined sources — Amazon: {len(amazon_products)}, eBay: {len(ebay_products)}", flush=True)
        
        # 2. Build fast_results: Interleave Amazon India and eBay so both get fair visibility
        fast_results = []
        all_live_products = []
        max_len = max(len(amazon_products), len(ebay_products))
        for i in range(max_len):
            if i < len(amazon_products):
                all_live_products.append(amazon_products[i])
            if i < len(ebay_products):
                all_live_products.append(ebay_products[i])
        
        for idx, product in enumerate(all_live_products[:num_results]):
            # Use Bing images for richer visuals if available, else source thumbnail
            img_url = bing_images[idx].get("image_url") if idx < len(bing_images) else product.get("image_url")
            
            fast_results.append({
                "name": product["name"],
                "url": product["url"],
                "source_url": product["source_url"],
                "image_url": img_url or product.get("image_url"),
                "price": product["price"],
                "rating_avg": product["rating_avg"],
                "rating_count": product.get("rating_count"),
                "brand": product["brand"],
                "source": product["source"],
                "details": f"{product['name']} - {product.get('price', '')} on Amazon India"
            })
        
        # 3. Supplement with DDG results to fill up to num_results
        if ddg_results and len(fast_results) < num_results:
            for idx, res in enumerate(ddg_results):
                if len(fast_results) >= num_results: break
                url = res["url"]
                domain = urlparse(url).netloc.lower()
                store_name = domain.replace("www.", "").split('.')[0].capitalize()
                
                # --- CRITICAL FIX ---
                # Do NOT include DDG fallback links for domains we already scrape natively!
                # This prevents "Check Site" duplicate cards for eBay or Amazon.
                if any(native in domain for native in ['amazon', 'ebay']):
                    continue
                
                img_idx = len(fast_results)
                img_url = bing_images[img_idx].get("image_url") if img_idx < len(bing_images) else None
                
                # Skip if URL already in fast_results
                existing_urls = {self._normalize_url(r["url"]) for r in fast_results}
                if self._normalize_url(url) in existing_urls: continue
                
                snippet = res.get("snippet", "")
                
                # --- ROBUST PRICE EXTRACTION ---
                price = "Check Site"
                
                # 1. Multi-currency and Alphanumeric patterns (e.g. $49, USD 50, 49.99 CAD, from ₹100)
                price_patterns = [
                    r'([$₹£€]\s?\d{1,7}(?:[.,]\d{2})?)',           # Standard: $19.99
                    r'(\d{1,7}(?:[.,]\d{2})?\s?[$₹£€])',           # Reverse: 19.99$
                    r'(?:USD|INR|GBP|EUR)\s?(\d{1,7}(?:[.,]\d{2})?)', # ISO: USD 19.99
                    r'from\s?([$₹£€]\s?\d{1,7})',                  # Range: from $10
                ]
                
                for pattern in price_patterns:
                    m = re.search(pattern, snippet, re.I)
                    if m:
                        price = m.group(0)
                        break
                
                # 2. Store-Specific Heuristics (if regex fails)
                if price == "Check Site":
                    if "best buy" in store_name.lower() or "bestbuy" in domain:
                        # Best Buy snippets often have "Price: $..."
                        m = re.search(r'price[:\s]+([\$\d\.]+)', snippet, re.I)
                        if m: price = m.group(1)
                    elif "wayfair" in store_name.lower():
                        # Wayfair often says "at Wayfair for $..."
                        m = re.search(r'for\s+([\$\d\.]+)', snippet, re.I)
                        if m: price = m.group(1)
                    elif "staples" in store_name.lower():
                        m = re.search(r'only\s+([\$\d\.]+)', snippet, re.I)
                        if m: price = m.group(1)
                
                fast_results.append({
                    "name": res.get("title", f"{query.title()} from {store_name}"),
                    "url": url,
                    "source_url": url,
                    "image_url": img_url,
                    "price": price,
                    "rating_avg": None,
                    "brand": store_name,
                    "source": store_name,
                    "details": snippet
                })
            
        # Pad with bing images if we need more
        if len(fast_results) < num_results:
            for img in bing_images:
                if len(fast_results) >= num_results: break
                img_src = img.get("source_url")
                if not any(self._normalize_url(r.get("source_url") or r.get("url")) == self._normalize_url(img_src) for r in fast_results):
                    fast_results.append({
                        "name": img.get("name"),
                        "url": img_src,
                        "source_url": img_src,
                        "image_url": img.get("image_url"),
                        "price": "Check Price",
                        "brand": "Verifying...",
                        "source": "Image Search",
                        "details": f"High-quality {query} found via visual search. Click for full details and pricing."
                    })
        return fast_results

    async def run_deep_crawl_process(self, query, fast_bing_products):
        print(f"DEBUG: Starting background run_deep_crawl_process for {query}", flush=True)
        urls_with_images = {self._normalize_url(p["source_url"]): p.get("image_url") for p in fast_bing_products if p.get("source_url")}
        urls = list(urls_with_images.keys())
        
        results = []
        if urls:
            from crawler import crawl_site
            from crawl4ai import AsyncWebCrawler
            
            # Use Semaphore to limit parallel browser tabs (avoid memory crashes)
            semaphore = asyncio.Semaphore(3)
            
            async def crawl_and_extract_task(url, crawler):
                async with semaphore:
                    try:
                        print(f"🚀 [CRAWL] Start -> {url}", flush=True)
                        content, _ = await crawl_site(url, crawler=crawler)
                        if not content or len(content) < 200:
                            return []
                        
                        clean = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL)
                        clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL)
                        clean = re.sub(r"<[^>]+>", " ", clean)
                        
                        extracted = await self.extract_product_data(clean, query, base_url=url)
                        
                        # MERGE LOGIC: If extracted product has no image (or bad image), use the one from Bing
                        for p in extracted:
                            norm_url = self._normalize_url(p.get("url") or p.get("source_url"))
                            img = p.get("image_url")
                            is_valid_img = img and str(img).startswith("http") and any(ext in str(img).lower() for ext in [".jpg", ".jpeg", ".png", ".webp", ".avif"])
                            
                            if not is_valid_img:
                                # Fallback to the image found in the fast path for this domain/url
                                p["image_url"] = urls_with_images.get(norm_url) or urls_with_images.get(url)

                        return extracted
                    except Exception as e:
                        print(f"ERROR: Deep crawl/extract failed for {url}: {e}", flush=True)
                        return []

            async with AsyncWebCrawler() as crawler:
                tasks = [crawl_and_extract_task(u, crawler) for u in urls]
                batch_results = await asyncio.gather(*tasks)
                for batch in batch_results:
                    results.extend(batch)

        # Fallback to the fast_bing_products for any URLs that failed to extract
        extracted_source_urls = [self._normalize_url(p.get("source_url") or p.get("url")) for p in results]
        
        for fast_p in fast_bing_products:
            fast_url = self._normalize_url(fast_p.get("source_url"))
            if fast_url and fast_url not in extracted_source_urls:
                results.append(fast_p)

        # 3. Process Images (S3 Upload & Filtering)
        from asset_processor import asset_processor
        if results:
            # Process images (synchronous call)
            results = asset_processor.process_product_images(results, category="retail", subcategory="live_search")
            for p in results:
                if p.get("s3_image_url"):
                    p["image_url"] = p["s3_image_url"] 

        return results[:10]

    async def extract_product_data(self, content, target_category="relevant", base_url=None):
        # PRO robust cleaning with BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # 1. Try to find structured data (JSON-LD) which is more reliable for Price/Rating
        structured_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, list): structured_data.extend(data)
                else: structured_data.append(data)
            except: pass
            
        # 2. Extract meta tags for price/rating
        meta_data = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower() or meta.get("property", "").lower()
            if any(x in name for x in ["price", "rating", "brand", "availability", "og:title"]):
                meta_data[name] = meta.get("content")

        # 3. Remove non-content elements
        for element in soup(["script", "style", "svg", "iframe", "canvas", "noscript", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        # 4. Remove common ad/nav containers by class/id
        for container in soup.find_all(attrs={"class": re.compile(r'nav|footer|sidebar|ad-|promo|header|menu|social|comment', re.I)}):
            container.decompose()

        # 5. Get clean text with structure preserved
        clean_text = soup.get_text(separator=' ', strip=True)
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Combine everything for the LLM
        content_summary = f"""
        URL: {base_url}
        META DATA: {json.dumps(meta_data)}
        STRUCTURED DATA: {json.dumps(structured_data)[:5000]} 
        PAGE TEXT: {clean_text[:50000]}
        """
        content = content_summary.strip()
        
        # SMART START: Search for prices or "Results" to skip the header
        # Skip small price markers (like currency switchers) by looking for the first price in a longer string
        start_idx = 0
        price_match = re.search(r'[\$£€₹]\d+', content) # Look for currency followed by digits
        if price_match:
            # Start 500 characters before the first price marker
            start_idx = max(0, price_match.start() - 500)
            print(f"DEBUG: Smart Start triggered at index {start_idx}", flush=True)
        
        truncated_content = content[start_idx : start_idx + 80000] 
        prompt = (
            f"Extract ALL product details for '{target_category}' from the provided text.\n"
            f"Focus on finding specific technical specs, prices, and ratings.\n\n"
            f"Return a JSON list of objects with these exact fields:\n"
            f"- name: Concise, descriptive product name (include model/size if found)\n"
            f"- price: The numerical price with currency (e.g., $19.99). Look for strings near 'Add to cart', 'MSRP', total, or large bold numbers. Prioritize sale prices. If completely absent, return null.\n"
            f"- brand: Brand name\n"
            f"- rating_avg: Numerical average rating (e.g., 4.5) - float or null\n"
            f"- rating_count: total number of customer reviews (e.g., 1250) - integer or null\n"
            f"- offers: Short summary of discounts or free shipping\n"
            f"- source: Store name or platform\n"
            f"- image_url: Direct link to the primary product image found in the text or metadata snippets.\n"
            f"- url: Original product URL\n"
            f"- moq: Minimum Order Quantity (e.g., '100 units' or '1 pc')\n"
            f"- details: A HIGHLY DETAILED summary of features, materials, and specifications.\n"
            f"- advantages: A list of 2-3 main pros or advantages of the product\n"
            f"- disadvantages: A list of 2-3 main cons or disadvantages of the product\n"
            f"- reviews: A list of 3-5 REAL user comments found in the text. Format: {{\"user\": \"name\", \"comment\": \"text\", \"rating\": 5}}\n"
            f"\nText to analyze:\n{truncated_content}"
        )
        try:
            print(f"DEBUG: Extraction LLM call start (content length: {len(truncated_content)})", flush=True)
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
                    # Handle LLM hallucinations like "Not found" or null
                    p_url = p.get("url") or p.get("source_url")
                    if not p_url or str(p_url).lower() in ("not found", "null", "none", "n/a"):
                        p["url"] = base_url
                        if "source_url" in p: p["source_url"] = base_url
                    
                    if p.get("image_url") and str(p.get("image_url")).lower() not in ("not found", "null", "none", "n/a"):
                        p["image_url"] = urljoin(base_url, p["image_url"])
                    
                    # Ensure final URLs are normalized
                    if p.get("url"):
                        p["url"] = urljoin(base_url, p["url"])
                    if p.get("source_url"):
                        p["source_url"] = urljoin(base_url, p["source_url"])
            
            return extracted
        except Exception as e:
            print(f"Extraction error: {e}", flush=True)
            return []

    def _get_category_prompt(self, template_id):
        base_prompt = (
            "You are an Elite AI Business Agent. Your goal is to provide a highly "
            "professional, data-driven Commercial Report. Use Markdown formatting.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "1. CITATIONS: Use inline citations [1], [2] etc. whenever you reference specific data from the provided MARKET CONTEXT.\n"
            "2. CHECKLISTS: Use emoji-checklists (e.g., ✅, 📋) for actionable execution steps or requirements.\n"
            "3. VISUALS: Use tables and carousels where instructed to make the report scannable.\n\n"
        )
        
        # Determine category from templates.json
        category_id = "product_research" # default
        import json
        try:
            with open("templates.json", "r") as f:
                data = json.load(f)
                for cat in data.get("categories", []):
                    for t in cat.get("templates", []):
                        if t.get("id") == template_id:
                            category_id = cat.get("id")
                            break
        except Exception as e:
            print(f"Error loading templates category: {e}", flush=True)

        if category_id == "business_analysis":
            return base_prompt + (
                "REPORT STRUCTURE:\n"
                "1. Market Analysis (Sales growth, market size)\n"
                "2. Scenario Breakdown (Financial modeling)\n"
                "3. Strategic Recommendations\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "Use heavily formatted Markdown TABLES with at least 4 columns to compare financials and scenarios. "
                "Include 'Summary Scorecards' using bold markdown numbers at the top.\n"
                "If MARKET CONTEXT is provided, base your data strictly on it."
            )
        elif category_id == "product_design":
            return base_prompt + (
                "REPORT STRUCTURE:\n"
                "1. Concept Visuals (Image Grids)\n"
                "2. Design Iterations\n"
                "3. Material Suggestions\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "For 'Concept Visuals', present 3 specific product design ideas. "
                "For each concept, MUST include a dynamically generated image using exactly: "
                "![Concept Name](https://image.pollinations.ai/prompt/hyper-realistic%20product%20photo%20of%20[detailed-description]?width=800&height=400&nologo=true) "
                "(replace [detailed-description] with URL-encoded design specs). Below each, add bullet points."
            )
        elif category_id == "supplier_sourcing":
            return base_prompt + (
                "REPORT STRUCTURE:\n"
                "1. Sourcing List (Visual Product Grid)\n"
                "2. Comparison Chart\n"
                "3. Manufacturer Audit\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "AT THE VERY TOP, you MUST output a `<product_grid>` tag combining the MARKET CONTEXT into exactly this JSON format: "
                "[{\"name\": \"...\", \"price\": \"...\", \"brand\": \"...\", \"image_url\": \"...\", \"source_url\": \"...\", \"details\": \"...\", \"moq\": \"...\", \"supplier_years\": \"...\", \"location\": \"...\", \"is_verified\": true}]. "
                "For 'is_verified', set to true if the source is a known reliable platform. "
                "Then below it, write a massive Markdown Spec Comparison Table mapping requirements side-by-side using citations [1][2]."
            )
        elif category_id == "go_to_market":
            return base_prompt + (
                "REPORT STRUCTURE:\n"
                "1. Product Title & Positioning\n"
                "2. A+ Content Copy / Ad Copy\n"
                "3. Listing Variations\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "Provide text blocks formatted for copy-pasting (Markdown Code blocks). "
                "Use pollinations AI markdown image syntax to produce 'Image Galleries' showing lifestyle vs detail shots: "
                "![Lifestyle Shot](https://image.pollinations.ai/prompt/lifestyle%20shot%20of%20[product]?width=800&height=400&nologo=true)."
            )
        else:
            # product_research and default
            return base_prompt + (
                "REPORT STRUCTURE:\n"
                "1. Top Selling Trends\n"
                "2. Success Factors\n"
                "3. Gap Analysis & Innovation Concepts\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "For 'Gap Analysis', present 2-3 specific product innovation ideas. "
                "For each, MUST include an inline image using this EXACT markdown format: "
                "![Concept Name](https://image.pollinations.ai/prompt/hyper-realistic%20product%20photo%20of%20[detailed-description]?width=800&height=400&nologo=true). "
                "Below the image, list 'Key Features' in bullet points. Use Bar Chart ascii simulations or Tables for trends."
            )

    async def generate_agent_report(self, query, template_id=None, subject=None):
        """
        Generates a professional, structured business report for an Agent Task.
        Dynamically adjusts formatting to match Accio category standards.
        """
        search_query = subject if subject else query

        # 1. Try local database first (strict relevance check)
        context_docs = []
        try:
            from query import fast_query
            # Use tighter threshold (0.9) to avoid unrelated items
            all_docs = fast_query(search_query, category="retail", threshold=0.9, k=10)

            # Keyword relevance filter: ensure returned docs actually match the subject
            subject_keywords = set(search_query.lower().split())
            stop_words = {"for", "a", "an", "the", "of", "in", "to", "and", "with", "on", "at", "from"}
            subject_keywords -= stop_words

            for doc, score in all_docs:
                doc_text = (doc.page_content + " " + str(doc.metadata.get("name", ""))).lower()
                if any(kw in doc_text for kw in subject_keywords):
                    context_docs.append((doc, score))

            print(f"RAG: {len(all_docs)} raw → {len(context_docs)} relevant for '{search_query}'", flush=True)
        except Exception as e:
            print(f"RAG search failed for report: {e}", flush=True)

        # 2. If local DB has no relevant data, handle gracefully
        if not context_docs:
            print(f"No relevant local data for '{search_query}'. Using empty context.", flush=True)
            context_text = "MARKET CONTEXT FROM LOCAL DATABASE:\n[NO LOCAL DATA WAS FOUND FOR THIS PRODUCT. GENERATE THE REPORT BASED ON YOUR OWN KNOWLEDGE BUT MENTION THAT LOCAL SUPPLIER DATA IS UNAVAILABLE.]\n"
        else:
            # Build context from local DB results
            context_text = "MARKET CONTEXT FROM LOCAL DATABASE:\n"
            for doc, score in context_docs:
                meta = doc.metadata
                img = meta.get('image_url') or meta.get('s3_image_url') or "https://via.placeholder.com/150"
                moq = meta.get('moq', 'N/A')
                loc = meta.get('location', 'N/A')
                yrs = meta.get('supplier_years', 'N/A')
                context_text += (
                    f"- {meta.get('name')} | Price: {meta.get('price')} | Brand: {meta.get('brand')} | "
                    f"MoQ: {moq} | Location: {loc} | Yrs: {yrs} | "
                    f"Image URL: {img} | URL: {meta.get('url') or meta.get('source_url')}\n"
                )

        system_prompt = self._get_category_prompt(template_id)
        user_prompt = f"{context_text}\n\nTemplate ID: {template_id or 'General Analysis'}\n\nTask: {query}"

        try:
            print(f"DEBUG: Generating Agent Report for {template_id}", flush=True)
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            )
            return response.content[0].text if response else "Failed to generate report."
        except Exception as e:
            print(f"Error generating agent report: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return f"BACKEND_ERROR: {str(e)}"

    async def rapid_extract_price_and_rating(self, session, url):
        """
        Hyper-fast extraction using raw HTML via aiohttp.
        Targets JSON-LD and Meta tags specifically.
        """
        try:
            # Enhanced headers to avoid "Bot Detection" on top retailers
            symbols = {"$": "$", "USD": "$", "RS": "₹", "INR": "₹", "GBP": "£", "EUR": "€"}
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/"
            }
            async with session.get(url, headers=headers, timeout=5) as response:
                if response.status != 200: 
                    print(f"DEBUG: Rapid extract failed for {url} with status {response.status}", flush=True)
                    return url, None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                data = {"price": None, "rating": None, "description": None}
                
                # 1. Structured Data (JSON-LD) - More robust traversal
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        content = script.string
                        if not content: continue
                        ld = json.loads(content)
                        
                        # Handle both single objects and lists/graphs
                        items = ld if isinstance(ld, list) else [ld]
                        if isinstance(ld, dict) and "@graph" in ld: items = ld["@graph"]
                        
                        for item in items:
                            if not isinstance(item, dict): continue
                            
                            # Look for AggregateRating
                            rate = item.get("aggregateRating")
                            if isinstance(rate, dict):
                                data["rating"] = rate.get("ratingValue") or rate.get("value")
                            
                            # Look for Offers
                            offers = item.get("offers")
                            if offers:
                                if isinstance(offers, list): offers = offers[0]
                                if isinstance(offers, dict):
                                    price = offers.get("price") or offers.get("lowPrice")
                                    curr = offers.get("priceCurrency")
                                    if price:
                                        data["price"] = f"{symbols.get(curr, '$')}{price}" if curr else str(price)
                                        if data["price"] and not any(s in str(data["price"]) for s in symbols.values()):
                                            data["price"] = f"${data['price']}"
                    except: pass
                
                # 2. Meta Tags (Extensive list for top retailers)
                if not data["price"]:
                    meta_selectors = [
                        ("property", "product:price:amount"),
                        ("property", "og:price:standard_amount"),
                        ("name", "twitter:data1"),
                        ("property", "price")
                    ]
                    for attr, val in meta_selectors:
                        tag = soup.find("meta", {attr: val})
                        if tag and tag.get("content"):
                            data["price"] = tag.get("content")
                            # Add symbol if naked
                            if data["price"] and not any(s in str(data["price"]) for s in symbols.values()):
                                data["price"] = f"${data['price']}"
                            break
                
                if not data["rating"]:
                    meta_r = soup.find("meta", property="og:rating") or soup.find("meta", name="rating")
                    if meta_r: data["rating"] = meta_r.get("content")
                
                # 3. Simple description
                meta_desc = soup.find("meta", name="description") or soup.find("meta", property="og:description")
                if meta_desc: data["description"] = meta_desc.get("content")[:500]
                
                print(f"DEBUG: Rapid extracted data for {url}: {data['price']}, {data['rating']}", flush=True)
                return url, data
        except Exception as e:
            print(f"DEBUG: Error in rapid extract for {url}: {e}", flush=True)
            return url, None

    async def live_search(self, query):
        prompt = f"Give a helpful answer for: {query}"
        try:
            print(f"DEBUG: Live search LLM start for {query}", flush=True)
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
            print("Live search error:", e, flush=True)
            
    async def search_sources(self, query, intent="shopping", limit=10):
        """
        Real-time lightweight DuckDuckGo Search for products and snippets via HTTPr.
        """
        print(f"DEBUG: Starting real-time DDG search_sources for: {query}", flush=True)
        search_results = []
        try:
            # Using DuckDuckGo HTML (lite) for easy scraping
            from urllib.parse import quote_plus
            top_sites = "(site:amazon.com OR site:amazon.in OR site:walmart.com OR site:target.com OR site:bestbuy.com OR site:croma.com OR site:flipkart.com)"
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query + ' ' + top_sites)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        soup = BeautifulSoup(content, 'html.parser')
                        # DDG HTML selectors
                        for result in soup.select('.result'):
                            title_el = result.select_one('.result__title a')
                            snippet_el = result.select_one('.result__snippet')
                            if title_el and snippet_el:
                                url = title_el.get('href')
                                if not url: continue
                                # Clean redirect URLs from DDG if needed
                                if "/l/?" in url:
                                    from urllib.parse import parse_qs
                                    parsed = urlparse(url)
                                    url = parse_qs(parsed.query).get('uddg', [url])[0]
                                
                                search_results.append({
                                    "url": url,
                                    "title": title_el.get_text(strip=True),
                                    "snippet": snippet_el.get_text(strip=True)
                                })
                            if len(search_results) >= limit: break
            
            if search_results:
                print(f"DEBUG: Found {len(search_results)} real search results from DDG.", flush=True)
                return search_results
                
        except Exception as e:
            print(f"Real-time search failed: {e}. Falling back to image-source URLs.", flush=True)

        # Fallback to images if search crawl fails
        image_results = await self.search_images(query)
        fallback = []
        for r in image_results.get("results", []):
            if r.get("source_url"):
                fallback.append({
                    "url": r.get("source_url"),
                    "title": r.get("name", query),
                    "snippet": f"Product from {r.get('source_url')}"
                })
        return fallback[:limit]

    async def _fetch_page(self, session, url):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
            print(f"DEBUG: Fetching URL: {url}", flush=True)
            async with session.get(url, timeout=8, headers=headers) as res:
                if res.status == 200:
                    return await res.text()
                print(f"DEBUG: Fetch failed with status {res.status} for {url}", flush=True)
                return ""
        except Exception as e:
            print(f"DEBUG: Fetch error for {url}: {e}", flush=True)
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
                    print(f"Rate limited. Waiting {wait_time}s...", flush=True)
                    await asyncio.sleep(wait_time)
                elif i == retries - 1:
                    print(f"LLM call failed after {retries} retries: {e}", flush=True)
                    return None
                else:
                    print(f"LLM error: {e}. Retrying...", flush=True)
                    await asyncio.sleep(1)

    async def cache_and_store_products(self, products, query):
        """
        Background task to ingest live product data into the local vector store.
        """
        if not products:
            return

        print(f"\n🚀 [BACKGROUND] Starting caching for: {query}", flush=True)
        print(f"📦 [BACKGROUND] Processing {len(products)} products...", flush=True)
        
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
                    "category": "retail",
                    "image_url": image_url,
                    "s3_image_url": image_url, 
                    "name": product.get("name"),
                    "price": str(product.get("price") or "Check Site"),
                    "brand": product.get("brand") or "Product",
                    "rating_avg": str(product.get("rating_avg") or ""),
                    "rating_count": str(product.get("rating_count") or ""),
                    "offers": str(product.get("offers") or ""),
                    "store_source": product.get("source") or "Search",
                    "reviews": json.dumps(product.get("reviews") or []),
                    "moq": str(product.get("moq") or "1 pc"),
                    "location": str(product.get("location") or "Global"),
                    "supplier_years": str(product.get("supplier_years") or "Verifying..."),
                    "details": product.get("details") or ""
                }
                
                ingest_items.append({
                    "content": description,
                    "url": source_url,
                    "metadata": metadata
                })
            
            if ingest_items:
                await add_multiple_contents_to_store(ingest_items)
                print(f"✅ [BACKGROUND] Successfully cached {len(ingest_items)} products for '{query}'\n", flush=True)
            
        except Exception as e:
            print(f"❌ [BACKGROUND] Error during caching: {e}", flush=True)

kimi_service = KimiService()