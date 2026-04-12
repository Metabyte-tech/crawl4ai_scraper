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
        self.semaphore = asyncio.Semaphore(6) # Global limit for parallel Playwright browsers
        self.base_retail_domains = [
            "amazon.com", "amazon.in", "flipkart.com", "ebay.com", "walmart.com"
        ]
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
        ]

    def _get_stealth_headers(self, domain="google.com"):
        import random
        return {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": f"https://{domain}/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }

    def _parse_price(self, price_str):
        """Extract numeric value from a price string like '$19.99' or '₹1,299'."""
        if not price_str or "Check" in str(price_str) or "Verifying" in str(price_str):
            return float('inf')
        try:
            # Try to find currency + number pattern first (e.g. "for $19.99")
            m = re.search(r'[\$₹]\s*([\d,.]+)', str(price_str))
            if m:
                clean = m.group(1).replace(',', '')
            else:
                # Fallback: remove commas and currency symbols, keep first numeric group
                clean = re.sub(r'[^\d.]', '', str(price_str).replace(',', ''))
            
            # If multiple dots (e.g. from bad extraction), take the first one
            if clean.count('.') > 1:
                parts = clean.split('.')
                clean = parts[0] + "." + parts[1]
            return float(clean) if clean and any(c.isdigit() for c in clean) else float('inf')
        except:
            return float('inf')

    def _extract_price_from_snippet(self, snippet, domain=None, store_name=None):
        """
        Unified robust price extraction from DuckDuckGo/Bing snippets.
        Handles multi-currency, original vs current price, and store-specific patterns.
        """
        if not snippet:
            return "Check Site"
            
        snippet_lower = snippet.lower()
        
        # 0. Priority: Handle "current price" or "now" keywords which often appear in Google snippets
        current_match = re.search(r'(?:current price|now|today|only|save)\s*[:\-]?\s*([$₹£€]\s?\d{1,7}(?:[.,]\d{2})?)', snippet_lower)
        if current_match:
            return current_match.group(1).strip()

        # 1. Multi-currency and ISO patterns
        price_patterns = [
            r'([$₹£€]\s?\d{1,7}(?:[.,]\d{2})?)',           # Standard: $19.99
            r'(\d{1,7}(?:[.,]\d{2})?\s?[$₹£€])',           # Reverse: 19.99$
            r'(?:USD|INR|GBP|EUR)\s?(\d{1,7}(?:[.,]\d{2})?)', # ISO: USD 19.99
            r'from\s?([$₹£€]\s?\d{1,7})',                  # Range: from $10
        ]
        
        found_prices = []
        for pattern in price_patterns:
            matches = re.findall(pattern, snippet, re.I)
            for m in matches:
                # Ensure it's not a junk match
                if any(c.isdigit() for c in m):
                    found_prices.append(m)
        
        if not found_prices:
            # 2. Store-Specific Heuristics (if standard patterns fail)
            if store_name and ("best buy" in store_name.lower() or "bestbuy" in str(domain).lower()):
                m = re.search(r'price[:\s]+([\$\d\.]+)', snippet, re.I)
                if m: return m.group(1)
            elif store_name and "wayfair" in store_name.lower():
                m = re.search(r'for\s+([\$\d\.]+)', snippet, re.I)
                if m: return m.group(1)
            return "Check Site"
            
        # 3. Smart selection (Original vs Current)
        # If multiple prices found, check for "was", "original", "list price" indicators near them
        if len(found_prices) > 1:
            # Simple heuristic: often the last mentioned price in a "was X now Y" snippet is the current one
            # or the one not preceded by "was"
            price_positions = []
            for p in set(found_prices):
                for m in re.finditer(re.escape(p), snippet):
                    # Check context before the match (last 20 chars)
                    context = snippet[max(0, m.start() - 20):m.start()].lower()
                    is_original = any(word in context for word in ["was", "original", "list", "save"])
                    price_positions.append({"price": p, "pos": m.start(), "is_original": is_original})
            
            # Prefer non-original prices, or the one closest to some high-intent keywords
            current_prices = [p for p in price_positions if not p["is_original"]]
            if current_prices:
                # Return the one with the lowest value among non-originals (common for deals)
                return min(current_prices, key=lambda x: self._parse_price(x["price"]))["price"]
            
        return found_prices[0]

    @staticmethod
    def _extract_brand(product_name: str, store_fallback: str = "") -> str:
        """
        Extract brand name from a product title.
        Heuristic: the brand is usually the first 1-2 capitalized words before
        common delimiters like '-', 'by', '|', ',', 'for', 'with'.
        Returns the store name as fallback if nothing meaningful is found.
        """
        if not product_name:
            return store_fallback
        
        # Common stop-words that are NOT brand names
        STOP = {
            "the", "a", "an", "and", "or", "for", "with", "in", "on", "at",
            "new", "best", "premium", "pack", "set", "lot", "combo", "men",
            "women", "kids", "adult", "size", "black", "white", "blue", "red"
        }
        
        # Split on common brand-product separators
        for sep in [' - ', ' | ', ' by ', ', ', ' for ', ' with ', ' & ']:
            if sep.lower() in product_name.lower():
                candidate = product_name.split(sep, 1)[0].strip()
                words = candidate.split()
                brand_words = []
                for w in words[:3]:  # max 3 words for a brand
                    clean = re.sub(r'[^a-zA-Z0-9\.\-]', '', w)
                    if clean.lower() not in STOP and len(clean) > 1:
                        brand_words.append(clean)
                if brand_words:
                    return " ".join(brand_words)
        
        # Fallback: take first 1-2 capitalized words from the title
        words = product_name.split()
        brand_words = []
        for w in words[:4]:
            clean = re.sub(r'[^a-zA-Z0-9\.\-]', '', w)
            if clean and clean[0].isupper() and clean.lower() not in STOP and len(clean) > 1:
                brand_words.append(clean)
                if len(brand_words) == 2:
                    break
        
        return " ".join(brand_words) if brand_words else store_fallback


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
            "toy", "gift", "miniature", "lego", "puzzle", "doll", "dolls", "car", "cars"
        }
        if "remote control" in q or "rc car" in q or any(x in words for x in shopping_keywords_exact):
            return "shopping"

        # 🚗 Vehicle / Mobility
        vehicle_keywords = {
            "bike", "vehicle", "mileage", "scooter", "truck",
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
            "certificat", "customizable", "logo", "hires",
            "nursery", "baby", "clothes", "clothing", "gown", "cheap", "affordable"
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
        # Use stealth headers to avoid 503
        headers = self._get_stealth_headers("www.amazon.in")
        url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=12) as response:
                    if response.status != 200:
                        print(f"DEBUG: Amazon returned status {response.status}", flush=True)
                        # Minimal retry logic if 503
                        if response.status == 503:
                            await asyncio.sleep(1)
                            headers = self._get_stealth_headers("www.bing.com")
                            async with session.get(url, headers=headers, timeout=12) as retry_res:
                                if retry_res.status == 200:
                                    html = await retry_res.text()
                                else:
                                    return []
                        else:
                            return []
                    else:
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
                        
                        # Fix 1: High-res Amazon image
                        raw_img = img_el.get("src") if img_el else None
                        img_url = re.sub(r'\._[^/]*\.', '.', raw_img) if (raw_img and "m.media-amazon.com" in raw_img) else raw_img
                        
                        products.append({
                            "name": name.get_text(strip=True),
                            "price": price_str or "Check Site",
                            "rating_avg": rating_val,
                            "rating_count": reviews_count,
                            "image_url": img_url,
                            "url": product_url,
                            "source_url": product_url,
                            "source": "Amazon India",
                            "brand": self._extract_brand(name.get_text(strip=True), "Amazon")
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
        headers = self._get_stealth_headers("www.ebay.com")
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
                        
                        # eBay image upscaling
                        raw_img = img.get("src") if img else None
                        img_url = re.sub(r's-l\d+', 's-l500', raw_img) if (raw_img and "s-l" in raw_img) else raw_img

                        if prices and len(prices) > 0:
                            products.append({
                                "name": name_text[:70],
                                "price": prices[0],  # Take the first price found
                                "rating_avg": None,
                                "image_url": img_url,
                                "url": link.get("href") if link else url,
                                "source_url": link.get("href") if link else url,
                                "source": "eBay",
                                "brand": self._extract_brand(name_text, "eBay"),
                                "details": ""
                            })
                        if len(products) >= limit: break
                    
                    print(f"DEBUG: eBay returned {len(products)} products with prices.", flush=True)
                    return products
        except Exception as e:
            print(f"DEBUG: eBay scrape error: {e}", flush=True)
            return []
    
    async def search_flipkart_products(self, query, limit=5):
        """
        Scrape Flipkart India search for competitive price data.
        """
        print(f"DEBUG: Searching Flipkart for: {query}", flush=True)
        headers = self._get_stealth_headers("www.google.co.in")
        url = f"https://www.flipkart.com/search?q={query.replace(' ', '%20')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    products = []
                    # Flipkart layout can vary; try common selectors
                    # Broaden selector to any div that looks like a product card
                    for item in (soup.select('div[data-id]') or soup.select('div._1AtVbE') or soup.select('div._1sdM6s')):
                        name_el = item.select_one('a.atJtCj') or item.select_one('div._4rR01T') or item.select_one('a.IRpwTa') or item.select_one('a.s1Q9rs') or item.select_one('div._2Wk9S9')
                        price_el = item.select_one('div.QiMO5r') or item.select_one('div._30jeq3') or item.select_one('div._3I9_ca')
                        img_el = item.select_one('img')
                        link_el = item.select_one('a.atJtCj') or item.select_one('a')
                        
                        if not name_el or not price_el: continue
                        
                        name = name_el.get_text(strip=True)
                        if len(name) < 10: continue
                        
                        p_url = urljoin("https://www.flipkart.com", link_el.get('href', '')) if link_el else url
                        products.append({
                            "name": name,
                            "price": price_el.get_text(strip=True),
                            "rating_avg": None,
                            "image_url": img_el.get("src") if img_el else None,
                            "url": p_url,
                            "source_url": p_url,
                            "source": "Flipkart",
                            "brand": self._extract_brand(name, "Flipkart")
                        })
                        if len(products) >= limit: break
                    
                    print(f"DEBUG: Flipkart returned {len(products)} products.", flush=True)
                    return products
        except Exception as e:
            print(f"DEBUG: Flipkart scrape error: {e}", flush=True)
            return []

    async def search_walmart_products(self, query, limit=5):
        """
        Scrape Walmart global search for competitive price data.
        """
        print(f"DEBUG: Searching Walmart for: {query}", flush=True)
        headers = self._get_stealth_headers("www.walmart.com")
        url = f"https://www.walmart.com/search?q={query.replace(' ', '+')}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        return []
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    products = []
                    # Walmart results often hide in JSON-LD or specific grids
                    for item in soup.select('[data-testid="list-view-id"], [data-testid="grid-view-id"], .sans-serif'):
                        name_el = item.select_one('span[data-automation-id="product-title"]') or item.select_one('.w_V_o') or item.select_one('span.normal')
                        price_el = item.select_one('[data-automation-id="product-price"]') or item.select_one('.w_iUH7') or item.select_one('div.mr2')
                        img_el = item.select_one('img')
                        link_el = item.select_one('a')
                        
                        if not name_el: continue
                        name = name_el.get_text(strip=True)
                        if len(name) < 10 or "skip to" in name.lower(): continue
                        
                        # Extract price text manually
                        price_text = price_el.get_text(strip=True) if price_el else "Check Site"
                        if not any(c.isdigit() for c in price_text): price_text = "Check Site"
                        
                        p_url = urljoin("https://www.walmart.com", link_el.get('href', '')) if link_el else url
                        
                        products.append({
                            "name": name,
                            "price": price_text,
                            "rating_avg": None,
                            "image_url": img_el.get("src") if img_el else None,
                            "url": p_url,
                            "source_url": p_url,
                            "source": "Walmart",
                            "brand": "Walmart"
                        })
                        if len(products) >= limit: break
                    
                    print(f"DEBUG: Walmart returned {len(products)} products.", flush=True)
                    return products
        except Exception as e:
            print(f"DEBUG: Walmart scrape error: {e}", flush=True)
            return []

    def _get_scrape_date(self):
        import datetime
        return datetime.datetime.now().strftime("%Y-%m-%d")

    async def _archive_results(self, results, query, category="retail"):
        """Archive search results to S3 for history and audit."""
        from s3_service import s3_service
        import datetime
        import uuid
        date_str = self._get_scrape_date()
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        
        # Results can come from many sources; use 'aggregated' as folder
        filename = f"aggregated/{category}/{date_str}/results_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        s3_service.upload_data(results, filename)

    async def get_fast_bing_data(self, query, num_results=20):
        """
        Orchestrates parallel retail scrapers and search fallbacks.
        Enforces a strict global timeout for responsiveness.
        """
        # 1. Run all native scrapers in parallel with a strict 12s timeout
        amazon_task = self.search_amazon_products(query, limit=8)
        ebay_task = self.search_ebay_products(query, limit=8)
        flipkart_task = self.search_flipkart_products(query, limit=8)
        walmart_task = self.search_walmart_products(query, limit=8)
        ddg_task = self.search_sources(query, limit=8)
        
        images_task = self.search_images(query)
        
        print("DEBUG: Launching parallel scrapers...", flush=True)
        try:
            # We run images_task in parallel to backfill any missing retailer images
            amazon_products, ebay_products, flipkart_products, walmart_products, ddg_results, bing_res = await asyncio.wait_for(
                asyncio.gather(
                    amazon_task, ebay_task, flipkart_task, walmart_task, ddg_task, images_task
                ),
                timeout=12.0
            )
            bing_images = bing_res.get("results", []) if bing_res else []
        except asyncio.TimeoutError:
            print("WARNING: Fast path scrapers timed out! Returning partial/empty results to maintain low latency.", flush=True)
            amazon_products, ebay_products, flipkart_products, walmart_products, ddg_results = [], [], [], [], []
            bing_images = []
        except Exception as e:
            print(f"ERROR: Fast path gather failed: {e}", flush=True)
            amazon_products, ebay_products, flipkart_products, walmart_products, ddg_results = [], [], [], [], []
            bing_images = []
        
        # --- CLIENT REQUIREMENT: All Scraped Websites ---
        # If any native scraper failed (0 results), trigger a targeted site-specific search
        # as a backup to ensure that website is at least somewhat represented.
        async def fetch_domain_fallback(domain_name, site_url):
            """
            Native Recovery Strategy:
            1. Find product URLs via Bing Images (more reliable than web search).
            2. Scrape individual product pages for real JSON-LD prices in parallel.
            """
            print(f"DEBUG: Triggering Native Recovery fallback for {domain_name}", flush=True)
            
            # 1. Get URLs from Bing Images (proven to work)
            img_results = await self.search_images(f"site:{site_url} {query}")
            urls = []
            for r in img_results.get("results", []):
                if r.get("source_url") and site_url in r.get("source_url").lower():
                    urls.append((r.get("source_url"), r.get("image_url")))
            
            if not urls:
                print(f"DEBUG: Native Recovery failed to find URLs for {domain_name}", flush=True)
                return []
                
            # 2. Parallel price/rating extraction from product pages using own session
            async with aiohttp.ClientSession() as fallback_session:
                extraction_tasks = []
                for url, img_url in urls[:3]: # Limit to top 3 for speed
                    extraction_tasks.append(self.rapid_extract_price_and_rating(fallback_session, url))
                extraction_results = await asyncio.gather(*extraction_tasks)
            
            fallback_items = []
            for (url, data), (orig_url, img_url) in zip(extraction_results, urls[:3]):
                if not data: continue
                
                # Use extracted data (JSON-LD/Meta) or reasonable defaults
                found_price = data.get("price") or "Check Site"
                found_title = data.get("description") or f"{query} from {domain_name}"
                if len(found_title) > 80: found_title = found_title[:77] + "..."
                
                fallback_items.append({
                    "name": found_title,
                    "url": url,
                    "source_url": url,
                    "image_url": img_url,
                    "price": found_price,
                    "rating_avg": data.get("rating"),
                    "source": domain_name,
                    "brand": domain_name
                })
            return fallback_items

        fallback_tasks = []
        if not amazon_products: fallback_tasks.append(fetch_domain_fallback("Amazon India", "amazon.in"))
        if not ebay_products: fallback_tasks.append(fetch_domain_fallback("eBay", "ebay.com"))
        if not flipkart_products: fallback_tasks.append(fetch_domain_fallback("Flipkart", "flipkart.com"))
        if not walmart_products: fallback_tasks.append(fetch_domain_fallback("Walmart", "walmart.com"))

        if fallback_tasks:
            try:
                # Fallback search also gets a strict timeout
                fallback_results = await asyncio.wait_for(asyncio.gather(*fallback_tasks), timeout=8.0)
                # Merge fallbacks into products lists
                for fb_list in fallback_results:
                    if not fb_list: continue
                    domain = fb_list[0]["source"]
                    if domain == "Amazon India": amazon_products = fb_list
                    elif domain == "eBay": ebay_products = fb_list
                    elif domain == "Flipkart": flipkart_products = fb_list
                    elif domain == "Walmart": walmart_products = fb_list
            except asyncio.TimeoutError:
                print("WARNING: Fallback search timed out! Proceeding with current partial results.", flush=True)
            except Exception as e:
                print(f"ERROR: Fallback gather failed: {e}", flush=True)

        print(f"DEBUG: Final sources — Amazon: {len(amazon_products)}, eBay: {len(ebay_products)}, Flipkart: {len(flipkart_products)}, Walmart: {len(walmart_products)}", flush=True)
        
        # 2. Build fast_results: Interleave all sources
        fast_results = []
        all_live_products = []
        max_len = max(len(amazon_products), len(ebay_products), len(flipkart_products), len(walmart_products))
        for i in range(max_len):
            if i < len(amazon_products): all_live_products.append(amazon_products[i])
            if i < len(ebay_products): all_live_products.append(ebay_products[i])
            if i < len(flipkart_products): all_live_products.append(flipkart_products[i])
            if i < len(walmart_products): all_live_products.append(walmart_products[i])
        
        # --- CLIENT REQUIREMENT: 'Cheap' / Ranking / Sorting Logic ---
        # Detect intent and categorical focus for price-sensitive queries
        ranking_keywords = ["cheap", "affordable", "low price", "budget", "under", "sort by price", "low to high", "cheapest"]
        is_ranking_requested = any(word in query.lower() for word in ranking_keywords)
        
        if is_ranking_requested:
            print("DEBUG: Price-based ranking/sorting requested. Applying low-to-high sort.", flush=True)
            # Sort by parsed numeric price. 'Check Site'/inf items go to the back.
            all_live_products.sort(key=lambda x: self._parse_price(x.get("price")))
        else:
            # Default: Rank by keyword relevance to avoid "Mixed results" (e.g. shoes in shirts query)
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            def relevance_score(p):
                name_words = set(re.findall(r'\b\w+\b', (p.get("name") or "").lower()))
                return len(query_words.intersection(name_words))
            
            all_live_products.sort(key=relevance_score, reverse=True)
            print(f"DEBUG: Re-ranked {len(all_live_products)} products by keyword relevance.", flush=True)

        for idx, product in enumerate(all_live_products[:num_results]):
            # --- CRITICAL FIX: Prioritize Native Image ---
            # Using index-based Bing image mapping causes product-image mismatches.
            # We now prioritize the retailer's native image (Amazon/eBay) which is correct by definition.
            # AssetProcessor already handles upscaling Amazon/eBay thumbnails to high-res.
            img_url = product.get("image_url")
            
            # Use Bing image ONLY as a fallback if the product has no image at all
            if not img_url and idx < len(bing_images):
                img_url = bing_images[idx].get("image_url")
            
            fast_results.append({
                "name": product["name"],
                "url": product["url"],
                "source_url": product["source_url"],
                "image_url": img_url,
                "price": product["price"],
                "rating_avg": product["rating_avg"],
                "rating_count": product.get("rating_count"),
                "brand": product["brand"],
                "source": product["source"],
                "details": f"{product['name']} - {product.get('price', '')} on {product['source']}"
            })
        
        # 3. Supplement with DDG results to fill up to num_results
        if ddg_results and len(fast_results) < num_results:
            for idx, res in enumerate(ddg_results):
                if len(fast_results) >= num_results: break
                url = res["url"]
                domain = urlparse(url).netloc.lower()
                store_name = domain.replace("www.", "").split('.')[0].capitalize()
                
                if any(native in domain for native in ['amazon', 'ebay']):
                    continue
                
                # --- CRITICAL FIX: Backfill with Bing Image ---
                # Instead of None, attempt to find a matching image from Bing
                img_url = None
                if idx < len(bing_images):
                    img_url = bing_images[idx].get("image_url")
                
                # Skip if URL already in fast_results
                existing_urls = {self._normalize_url(r["url"]) for r in fast_results}
                if self._normalize_url(url) in existing_urls: continue
                
                snippet = res.get("snippet", "")
                
                # --- ROBUST PRICE EXTRACTION ---
                price = self._extract_price_from_snippet(snippet, domain, store_name)
                
                # De-duplication check by URL and Name
                norm_url = self._normalize_url(url)
                if any(self._normalize_url(p.get("url")) == norm_url for p in fast_results):
                    continue
                if any(p.get("name") == res.get("title") for p in fast_results):
                    continue

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
            
        # 2. Archive the JSON results in the background
        # Note: Image processing and storage are now handled in the background by api.py task
        asyncio.create_task(self._archive_results(fast_results, query))
        
        return fast_results

    async def run_deep_crawl_process(self, query, fast_bing_products):
        print(f"DEBUG: Starting background run_deep_crawl_process for {query}", flush=True)
        urls_with_images = {self._normalize_url(p["source_url"]): p.get("image_url") for p in fast_bing_products if p.get("source_url")}
        urls = list(urls_with_images.keys())
        
        results = []
        if urls:
            from crawler import crawl_site
            from crawl4ai import AsyncWebCrawler
            
            # Using the global class semaphore to avoid system exhaustion
            # semaphore = asyncio.Semaphore(3) # Removed local
            
            async def crawl_and_extract_task(url, crawler):
                async with self.semaphore:
                    try:
                        print(f"🚀 [CRAWL] Start -> {url}", flush=True)
                        content, _ = await crawl_site(url, crawler=crawler)
                        if not content or len(content) < 200:
                            return []
                        
                        clean = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL)
                        clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL)
                        clean = re.sub(r"<[^>]+>", " ", clean)
                        
                        # --- CLIENT REQUIREMENT: Data Archival (Markdown) ---
                        from s3_service import s3_service
                        import datetime
                        import uuid
                        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                        safe_domain = urlparse(url).netloc.replace(".", "_")
                        md_filename = f"{safe_domain}/markdown/{date_str}/raw_{uuid.uuid4().hex[:8]}.md"
                        s3_service.upload_data(content, md_filename, content_type="text/markdown")
                        
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
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        
        if results:
            # Process images (asynchronous call)
            # Pass source/date for categorized folders
            results = await asset_processor.process_product_images(results, category="retail", subcategory="deep_crawl", source="deep_crawl", scrape_date=date_str)
            for p in results:
                if p.get("s3_image_url"):
                    p["image_url"] = p["s3_image_url"] 

        # Archive final deep results
        asyncio.create_task(self._archive_results(results, query, category="deep_crawl"))
        return results[:20]

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
        Real-time lightweight Google Search for products and snippets using curl.
        """
        print(f"DEBUG: Starting real-time search_sources for: {query}", flush=True)
        search_results = []
        try:
            from urllib.parse import quote_plus
            import subprocess
            
            final_query = query
            if "site:" not in query.lower():
                top_sites = "(site:amazon.in OR site:flipkart.com OR site:walmart.com OR site:ebay.com OR site:bestbuy.com OR site:croma.com)"
                final_query = f"{query} {top_sites}"
                
            search_url = f"https://www.google.com/search?q={quote_plus(final_query)}"
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            
            # Use curl subprocess as it's proven to work in this environment
            cmd = [
                "curl", "-s", "-L",
                "-H", f"User-Agent: {user_agent}",
                search_url
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                content = stdout.decode('utf-8', errors='ignore')
                soup = BeautifulSoup(content, "html.parser")
                
                # Google search results are typically in div.g or matching containers
                for result in (soup.select('div.g') or soup.select('div.tF2Cxc') or soup.select('div.MjjYud')):
                    title_el = result.select_one('h3')
                    link_el = result.select_one('a')
                    snippet_el = result.select_one('div.VwiC3b') or result.select_one('span.aCOp9b') or result.select_one('div.itY3B')
                    
                    if title_el and link_el:
                        url = link_el.get('href', '')
                        if not url or any(x in url for x in ["google.com", "google.co.in", "youtube.com"]): continue
                        
                        search_results.append({
                            "url": url,
                            "title": title_el.get_text(strip=True),
                            "snippet": snippet_el.get_text(strip=True) if snippet_el else ""
                        })
                    if len(search_results) >= limit: break
            else:
                print(f"DEBUG: curl failed with code {process.returncode}: {stderr.decode()}", flush=True)
                raise Exception("curl search failed")
            
            if search_results:
                 print(f"DEBUG: Found {len(search_results)} real search results from Google via curl.", flush=True)
                 return search_results
                
        except Exception as e:
            print(f"Real-time search failed: {e}. Falling back to image-source discovery.", flush=True)

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
        Includes S3 image processing and archival.
        """
        if not products:
            return

        print(f"\n🚀 [BACKGROUND] Starting caching and S3 enrichment for: {query}", flush=True)
        
        try:
            from asset_processor import asset_processor
            from ingest import add_multiple_contents_to_store
            
            # 1. PROCESS IMAGES FOR S3 (In the background!)
            date_str = self._get_scrape_date()
            products = await asset_processor.process_product_images(
                products, 
                category="retail", 
                subcategory="fast_carousel", 
                source="fast_carousel", 
                scrape_date=date_str
            )

            print(f"📦 [BACKGROUND] Processing {len(products)} products after S3 enrichment...", flush=True)
            
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