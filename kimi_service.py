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

    def __init__(self):
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = "claude-haiku-4-5-20251001"
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
        if not price_str or any(w in str(price_str) for w in ["Check", "Verifying", "Request"]):
            return float('inf')
        try:
            # Standardize: remove whitespace and commas
            s = str(price_str).replace(',', '').replace(' ', '').replace('Rs.', 'Rs').replace('rs.', 'rs')
            
            # Extract number - handle currencies better (using unicode for Rupee \u20b9)
            m = re.search(r'(?:[$₹\u20b9£€]|rs|inr|usd|gbp|eur)\s*([\d,]+\.?\d*)', s, re.IGNORECASE)
            if m:
                clean = m.group(1)
            else:
                clean = re.sub(r'[^\d.]', '', s)
            
            if not clean or not any(c.isdigit() for c in clean):
                return float('inf')
                
            if clean.count('.') > 1:
                parts = clean.split('.')
                clean = parts[0] + "." + parts[1][:2]
            
            val = float(clean)
            
            # Heuristic: if price is > 1000 and has no dot, assume cents, ONLY for USD-like prices
            # We strictly check for $ or USD to ensure INR (e.g. ₹21,998) is never mangled.
            if val > 100 and '.' not in clean and any(sym in str(price_str).lower() for sym in ['$', 'usd']):
                val = val / 100.0
            return val
        except:
            return float('inf')

    def _extract_price_from_snippet(self, snippet, domain=None, store_name=None):
        """
        Unified robust price extraction from DuckDuckGo/Bing snippets and scrapers.
        Handles multi-currency, merged text (e.g. NOW$2399current price), and store patterns.
        """
        if not snippet: return "Request Price"
        
        prices = []
        seen = set()
        # Find formatted prices globally
        matches = re.finditer(r'(?i)([$₹£€]|rs\.?|inr|usd|gbp|eur)\s*([\d,]+\.?\d*)', str(snippet))
        
        symbols_map = {"rs": "₹", "rs.": "₹", "inr": "₹", "usd": "$", "gbp": "£", "eur": "€"}
        for m in matches:
            sym = m.group(1).lower()
            sym = symbols_map.get(sym, m.group(1).upper() if len(m.group(1)) > 1 else m.group(1))
            num = m.group(2).strip()
            if num and num != '.' and any(c.isdigit() for c in num):
                if num.endswith('.') and num.count('.') == 1:
                    num = num[:-1]
                if any(c.isdigit() for c in num):
                    p_str = f"{sym}{num}"
                    if p_str not in seen:
                        seen.add(p_str)
                        prices.append(p_str)

        if not prices:
            # Fallback ISO or reverse
            reverse_prices = re.findall(r'\d{1,7}(?:[.,]\d{3})*(?:[.,]\d{2})?\s?[$₹£€]', str(snippet))
            for rp in reverse_prices:
                prices.append(rp.strip())

        if prices:
            valid_choices = []
            for p in prices:
                # Standardize to avoid "₹ 3,999" vs "₹3,999"
                p_clean = p.replace(" ", "").replace(",", "")
                valid_choices.append((p, self._parse_price(p_clean)))
                    
            # Heuristic: Filter out outliers and original prices
            # 1. Skip prices with 'was', 'original' context
            final_candidates = []
            for p, val in valid_choices:
                is_original = False
                for m in re.finditer(re.escape(p), str(snippet)):
                    ctx = str(snippet)[max(0, m.start() - 30):m.start()].lower()
                    if any(w in ctx for w in ["was", "original", "list", "save", "regular", "previous"]):
                        is_original = True
                if not is_original:
                    final_candidates.append((p, val))
            
            if not final_candidates:
                final_candidates = valid_choices

            # Return the FIRST plausible price found in the text, preserving natural order
            if final_candidates:
                val = final_candidates[0][1]
                if val == float('inf'): return "Request Price"
                
                # Format with 2 decimal places and original symbol if possible
                orig = final_candidates[0][0]
                symbol = "$"
                for s in "$₹£€₹":
                    if s in orig:
                        symbol = s
                        break
                
                return f"{symbol}{val:,.2f}"
            
        return "Request Price"


    async def rewrite_query_contextual(self, query, history):
        """
        Rewrites a short/contextual query (e.g. "under $100") into a standalone
        search query (e.g. "kids shoes under $100") based on conversation history.
        """
        if not history or len(history) < 1:
            return query
            
        context = ""
        # Look at last 3 turns
        for msg in history[-3:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content") or ""
            # Strip bulky grid tags to save tokens
            content = re.sub(r"<product_grid>.*?</product_grid>", "[Product Grid]", content, flags=re.DOTALL)
            context += f"{role}: {content}\n"
            
        prompt = f"""Conversation History:
{context}

Latest Message: {query}

Standalone Query:"""
        
        system = "Rewrite the latest message into a standalone search query that preserves context. Return ONLY the search query text."
        
        try:
            print(f"🔄 [RAG] Contextualizing query: {query}", flush=True)
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if response:
                rewritten = response.content[0].text.strip().strip('"').strip("'")
                # Safety check: if LLM returns an empty string or just the same query, use original
                if rewritten and len(rewritten) > 2:
                    return rewritten
        except Exception as e:
            print(f"DEBUG: Contextualization failed: {e}", flush=True)
            
        return query


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
            "nursery", "baby", "clothes", "clothing", "gown", "cheap", "affordable",
            "diaper", "stroller", "pacifier", "crib", "stuffed animal", "lego", "mattel"
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
                    
                    # Bing encodes JSON in data-m attribute - we extract the whole JSON block to elegantly get url and title
                    results_data = re.findall(r'm="({.*?})"', html_content)
                    
                    blocks = []
                    import html
                    for m_str in results_data:
                        try:
                            m_json = m_str.replace('&quot;', '"').replace('&amp;', '&')
                            data = json.loads(m_json)
                            if 'turl' in data and 'purl' in data:
                                raw_t = html.unescape(data.get('t', '')).strip()
                                # Clean up bad titles like 100_7384.JPG
                                if not raw_t or any(raw_t.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.img', 'image']) or len(raw_t) <= 3:
                                    # Fallback to domain name
                                    try:
                                        from urllib.parse import urlparse
                                        domain = urlparse(data['purl']).netloc.replace('www.', '').split('.')[0].capitalize()
                                        raw_t = f"{clean_query.title()} at {domain}"
                                    except:
                                        raw_t = clean_query.title()
                                        
                                blocks.append((data['turl'], data['purl'], raw_t))
                        except:
                            pass

                    # Deduplicate and filter
                    real_results = []
                    seen = set()
                    for img_url, pg_url, raw_title in blocks:
                        if img_url.startswith("//"): img_url = "https:" + img_url
                        if pg_url.startswith("//"): pg_url = "https:" + pg_url
                        
                        if not img_url.startswith("http"): continue
                        # Filter out potential internal/junk URLs
                        if any(x in img_url for x in ["bing.com", "google.com", "gstatic.com", "microsoft.com"]): continue
                        
                        # Filter out non-retail stock photography, social, wallpaper and graphic illustration sites
                        blocked_domains = [
                            "pngtree", "pngkey", "pngimg", "freepik", "dreamstime", "publicdomainpictures",
                            "123rf", "istockphoto", "shutterstock", "vectorstock", "vecteezy", "pinterest",
                            "alamy", "depositphotos", "pixabay", "pexels", "unsplash", "wallpaper", "clipart",
                            "deviantart", "flickr", "giphy", "tenor", "imgur", "wikipedia", "wikimedia"
                        ]
                        if any(d in pg_url.lower() or d in img_url.lower() for d in blocked_domains):
                            continue
                            
                        if img_url in seen: continue
                        seen.add(img_url)
                        
                        real_results.append({
                            "name": raw_title,
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

        # Final fallback: return empty results instead of dummy data
        return {
            "type": "images",
            "query": clean_query,
            "results": []
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
                        
                        # Filter out Prime Video, Kindle, and Digital services from physical retail search
                        item_text = item.get_text(strip=True).lower()
                        if any(x in item_text for x in ["prime video", "to rent", "prime membership"]):
                            continue
                        
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
                        
                        # Bypass lazy loading placeholders
                        def _is_placeholder(u):
                            if not u: return True
                            u_str = str(u).lower()
                            return any(p in u_str for p in ["grey-pixel", "01rrzvo", "spacer", "placeholder", "transparent", "pixel", ".svg"])
                        
                        # Fix 1: High-res Amazon image
                        raw_img = None
                        if img_el:
                            raw_img = img_el.get("data-lazy-src") or img_el.get("data-src")
                            if not raw_img or _is_placeholder(raw_img):
                                # Try srcset, safely extracting URLs ignoring native commas
                                srcset = img_el.get("srcset")
                                if srcset:
                                    urls = re.findall(r'https?://[^\s]+', srcset)
                                    valid_urls = [u for u in urls if not _is_placeholder(u)]
                                    if valid_urls:
                                        raw_img = valid_urls[-1] # Extract highest resolution image
                            if not raw_img or _is_placeholder(raw_img):
                                raw_img = img_el.get("src")
                            if raw_img and _is_placeholder(raw_img):
                                raw_img = None # Reset to None so it can fall back to Bing images
                        # Use the exact extracted thumbnail image instead of aggressively upscaling, 
                        # as upscaling often leads to 404 Not Found and triggers unwanted seed image fallbacks.
                        img_url = raw_img
                        
                        # NEW: Robust Amazon Price Extraction
                        if not price_str or price_str == "Request Price":
                            # Try finding price in decimals
                            p_off = item.select_one('.a-offscreen')
                            if p_off:
                                price_str = p_off.get_text(strip=True)
                            else:
                                # AGGRESSIVE HUNT: Look for any currency symbol in the item's HTML
                                itxt = item.get_text(separator=" ", strip=True)
                                p_any = self._extract_price_from_snippet(itxt)
                                if p_any != "Request Price":
                                    price_str = p_any
                                else:
                                    # NUCLEAR OPTION: Regex directly on raw HTML for the first currency match
                                    html_str = str(item)
                                    m_nuke = re.search(r'(₹|Rs\.?)\s*([\d,]+)', html_str)
                                    if m_nuke:
                                        price_str = f"₹{m_nuke.group(2)}"

                        # Skip bogus small prices for expensive categories (like Laptops)
                        if price_str and "laptop" in (query + name.get_text()).lower():
                            p_val = self._parse_price(price_str)
                            if p_val < 500: # Laptops aren't under ₹500
                                price_str = "Request Price"
                        products.append({
                            "name": name.get_text(strip=True),
                            "price": price_str if price_str else "Request Price",
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
                            
                        text = item.get_text()
                        found_price = self._extract_price_from_snippet(text)

                        img = item.find('img')
                        link = item.find('a')
                        
                        # eBay image upscaling
                        raw_img = None
                        if img:
                            raw_img = img.get("data-lazy-src") or img.get("data-src")
                            def _is_placeholder(u):
                                if not u: return True
                                u_str = str(u).lower()
                                return any(p in u_str for p in ["grey-pixel", "01rrzvo", "spacer", "placeholder", "transparent", "pixel", ".svg", "gif"])
                            if not raw_img or _is_placeholder(raw_img):
                                raw_img = img.get("src")
                            if raw_img and _is_placeholder(raw_img):
                                raw_img = None
                        img_url = re.sub(r's-l\d+', 's-l500', raw_img) if (raw_img and "s-l" in raw_img) else raw_img

                        # NEW: Enhanced eBay Extraction (Ratings/Reviews)
                        rating_val = None
                        reviews_count = None
                        
                        # eBay sometimes shows rating in a span with 'aria-label' or 'st-stars'
                        stars_el = item.find(class_=re.compile(r'star|rating'))
                        if stars_el and stars_el.get('aria-label'):
                            m = re.search(r'(\d+\.?\d*)\s*out of 5', stars_el.get('aria-label'))
                            if m: rating_val = float(m.group(1))
                        
                        # Review count often follows the stars
                        reviews_el = item.find(class_=re.compile(r'reviews|total-ratings'))
                        if reviews_el:
                            m = re.search(r'([\d,]+)', reviews_el.get_text())
                            if m: reviews_count = m.group(1)

                        if found_price != "Request Price":
                            products.append({
                                "name": name_text[:70],
                                "price": found_price,
                                "rating_avg": rating_val,
                                "rating_count": reviews_count,
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
                        
                        # NEW: Enhanced Flipkart Extraction (Ratings/Reviews)
                        rating_el = item.select_one('div._3LWZlK') or item.select_one('span._2_R_o9')
                        rating_val = None
                        if rating_el:
                            try: rating_val = float(rating_el.get_text(strip=True).replace('★', ''))
                            except: pass
                            
                        reviews_el = item.select_one('span._2_R_o9') or item.select_one('span._2_R_o9')
                        reviews_count = None
                        if reviews_el:
                            m = re.search(r'([\d,]+)', reviews_el.get_text())
                            if m: reviews_count = m.group(1)

                        def _get_valid_flipkart_img(el):
                            if not el: return None
                            u = el.get("data-lazy-src") or el.get("data-src") or el.get("src")
                            if not u: return None
                            u_str = str(u).lower()
                            if any(p in u_str for p in ["grey-pixel", "01rrzvo", "spacer", "placeholder", "transparent", "pixel", ".svg"]):
                                return None
                            return u

                        products.append({
                            "name": name,
                            "price": self._extract_price_from_snippet(price_el.get_text(strip=True)),
                            "rating_avg": rating_val,
                            "rating_count": reviews_count,
                            "image_url": _get_valid_flipkart_img(img_el),
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
                        
                        # Extract price text manually and use unified method
                        if price_el:
                            for el in price_el.find_all(['sup', 'sub', 'span']):
                                txt = el.get_text(strip=True)
                                if len(txt) == 2 and txt.isdigit() and not el.get_text().startswith('.'):
                                    el.string = f".{txt}"
                        raw_price_text = price_el.get_text(strip=True) if price_el else ""
                        price_text = self._extract_price_from_snippet(raw_price_text, "walmart.com", "Walmart")
                        
                        p_url = urljoin("https://www.walmart.com", link_el.get('href', '')) if link_el else url
                        
                        # NEW: Enhanced Walmart Extraction
                        rating_el = item.select_one('[data-testid="rating-number"]') or item.select_one('.w_V_o')
                        rating_val = None
                        reviews_count = None
                        if rating_el:
                            txt = rating_el.get_text(strip=True)
                            m = re.search(r'(\d+\.?\d*)', txt)
                            if m: rating_val = float(m.group(1))
                            m_rev = re.search(r'\((\d+)\)', txt)
                            if m_rev: reviews_count = m_rev.group(1)

                        if not price_text or price_text == "Request Price":
                            # AGGRESSIVE HUNT for Walmart
                            p_any = self._extract_price_from_snippet(item.get_text())
                            if p_any != "Request Price":
                                price_text = p_any
                            else:
                                # NUCLEAR OPTION for Walmart ($)
                                m_nuke = re.search(r'\$\s*([\d,]+\.?\d*)', str(item))
                                if m_nuke:
                                    price_text = f"${m_nuke.group(1)}"

                        def _get_valid_walmart_img(el):
                            if not el: return None
                            u = el.get("data-lazy-src") or el.get("data-src") or el.get("src")
                            if not u: return None
                            u_str = str(u).lower()
                            if any(p in u_str for p in ["grey-pixel", "01rrzvo", "spacer", "placeholder", "transparent", "pixel", ".svg"]):
                                return None
                            return u

                        products.append({
                            "name": name,
                            "price": price_text,
                            "rating_avg": rating_val,
                            "rating_count": reviews_count,
                            "image_url": _get_valid_walmart_img(img_el),
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

    def _validate_product_relevance(self, product_name: str, query: str, min_score: float = 0.3) -> tuple:
        """
        Validate if a product matches the query with semantic understanding.
        Returns (is_relevant, relevance_score)
        
        Prevents issues like "slider" appearing for "kids play ten"
        """
        if not product_name or not query:
            return False, 0.0
        
        # Normalize inputs
        p_lower = str(product_name).lower().strip()
        q_lower = query.lower().strip()
        
        # Stop words that don't indicate relevance
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'for', 'with', 'in', 'on', 'at', 'from', 'to',
            'of', 'as', 'by', 'is', 'are', 'be', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'new', 'pack', 'set', 'lot', 'combo', 'best', 'premium', 'sale', 'buy'
        }
        
        # Extract meaningful words (length >= 3 to avoid single letters)
        query_words = [w for w in re.findall(r'\b\w+\b', q_lower) if len(w) >= 3 and w not in stop_words]
        product_words = [w for w in re.findall(r'\b\w+\b', p_lower) if len(w) >= 3 and w not in stop_words]
        
        if not query_words:
            return False, 0.0

        # Exact phrase match allows singular/plural variants to pass immediately
        phrase_variants = {q_lower}
        words = q_lower.split()
        if len(words) == 2:
            if words[1].endswith('s'):
                phrase_variants.add(f"{words[0]} {words[1][:-1]}")
            else:
                phrase_variants.add(f"{words[0]} {words[1]}s")
        if any(variant in p_lower for variant in phrase_variants):
            return True, 1.0

        # Calculate how many query words appear in product name
        matching_words = sum(1 for qw in query_words if any(qw in pw or pw in qw for pw in product_words))
        relevance_score = matching_words / len(query_words) if query_words else 0.0

        # For multi-word queries (3+ words), require higher relevance
        if len(query_words) >= 3:
            required_score = 0.5  # Need at least 50% of words to match
        elif len(query_words) == 2:
            required_score = 0.5  # Need both or one of two
        else:
            required_score = 0.3  # Single word is more lenient

        if len(query_words) == 2 and matching_words == 1:
            generic_words = {
                'wooden', 'plastic', 'metal', 'small', 'large', 'mini', 'kids', 'baby',
                'outdoor', 'indoor', 'home', 'garden', 'decorative', 'antique', 'vintage',
                'children', 'toy', 'set', 'pack'
            }
            specific_matches = [qw for qw in query_words if qw not in generic_words]
            if specific_matches and not any(s in p_lower for s in specific_matches):
                return False, 0.0

        is_relevant = relevance_score >= required_score
        
        # Additional check: if the product name contains words that contradict the query
        contradiction_keywords = {
            'slider': ['kids play ten', 'toys', 'games', 'puzzle'],
            'frame': ['phone', 'laptop', 'tablet', 'electronics'],
            'stand': ['shirt', 'clothes', 'apparel', 'shoes'],
        }
        
        for bad_word, queries in contradiction_keywords.items():
            if bad_word in p_lower and any(q in q_lower for q in queries):
                is_relevant = False
                relevance_score = 0.0
                break
        
        return is_relevant, relevance_score

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
        Orchestrates parallel retail scrapers using Crawl4AI headless Chromium
        to bypass WAF blocking (Amazon 503, eBay 403, etc.) and extracts product lists 
        directly using Claude without saving anything to the vector store or S3.
        """
        from crawler import crawl_site_fast, get_browser_config
        from crawl4ai import AsyncWebCrawler
        import urllib.parse
        
        encoded_query = urllib.parse.quote_plus(query)
        urls = {
            "Amazon India": f"https://www.amazon.in/s?k={encoded_query}",
            "eBay": f"https://www.ebay.com/sch/i.html?_nkw={encoded_query}",
            "Flipkart": f"https://www.flipkart.com/search?q={encoded_query}",
            "Walmart": f"https://www.walmart.com/search?q={encoded_query}"
        }
        
        browser_config = get_browser_config()
        # Ensure images are disabled to maximize speed
        if "--blink-settings=imagesEnabled=false" not in browser_config.extra_args:
            browser_config.extra_args.append("--blink-settings=imagesEnabled=false")
            
        scraped_pages = {}
        is_remote = browser_config.browser_mode == "custom"
        try:
            # Create a single shared crawler to run crawls concurrently under 1 browser process!
            async with AsyncWebCrawler(config=browser_config) as crawler:
                tasks = []
                retailers = list(urls.keys())
                for retailer in retailers:
                    url = urls[retailer]
                    tasks.append(crawl_site_fast(url, crawler=crawler))
                    
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for retailer, res in zip(retailers, results):
                    if isinstance(res, Exception):
                        print(f"ERROR: Crawl4AI failed for {retailer}: {res}", flush=True)
                        continue
                    content, links = res
                    if content and len(content.strip()) > 50:
                        scraped_pages[retailer] = content
                        print(f"DEBUG: Successfully crawled {retailer} search results (length: {len(content)})", flush=True)
                    else:
                        print(f"WARNING: Crawl4AI returned empty search results for {retailer}", flush=True)
        except Exception as e:
            if is_remote:
                print(f"WARNING: Remote Crawl4AI parallel search failed: {e}. Attempting local fallback...", flush=True)
                try:
                    from crawler import get_browser_config
                    local_config = get_browser_config(force_local=True)
                    if "--blink-settings=imagesEnabled=false" not in local_config.extra_args:
                        local_config.extra_args.append("--blink-settings=imagesEnabled=false")
                    async with AsyncWebCrawler(config=local_config) as crawler:
                        tasks = []
                        retailers = list(urls.keys())
                        for retailer in retailers:
                            url = urls[retailer]
                            tasks.append(crawl_site_fast(url, crawler=crawler))
                            
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for retailer, res in zip(retailers, results):
                            if isinstance(res, Exception):
                                print(f"ERROR: Local Crawl4AI failed for {retailer}: {res}", flush=True)
                                continue
                            content, links = res
                            if content and len(content.strip()) > 50:
                                scraped_pages[retailer] = content
                                print(f"DEBUG: Successfully crawled {retailer} search results locally (length: {len(content)})", flush=True)
                            else:
                                print(f"WARNING: Local Crawl4AI returned empty search results for {retailer}", flush=True)
                except Exception as local_err:
                    print(f"ERROR: Fatal exception in local Crawl4AI fallback: {local_err}", flush=True)
            else:
                print(f"ERROR: Fatal exception in local Crawl4AI parallel search: {e}", flush=True)
            
        # Parse products from each retailer's HTML using Claude!
        extracted_products = []
        
        async def parse_retailer_products(retailer, content):
            print(f"DEBUG: parse_retailer_products called for {retailer} with content length {len(content)}", flush=True)
            prompt = (
                f"Extract a list of the top 10 products from this {retailer} search result page webpage.\n"
                f"You MUST extract exactly:\n"
                f"- name: Concise product title/name (keep it clean and brief)\n"
                f"- price: Price string with currency symbol (e.g. ₹1,299 or $19.99)\n"
                f"- image_url: A valid product thumbnail image URL from the webpage\n"
                f"- url: The product detail page path/URL\n"
                f"- rating: Average rating as float (out of 5), or null if not found\n"
                f"- brand: Brand name or {retailer}\n"
                f"- details: A short sentence highlighting specifications (MAX 8 words, keep it very brief)\n"
                f"\nSearch page webpage content:\n{content[:60000]}"
            )
            try:
                response = await self._call_with_retry(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=4000,
                        system="Return ONLY a valid JSON object with a key 'products' containing the list of products.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                )
                if not response:
                    print(f"DEBUG: LLM returned empty/null response for {retailer}", flush=True)
                    return []
                raw_text = response.content[0].text
                print(f"DEBUG: LLM Raw response for {retailer} (length: {len(raw_text)}):\n{raw_text}\n", flush=True)
                data = self._safe_json_parse(raw_text, "products")
                products = data if isinstance(data, list) else data.get("products", [])
                print(f"DEBUG: Successfully parsed {len(products)} products from LLM response for {retailer}", flush=True)
                
                base_url = "https://www.amazon.in" if retailer == "Amazon India" else \
                           "https://www.ebay.com" if retailer == "eBay" else \
                           "https://www.flipkart.com" if retailer == "Flipkart" else \
                           "https://www.walmart.com"
                
                for p in products:
                    p["source"] = retailer
                    if not p.get("brand"):
                        p["brand"] = retailer
                    p_url = p.get("url") or p.get("source_url")
                    if p_url:
                        p["url"] = urllib.parse.urljoin(base_url, p_url)
                        p["source_url"] = p["url"]
                    else:
                        p["url"] = base_url
                        p["source_url"] = base_url
                        
                return products
            except Exception as e:
                print(f"ERROR: Failed to parse products for {retailer} via LLM: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return []
                
        if scraped_pages:
            parse_tasks = []
            for retailer, html in scraped_pages.items():
                parse_tasks.append(parse_retailer_products(retailer, html))
            parse_results = await asyncio.gather(*parse_tasks)
            for res_list in parse_results:
                extracted_products.extend(res_list)
                
        # Interleave product results to show a diverse grid of Amazon, eBay, Flipkart, Walmart
        fast_results = []
        retailer_groups = {}
        for p in extracted_products:
            source = p.get("source", "Other")
            if source not in retailer_groups:
                retailer_groups[source] = []
            retailer_groups[source].append(p)
            
        max_len = max([len(lst) for lst in retailer_groups.values()], default=0)
        for i in range(max_len):
            for source in retailer_groups:
                if i < len(retailer_groups[source]):
                    p = retailer_groups[source][i]
                    img_url = p.get("image_url")
                    
                    def _is_placeholder(u):
                        if not u: return True
                        u_str = str(u).lower()
                        return any(pl in u_str for pl in ["grey-pixel", "01rrzvo", "spacer", "placeholder", "transparent", "pixel", ".svg"])
                    
                    if img_url and _is_placeholder(img_url):
                        img_url = None
                        
                    fast_results.append({
                        "name": p["name"],
                        "url": p["url"],
                        "source_url": p["source_url"],
                        "image_url": img_url,
                        "price": p["price"],
                        "rating_avg": p.get("rating"),
                        "rating_count": None,
                        "brand": p["brand"],
                        "source": p["source"],
                        "details": f"{p['name']} - {p.get('price', '')} on {p['source']}"
                    })
                    
        # NUCLEAR LAST RESORT: If Crawl4AI yielded 0 products (all blocked/filtered),
        # generate clickable retailer search-link cards so the UI always shows a grid.
        if not fast_results:
            print("DEBUG: NUCLEAR fallback — generating retailer search cards.", flush=True)
            safe_q = query.replace(" ", "+")
            fast_results = [
                {
                    "name": f"{query.title()} on Amazon India",
                    "url": f"https://www.amazon.in/s?k={safe_q}",
                    "source_url": f"https://www.amazon.in/s?k={safe_q}",
                    "image_url": "",
                    "price": "Check Price",
                    "rating_avg": None,
                    "brand": "Amazon India",
                    "source": "Amazon India",
                    "details": f"Click to search for {query} on Amazon India"
                },
                {
                    "name": f"{query.title()} on Flipkart",
                    "url": f"https://www.flipkart.com/search?q={safe_q}",
                    "source_url": f"https://www.flipkart.com/search?q={safe_q}",
                    "image_url": "",
                    "price": "Check Price",
                    "rating_avg": None,
                    "brand": "Flipkart",
                    "source": "Flipkart",
                    "details": f"Click to search for {query} on Flipkart"
                },
                {
                    "name": f"{query.title()} on eBay",
                    "url": f"https://www.ebay.com/sch/i.html?_nkw={safe_q}",
                    "source_url": f"https://www.ebay.com/sch/i.html?_nkw={safe_q}",
                    "image_url": "",
                    "price": "Check Price",
                    "rating_avg": None,
                    "brand": "eBay",
                    "source": "eBay",
                    "details": f"Click to search for {query} on eBay"
                },
                {
                    "name": f"{query.title()} on Walmart",
                    "url": f"https://www.walmart.com/search?q={safe_q}",
                    "source_url": f"https://www.walmart.com/search?q={safe_q}",
                    "image_url": "",
                    "price": "Check Price",
                    "rating_avg": None,
                    "brand": "Walmart",
                    "source": "Walmart",
                    "details": f"Click to search for {query} on Walmart"
                },
            ]
            
        print(f"DEBUG: Returning {len(fast_results)} parsed products directly to user chat.", flush=True)
        return fast_results

    async def run_deep_crawl_process(self, query, fast_bing_products):
        print(f"DEBUG: Starting background run_deep_crawl_process for {query}", flush=True)
        urls_with_images = {self._normalize_url(p["source_url"]): p.get("image_url") for p in fast_bing_products if p.get("source_url")}
        urls = list(urls_with_images.keys())
        
        # If no fast products are provided but the query itself is a valid URL, add it.
        if not urls and query.startswith("http"):
            urls.append(query)
            

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
            results = asset_processor.process_product_images(results, category="retail", subcategory="deep_crawl")
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
            f"- price: The numerical price with currency (e.g., $19.99). MANDATORY: If a price exists anywhere in the text (near 'price', 'now', 'Rs.', symbols), you MUST extract it. Check headers, sidebars, and main content. If absolutely zero price symbols exist, ONLY then return null.\n"
            f"- brand: Brand name\n"
            f"- rating_avg: Numerical average rating (e.g., 4.5) - float or null\n"
            f"- rating_count: total number of customer reviews (e.g., 1250) - integer or null\n"
            f"- offers: Short summary of discounts or free shipping\n"
            f"- source: Store name or platform\n"
            f"- image_url: Direct link to the primary product image. Look for 'large', 'high-res', or 'original' images.\n"
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



    async def rapid_extract_price_and_rating(self, session, url):
        """
        Hyper-fast extraction using raw HTML via aiohttp.
        Targets JSON-LD and Meta tags specifically.
        """
        try:
            # Enhanced headers to avoid "Bot Detection" on top retailers
            symbols = {"$": "$", "USD": "$", "RS": "₹", "INR": "₹", "GBP": "£", "EUR": "€"}
            domain = urlparse(url).netloc
            headers = self._get_stealth_headers(domain)
            async with session.get(url, headers=headers, timeout=8) as response:
                if response.status != 200: 
                    print(f"DEBUG: Rapid extract failed for {url} with status {response.status}", flush=True)
                    return url, None
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                data = {"price": None, "rating": None, "description": None, "image_url": None}
                
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
                            
                            # Look for Image
                            img = item.get("image")
                            if img and not data["image_url"]:
                                if isinstance(img, list):
                                    img = img[0]
                                if isinstance(img, dict):
                                    data["image_url"] = img.get("url") or img.get("contentUrl")
                                elif isinstance(img, str):
                                    data["image_url"] = img

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
                
                # Double-check all prices using the new formatter to standardize output
                if data["price"]:
                    data["price"] = self._extract_price_from_snippet(data["price"])
                    
                if not data["rating"]:
                    meta_r = soup.find("meta", property="og:rating") or soup.find("meta", attrs={"name": "rating"})
                    if meta_r: data["rating"] = meta_r.get("content")
                    
                if not data["image_url"]:
                    meta_img = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
                    if meta_img: data["image_url"] = meta_img.get("content")
                
                # 3. Simple description
                meta_desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", property="og:description")
                if meta_desc: data["description"] = meta_desc.get("content")[:500]
                
                print(f"DEBUG: Rapid extracted data for {url}: {data['price']}, {data['rating']}, IMG: {bool(data['image_url'])}", flush=True)
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
        Real-time lightweight Search for products and snippets using DuckDuckGo.
        Provides resilient price snippets when specialized native scrapers miss.
        """
        print(f"DEBUG: Starting real-time search_sources for: {query}", flush=True)
        search_results = []
        try:
            from urllib.parse import quote_plus
            
            # Use raw query to allow DDG's natural e-commerce indexing to flourish
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, headers=headers, timeout=12) as response:
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")
                    
                    # DuckDuckGo HTML results container
                    for result in soup.find_all('div', class_=re.compile(r'result\s+results_links')):
                        title_el = result.find('a', class_='result__url') or result.find('a', class_='result__a')
                        snippet_el = result.find('a', class_='result__snippet')
                        
                        if title_el:
                            url = title_el.get('href', '')
                            # DDG sometimes prefixes outgoing links with /url?q=
                            if '/url?q=' in url:
                                import urllib.parse as up
                                qs = up.parse_qs(up.urlparse(url).query)
                                url = qs.get("q", [url])[0]

                            # Filter out non-retail domains that dilute commercial searches with stock/junk pages
                            blocked_domains = [
                                "duckduckgo.com", "youtube.com", "pngtree", "pngkey", "pngimg", "freepik",
                                "dreamstime", "publicdomainpictures", "123rf", "istockphoto", "shutterstock",
                                "vectorstock", "vecteezy", "pinterest", "alamy", "depositphotos", "pixabay",
                                "pexels", "unsplash", "wallpaper", "clipart", "wikipedia", "wikimedia", "flickr",
                                "news", "betting", "casino", "blog", "review", "forum", "reddit", "quora", "imdb",
                                "pngall", "pxhere", "stock", "image", "photo", "free"
                            ]
                            
                            title_lower = title_el.get_text(strip=True).lower()
                            if not url or any(d in url.lower() for d in blocked_domains) or any(w in title_lower for w in ["news", "betting", "review", "png", "free image", "stock"]):
                                continue
                            
                            search_results.append({
                                "url": url,
                                "title": title_el.get_text(strip=True),
                                "snippet": snippet_el.get_text(strip=True) if snippet_el else ""
                            })
                        if len(search_results) >= limit: break

            if search_results:
                 print(f"DEBUG: Found {len(search_results)} real search results from DDG.", flush=True)
                 return search_results
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Real-time search failed: {e}. Falling back to image-source discovery.", flush=True)

        # Fallback removed. If DDG search fails or is completely filtered out, 
        # it is better to return [] than to hallucinate products from Bing Image Search.
        return []

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

    async def cache_and_store_products(self, products, query, category_tag=None):
        """
        Background task to ingest live product data into the local vector store.
        Includes S3 image processing and archival.
        """
        if not products:
            return

        tag_str = category_tag or "retail"
        print(f"\n🚀 [BACKGROUND] Starting caching and S3 enrichment for: {query} (Tag: {tag_str})", flush=True)
        
        try:
            from asset_processor import asset_processor
            from ingest import add_multiple_contents_to_store
            
            # 1. PROCESS IMAGES FOR S3 (In the background!)
            products = asset_processor.process_product_images(
                products, 
                category="retail", 
                subcategory=tag_str
            )

            print(f"📦 [BACKGROUND] Processing {len(products)} products after S3 enrichment...", flush=True)
            
            ingest_items = []
            for product in products:
                # Basic description formatting for RAG
                source_url = product.get('source_url') or product.get('url') or "unknown"
                image_url = product.get('image_url')
                
                description = (
                    f"Product: {product.get('name')}\n"
                    f"Brand: {product.get('brand', 'Product')}\n"
                    f"Price: {product.get('price', 'Check Site')}\n"
                    f"Category: {tag_str}\n"
                    f"Details: {product.get('details', 'No details available')}\n"
                    f"Image URL: {image_url}\n"
                    f"Source URL: {source_url}"
                )
                
                # Metadata for ChromaDB
                metadata = {
                    "source": source_url,
                    "type": "live_cache",
                    "category": tag_str,  # CRITICAL: Strict category tagging
                    "image_url": image_url,
                    "s3_image_url": image_url, 
                    "name": product.get("name"),
                    "price": str(product.get("price") or "Request Price"),
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
                print(f"✅ [BACKGROUND] Successfully cached {len(ingest_items)} products for '{query}' (Category: {tag_str})\n", flush=True)
            
        except Exception as e:
            print(f"❌ [BACKGROUND] Error during caching: {e}", flush=True)

    async def extract_direct_product(self, url: str):
        """
        Directly scrape a single product URL using Crawl4AI and extract structured details 
        via LLM, without saving anything to the vector store or S3.
        """
        from crawler import crawl_site_fast
        
        print(f"DEBUG: Starting real-time direct scrape for: {url}", flush=True)
        try:
            # 1. Scrape using high-speed crawl4ai config
            content, links = await crawl_site_fast(url)
            if not content:
                print(f"DEBUG: Direct scrape failed to get content for {url}", flush=True)
                return {"error": "Failed to retrieve page content."}
                
            # 2. Extract structured product info via LLM
            truncated_content = content[:60000]  # First 60k chars for fast LLM processing
            prompt = (
                f"Extract complete product details from this webpage HTML/Markdown content.\n"
                f"You MUST extract:\n"
                f"- name: The clear name/title of the product\n"
                f"- price: The price with original currency symbol (e.g. $29.99, ₹1,299)\n"
                f"- image_url: A high-quality product image URL from the text\n"
                f"- description: A clear 2-3 sentence overview of the product specifications and features\n"
                f"- rating: The average product rating as a float (out of 5), or null if not found\n"
                f"- reviews: Up to 3 user review comments as a list of dicts: {{\"user\": \"name\", \"comment\": \"text\", \"rating\": 5}}\n"
                f"\nContent to parse:\n{truncated_content}"
            )
            
            print(f"DEBUG: Real-time LLM extraction start for {url}", flush=True)
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    system="Return ONLY a single valid JSON object representing the product details.",
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            if not response:
                return {"error": "Failed to extract product details from LLM."}
                
            data = self._safe_json_parse(response.content[0].text, "product")
            # Normalize the url
            if isinstance(data, dict):
                data["url"] = url
                return data
            return {"error": "Failed to parse product data.", "raw": response.content[0].text}
            
        except Exception as e:
            print(f"Error during direct product extraction: {e}", flush=True)
            return {"error": str(e)}

kimi_service = KimiService()