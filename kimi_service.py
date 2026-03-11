import os
import json
import asyncio
import time
import re
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
class KimiService:
    def __init__(self):
        self.api_key = os.getenv("MOONSHOT_API_KEY")
        self.client = Anthropic(
            api_key=self.api_key,
        )
        self.model = "claude-3-haiku-20240307"
        # Strict concurrency: 1 request at a time to stay under 50k tokens/min
        self.semaphore = asyncio.Semaphore(1)
        self.base_retail_domains = [
            "amazon.in", "flipkart.com", "ajio.com", "myntra.com",
            "firstcry.com", "nykaa.com", "jiomart.com", "meesho.com",
            "m.media-amazon.com", "assets.ajio.com", "cdn.fcglcdn.com",
            "mxwholesale.co.uk", "brightminds.co.uk", "babybrandsdirect.co.uk", "puckator-dropship.co.uk",
            "amazon.com", "walmart.com", "apple.com", "bestbuy.com", "ebay.com", "target.com", "costco.com"
        ]
    def _normalize_url(self, url, host=None):
        """Ensures URL starts with https:// and handles relative paths."""
        if not url or not isinstance(url, str):
            return None
        url = url.strip()
        if url.startswith("//"):
            return "https:" + url
        if not url.startswith("http"):
            # If it's just a domain like amazon.in/dp/..., prepend https://
            if any(domain in url for domain in self.base_retail_domains) or url.count("/") > 0:
                return "https://" + url.lstrip("/")
        return url
    def _safe_json_parse(self, text, default_key):
        """
        Safely parses JSON from LLM response, handling markdown blocks,
        trailing commas, and malformed text.
        """
        try:
            # Basic cleanup
            text = text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1).strip()
            # Handle trailing commas in JSON lists/objects
            text = re.sub(r',\s*([\]}])', r'\1', text)
            # Remove any non-JSON text after the final closing bracket/brace
            last_bracket = text.rfind(']')
            last_brace = text.rfind('}')
            end_idx = max(last_bracket, last_brace)
            if end_idx != -1:
                text = text[:end_idx+1]
            return json.loads(text)
        except json.JSONDecodeError as e:
            try:
                # Stage 1: Greedy Extraction (Look for furthest balanced braces/brackets)
                # Try to find a balanced list or object
                def find_balanced(s, open_char, close_char):
                    first = s.find(open_char)
                    if first == -1: return None
                    last = s.rfind(close_char)
                    if last == -1 or last < first: return None
                    return s[first:last+1]
                potential = find_balanced(text, '[', ']') or find_balanced(text, '{', '}')
                if potential:
                    # Stage 2: Recursive Cleaning
                    # Remove trailing commas inside the extracted block
                    potential = re.sub(r',\s*([\]}])', r'\1', potential)
                    # Remove potential unescaped control characters
                    potential = re.sub(r'[\x00-\x1F\x7F]', '', potential)
                    try:
                        return json.loads(potential)
                    except json.JSONDecodeError:
                        # Stage 3: Extreme Measure - Truncate and Close
                        # If it's a list, try to find the last complete object
                        if potential.startswith('['):
                            # Try to find the last closing brace that belongs to an object
                            last_obj_end = potential.rfind('}')
                            if last_obj_end != -1:
                                truncated = potential[:last_obj_end+1] + ']'
                                # Fix trailing comma before the new closing bracket
                                truncated = re.sub(r',\s*\]', ']', truncated)
                                return json.loads(truncated)
            except Exception as inner_e:
                print(f"JSON Parse Failure: {e} | Recovery failed: {inner_e}")
        return {default_key: []}
    async def _call_with_retry(self, func, max_retries=3):
        """
        Calls an Anthropic method with basic exponential backoff for 429s.
        Releases semaphore during sleep to avoid blocking other tasks.
        """
        for i in range(max_retries):
            try:
                async with self.semaphore:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func)
            except Exception as e:
                # Handle 429 (Rate Limit) by sleeping and retrying
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                if is_rate_limit and i < max_retries - 1:
                    wait_time = (5 ** i) + 5 # Exponential backoff
                    print(f"Rate limited (429). Task releasing semaphore and retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    async def search_sources(self, topic):
        """
        Uses Kimi's native reasoning to identify top authoritative URLs for any topic.
        """
        # Detect if the topic is for a specific tech device
        tech_device_kws = ["macbook", "iphone", "ipad", "laptop", "phone", "smartphone", "tablet", "pixel", "galaxy", "airpods"]
        is_tech_device = any(kw in topic.lower() for kw in tech_device_kws)
        is_retail = is_tech_device or any(kw in topic.lower() for kw in ["shoes", "kids", "shopping", "clothes", "toys", "products", "electronics", "gadgets"])
        if is_retail:
            topic_str = topic.strip("'\"")
            if is_tech_device:
                # For specific devices, prioritize official brand stores and curated product listings
                system_msg = "You are a tech shopping expert. Find REAL product pages for the specific device requested. Focus on official stores and direct product pages, NOT generic search result pages full of accessories. Output ONLY a JSON list of URLs."
                # Extract the core device name for better URL construction
                device_name = topic_str.split()[0] if topic_str else topic_str  # e.g. "macbook" from "direct macbook product..."
                for kw in tech_device_kws:
                    if kw in topic_str.lower():
                        device_name = kw
                        break
                prompt = (
                    f"Find the best 5 DIRECT product listing URLs for the device '{device_name}'. "
                    f"Prefer:\n"
                    f"  1. Apple.com/shop for Apple products (e.g. apple.com/shop/buy-mac/macbook-air)\n"
                    f"  2. BestBuy.com category pages (e.g. bestbuy.com/site/searchpage.jsp?st={device_name})\n"
                    f"  3. Amazon.com/s?k={device_name}&rh=n%3A565108 (Electronics category, NOT accessories)\n"
                    f"NEVER use generic Amazon search pages that show mostly accessories or cases. "
                    f"ONLY provide URLs that lead to a page where the DEVICE ITSELF is listed for purchase. "
                    f"Output ONLY a JSON list of direct product URLs."
                )
            else:
                system_msg = "You are a shopping expert. Find REAL product catalog or search result pages from top retailers. Output ONLY a JSON list of URLs."
                prompt = f"Find the best 5 authoritative and official website SEARCH RESULT URLs for the specific product: '{topic_str}'. Use URLs like 'amazon.com/s?k={topic_str}' or similar for Walmart, BestBuy, etc. NEVER provide just the homepage. Output ONLY a JSON list of direct search result URLs."
        else:
            system_msg = "You are a research expert. Output a JSON list of REAL, official documentation or resource URLs. NEVER hallucinate URLs. Return ONLY the JSON object."
            prompt = f"Find the best 5 authoritative and official website URLs for the topic: '{topic}'. Prioritize official documentation, GitHub repositories, or high-quality technical guides. Output ONLY a JSON list."
        try:
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            print(f"DEBUG: Kimi search topic: '{topic}' | is_retail: {is_retail}")
            data = self._safe_json_parse(response.content[0].text, "urls")
            urls = data if isinstance(data, list) else data.get("urls", [])
            print(f"DEBUG: Raw seeds from Kimi: {urls}")
            # Validation: Filter out hallucinated/dummy URLs and ensure they look like real domains
            valid_urls = []
            for u in urls:
                if not isinstance(u, str) or len(u) < 10: continue
                if any(bad in u for bad in ["example.com", "placeholder", "dummy", "hallucinated"]):
                    continue
                # Reject URLs with repeated www. subdomains (e.g. www.www.www.shopify.com)
                if u.count("www.") > 1:
                    print(f"DEBUG: Rejected repeated-www URL: {u}")
                    continue
                # Reject URLs where path/query is longer than 200 chars (URL-encoded topic strings)
                from urllib.parse import urlparse
                try:
                    parsed = urlparse(u if u.startswith("http") else f"https://{u}")
                    # If the query string or path contains the full topic verbatim, it's a bad URL
                    if len(parsed.query) > 200 or len(parsed.path) > 200:
                        print(f"DEBUG: Rejected overly long URL (likely encoded topic): {u[:80]}...")
                        continue
                except Exception:
                    pass
                # If retail, try to ensure it's a known or plausible shopping domain
                if is_retail:
                    # Allow known domains or anything with /dp/, /p/, /product/
                    if any(d in u for d in self.base_retail_domains) or "/dp/" in u or "/p/" in u or "product" in u:
                        valid_urls.append(self._normalize_url(u))
                else:
                    valid_urls.append(self._normalize_url(u))
            print(f"DEBUG: Discovered source seeds: {valid_urls}")
            return valid_urls
        except Exception as e:
            print(f"Error in search_sources: {e}")
            return []
    async def extract_product_data(self, markdown_content, target_category="relevant"):
        """
        Extracts structured product data. Truncates content to fit token limits.
        """
        # Further truncate content to ~20k characters to avoid 50k token/min limit
        # Claude Haiku has 200k context, so 20k is safe and should cover most product pages.
        truncated_content = markdown_content[:20000]
        prompt = (
            f"system_msg = \"You are a surgical data extraction tool. Extract product information ONLY from the provided markdown. DO NOT invent URLs. If an image link is not explicitly shown IN IMMEDIATE PROXIMITY to the product name in the text, you MUST return null. NEVER guess.\"\n\n"
            f"Identify and extract all products matching or closely related to the target category: '{target_category}' from this markdown. For each item, include: name, price (convert to number), "
            f"currency (default to INR/₹), age_group (if applicable), brand, image_url, and the direct product page link (absolute URL).\n\n"
            f"CRITICAL PRODUCT RULES:\n"
            f"1. **RELEVANCE**: Extract items that match the category '{target_category}' (including related accessories if present).\n"
            f"2. **REJECT NON-PRODUCTS**: Do NOT extract logos, advertising banners, site navigation icons, or promotional bundles that aren't the main items.\n\n"
            f"CRITICAL URL RULES:\n"
            f"1. **STRICT EXTRACTION**: ONLY use image URLs explicitly found in <img> tags or markdown image links (![...](...)). The image MUST be physically adjacent to the product name in the markdown.\n"
            f"2. **LOGO PREVENTION**: NEVER extract URLs containing 'logo', 'sprite', 'icon', 'banner', 'nav', 'avatar', or 'header' as product images.\n"
            f"3. **ZERO TOLERANCE FOR HALLUCINATION**: If the markdown doesn't have a working image link immediately next to the product, set image_url to null. NEVER combine names to make a URL.\n"
            f"4. **AVOID PLACEHOLDERS**: NEVER extract base64 data-URIs, '1x1.gif', 'pixel.gif', 'spacer.gif', or ANY URL containing 'example.com' or 'placeholder' as image_url.\n"
            f"5. **REAL DOMAINS ONLY**: Favor URLs from known CDNs. If a URL looks like it was generated (e.g., 'zig-and-go.s3.amazonaws.com'), it is likely a hallucination and MUST be ignored.\n"
            f"6. **STRICT DOMAIN MATCH**: The image_url  be from a real, active domain. NEVER suggest fake domains.\n\n"
            f"7. **AVOID UNRELATED IMAGES**: Do NOT grab random images from the page (e.g. author photos, sponsor ads) just to fill the image_url field. Return null if you are not 100% sure the image is the product.\n\n"
            f"8. **REQUIRE ACTUAL PRODUCT NAME**: The 'name' field MUST be the full descriptive name of the product, not just 'macbook' or 'apple'.\n\n"
            f"Markdown:\n{truncated_content}"
        )
        try:
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    system="You are a strict data extractor. Return a JSON list of products. Return ONLY the JSON object. NEVER hallucinate or make up URLs based on product names. If no URL is found, return null for that field.",
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            data = self._safe_json_parse(response.content[0].text, "products")
            products = data if isinstance(data, list) else data.get("products", [])
            print(f"DEBUG: Kimi raw extraction (truncated): {str(products)[:500]}...")
            
            # Basic normalization of extracted products
            for p in products:
                if "image_url" in p:
                    p["image_url"] = self._normalize_url(p["image_url"])
                if "url" in p:
                    p["url"] = self._normalize_url(p["url"])
                
            return products
        except Exception as e:
            print(f"Error in extract_product_data: {e}")
            return []
kimi_service = KimiService()
