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
            "m.media-amazon.com", "assets.ajio.com", "cdn.fcglcdn.com"
            "m.media-amazon.com", "assets.ajio.com", "cdn.fcglcdn.com",
            "mxwholesale.co.uk", "brightminds.co.uk"
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
        """
        for i in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func)
            except Exception as e:
                if "429" in str(e) and i < max_retries - 1:
                    wait_time = (5 ** i) + 5 # More aggressive backoff
                    print(f"Rate limited (429). Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    async def search_sources(self, topic):
        """
        Uses Kimi's native reasoning to identify top authoritative URLs for any topic.
        """
        is_retail = any(kw in topic.lower() for kw in ["shoes", "kids", "shopping", "clothes", "toys", "products"])
        system_msg = "You are a research expert. Output a JSON list of REAL, official documentation or resource URLs. NEVER hallucinate URLs. Return ONLY the JSON object."
        if is_retail:
            system_msg = "You are a shopping expert. Find REAL product catalog or category pages from top retailers like Amazon, Ajio, Flipkart, Myntra. Output ONLY a JSON list of URLs."
        prompt = f"Find the best 5 authoritative and official website URLs for the topic: '{topic}'. Prioritize official documentation, GitHub repositories, or high-quality technical guides. IMPORTANT: Only provide valid, direct, and real URLs. Output ONLY a JSON list."
        async with self.semaphore:
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
        # Further truncate content to ~8k characters to avoid 50k token/min limit
        truncated_content = markdown_content[:8000]
        prompt = (
            f"system_msg = \"You are a surgical data extraction tool. Extract product information ONLY from the provided markdown. DO NOT invent URLs. If an image link is not explicitly shown in the text (like ![alt](url) or <img src='url'>), you MUST return null. NEVER guess based on product names.\"\n\n"
            f"Identify and extract all products or items related to '{target_category}' from this markdown. For each item, include: name, price (convert to number), "
            f"currency (default to INR/₹), age_group (if applicable), brand, image_url, and the direct product page link (absolute URL).\n\n"
            f"CRITICAL PRODUCT RULES:\n"
            f"1. **RELEVANCE**: Only extract items that truly belong to the category '{target_category}'.\n"
            f"2. **REJECT NON-PRODUCTS**: Do NOT extract logos, advertising banners, site navigation icons, or promotional bundles that aren't the main items.\n\n"
            f"CRITICAL URL RULES:\n"
            f"1. **STRICT EXTRACTION**: ONLY use image URLs explicitly found in <img> tags or markdown image links (![...](...)).\n"
            f"2. **LOGO PREVENTION**: NEVER extract URLs containing 'logo', 'sprite', 'icon', 'banner', 'nav', or 'header' as product images.\n"
            f"3. **ZERO TOLERANCE FOR HALLUCINATION**: If the markdown doesn't have a working image link, set image_url to null. NEVER combine names to make a URL like 'product-name.com/img.jpg'.\n"
            f"4. **AVOID PLACEHOLDERS**: NEVER extract base64 data-URIs, '1x1.gif', or 'pixel.gif' as image_url.\n"
            f"5. **REAL DOMAINS ONLY**: Favor URLs from known CDNs. If a URL looks like it was generated (e.g., 'zig-and-go.s3.amazonaws.com'), it is likely a hallucination and MUST be ignored.\n\n"
            f"Markdown:\n{truncated_content}"
        )
        async with self.semaphore:
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
