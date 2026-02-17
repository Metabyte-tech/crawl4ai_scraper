import os
import json
import asyncio
import time
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class KimiService:
    def __init__(self):
        self.client = Anthropic(
            api_key=os.getenv("MOONSHOT_API_KEY"),
        )
        self.model = "claude-3-haiku-20240307"
        # Strict concurrency: 1 request at a time to stay under 50k tokens/min
        self.semaphore = asyncio.Semaphore(1) 

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
                import re
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1).strip()
            
            # Handle trailing commas in JSON lists/objects (common LLM error)
            # This is a bit risky but often saves a malformed response
            import re
            text = re.sub(r',\s*([\]}])', r'\1', text)
            
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                # Attempt to find the first [ or { and last ] or }
                start_braces = text.find('{')
                start_brackets = text.find('[')
                
                indices = [i for i in [start_braces, start_brackets] if i != -1]
                if indices:
                    start_idx = min(indices)
                    end_braces = text.rfind('}')
                    end_brackets = text.rfind(']')
                    end_idx = max(i for i in [end_braces, end_brackets] if i != -1)
                    
                    if start_idx < end_idx:
                        potential_json = text[start_idx:end_idx+1]
                        # Apply trailing comma fix here as well
                        potential_json = re.sub(r',\s*([\]}])', r'\1', potential_json)
                        return json.loads(potential_json)
            except Exception as e:
                print(f"Failed to extract JSON from malformed text: {e}")
        
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

    async def search_stores(self, location, product_type):
        """
        Uses Kimi's native reasoning to identify top retail URLs.
        """
        prompt = f"Find the best 5 kids {product_type} retail store website URLs in {location}. Prioritize popular Indian retailers like Amazon.in, Flipkart, Ajio, FirstCry, Myntra, and Nykaa Fashion. IMPORTANT: Only provide valid, direct, and real online shopping URLs. NEVER provide 'example.com' or placeholder URLs. Output ONLY a JSON list."
        
        async with self.semaphore:
            try:
                response = await self._call_with_retry(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system="You are a retail discovery expert. Output a JSON list of REAL URLs. NEVER hallucinate URLs. Return ONLY the JSON object.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                )
                data = self._safe_json_parse(response.content[0].text, "urls")
                urls = data if isinstance(data, list) else data.get("urls", [])
                
                # Validation: Filter out hallucinated/dummy URLs
                valid_urls = [u for u in urls if isinstance(u, str) and "example.com" not in u and "placeholder" not in u and "dummy" not in u]
                return valid_urls
            except Exception as e:
                print(f"Error in search_stores: {e}")
                return []

    async def extract_product_data(self, markdown_content):
        """
        Extracts structured product data. Truncates content to fit token limits.
        """
        # Further truncate content to ~8k characters to avoid 50k token/min limit
        truncated_content = markdown_content[:8000]
        prompt = f"Extract all kids products from this markdown. For each product, include: name, price (convert to number), currency (default to INR/₹), age_group, brand, absolute image URL, and the direct product page URL (link) if available.\n\nIMPORTANT: Only extract URLs that are EXPLICITLY present in the markdown. NEVER hallucinate or make up URLs like example.com. If no product URL is found, omit the 'url' field.\n\nMarkdown:\n{truncated_content}"
        
        async with self.semaphore:
            try:
                response = await self._call_with_retry(
                    lambda: self.client.messages.create(
                        model=self.model,
                        max_tokens=2048,
                        system="You are a structured data extractor. Return a JSON list of products. Return ONLY the JSON object. NEVER hallucinate URLs.",
                        messages=[{"role": "user", "content": prompt}],
                    )
                )
                data = self._safe_json_parse(response.content[0].text, "products")
                products = data if isinstance(data, list) else data.get("products", [])
                
                # Validation: Cleanup hallucinated URLs in products
                for p in products:
                    if "image_url" in p and ("example.com" in str(p["image_url"]) or "dummy" in str(p["image_url"])):
                        p["image_url"] = None
                    if "url" in p and ("example.com" in str(p["url"]) or "dummy" in str(p["url"])):
                        p["url"] = None
                        
                return products
            except Exception as e:
                print(f"Error in extract_product_data: {e}")
                return []

kimi_service = KimiService()
