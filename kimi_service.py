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
            print(f"Error generating plan: {e}")
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
        
        # 🤖 Agent Task / Template execution detection
        # Templates usually start with specific "Professional" verbs or keywords
        agent_keywords = [
            "commercial feasibility", "market segment", "qualified suppliers", 
            "viral hits", "marketing strategy", "business architect", "report for"
        ]
        if any(x in q for x in agent_keywords):
            return "agent_task"

        # 🧸 Shopping / Products (Check this BEFORE vehicle to catch "car toys")
        shopping_keywords = [
            "toy", "gift", "miniature", "remote control", "rc ", "lego", "puzzle", "doll"
        ]
        if any(x in q for x in shopping_keywords):
            return "shopping"

        # 🚗 Vehicle / Mobility
        vehicle_keywords = [
            "car", "bike", "vehicle", "mileage", "scooter", "truck",
            "suv", "sedan", "hatchback", "coupe", "ev", "electric car",
            "thar", "mahindra", "toyota", "honda", "hyundai", "kia",
            "maruti", "suzuki", "ford", "chevrolet", "bmw", "mercedes",
            "audi", "volkswagen", "jeep", "defender", "land rover",
            "porsche", "ferrari", "lamborghini", "range rover", "tata",
            "nexon", "creta", "innova", "fortuner", "scorpio", "bolero",
            "swift", "brezza", "ertiga", "baleno", "i20", "venue",
            "xuv", "compass", "duster", "kwid", "redi-go", "harrier",
        ]
        if any(x in q for x in vehicle_keywords):
            return "vehicle"
        
        # � Informational / General
        # If it contains informational words, it should be "general" even if it has product keywords
        info_words = ["how", "why", "who", "what", "where", "tell", "explain", "list", "history", "about", "meaning", "definition"]
        if any(x in q for x in info_words):
            return "general"

        # �🛒 Shopping / Products
        shopping_keywords = [
            "buy", "price", "shop", "laptop", "phone", "macbook", "iphone", 
            "toy", "gift", "tshirt", "t-shirt", "shirt", "shoes", "shoe", 
            "cloth", "clothing", "wear", "jean", "pant", "fashion", "brand",
            "electronics", "gadget", "watch", "accessory", "bottle", "glass",
            "box", "bag", "lunch", "home", "kitchen", "furniture", "book", 
            "tool", "beauty", "care", "health", "product", "item", "unit", "set"
        ]
        if any(x in q for x in shopping_keywords):
            return "shopping"
        
        # 🔍 Heuristic for short product-like queries (e.g. "milk glass bottle 90ml")
        # If it's a short query with no info words (already checked above), it's likely a product search
        words = q.split()
        if 1 <= len(words) <= 5:
            # Avoid misclassifying greetings like "hi", "hello", "hey"
            greetings = ["hi", "hello", "hey", "hola", "namaste"]
            if len(words) == 1 and words[0] in greetings:
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
            print(f"DEBUG: Vehicle LLM start for {query}")
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
                print(f"DEBUG: Vehicle LLM returned all N/A for {query}. Falling back to images.")
                return await self.search_images(query)
            return result
        except Exception as e:
            print("Vehicle error:", e)
            return await self.search_images(query)

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
        
        # 2. Synchronous "Quick Extraction" for Top 2 Results
        # This provides real data for the first few items without waiting for the full deep crawl.
        top_urls = urls[:4]
        synced_products = []
        if top_urls:
            print(f"DEBUG: Performing sync extraction for top 4 results: {top_urls}")
            from crawler import crawl_site
            from crawl4ai import AsyncWebCrawler
            async with AsyncWebCrawler() as crawler:
                for idx, url in enumerate(top_urls):
                    try:
                        content, _ = await crawl_site(url, crawler=crawler)
                        if content:
                            clean = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL)
                            clean = re.sub(r"<style.*?</style>", "", clean, flags=re.DOTALL)
                            clean = re.sub(r"<[^>]+>", " ", clean)
                            extracted = await self.extract_product_data(clean, query, base_url=url)
                            if extracted:
                                # Use the first/most relevant one
                                p = extracted[0]
                                if not p.get("image_url") and idx < len(bing_images):
                                    p["image_url"] = bing_images[idx].get("image_url")
                                synced_products.append(p)
                    except Exception as e:
                        print(f"DEBUG: Sync extraction failed for {url}: {e}")

        synced_urls = [self._normalize_url(p.get("url") or p.get("source_url")) for p in synced_products if p]
        
        # 3. Combine results
        for p in synced_products:
            if p: fast_results.append(p)
            
        for idx, url in enumerate(urls):
            normalized = self._normalize_url(url)
            if normalized in synced_urls: continue
            
            img_url = bing_images[idx].get("image_url") if idx < len(bing_images) else None
            domain = urlparse(url).netloc.replace("www.", "")
            fast_results.append({
                "name": f"Product Option {idx+1}",
                "url": url,
                "source_url": url,
                "image_url": img_url,
                "price": "Check Site",
                "brand": "Verifying...",
                "source": domain.split('.')[0].capitalize(),
                "details": f"Finding the best price and details for this {query} from {domain}..."
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
        # PRO robust cleaning with BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # 1. Remove non-content elements
        for element in soup(["script", "style", "svg", "iframe", "canvas", "noscript", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        # 2. Remove common ad/nav containers by class/id
        for container in soup.find_all(attrs={"class": re.compile(r'nav|footer|sidebar|ad-|promo|header|menu|social|comment', re.I)}):
            container.decompose()

        # 3. Get clean text with structure preserved
        content = soup.get_text(separator=' ', strip=True)
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # SMART START: Search for prices or "Results" to skip the header
        # Skip small price markers (like currency switchers) by looking for the first price in a longer string
        start_idx = 0
        price_match = re.search(r'[\$£€₹]\d+', content) # Look for currency followed by digits
        if price_match:
            # Start 500 characters before the first price marker
            start_idx = max(0, price_match.start() - 500)
            print(f"DEBUG: Smart Start triggered at index {start_idx}")
        
        truncated_content = content[start_idx : start_idx + 60000] 
        prompt = (
            f"Extract ALL product details for '{target_category}' from the text.\n"
            f"Return a JSON list of objects with these exact fields:\n"
            f"- name: Concise product name\n"
            f"- price: Current price (with currency symbol e.g., ₹1,349 or $99). IMPORTANT: If price is not explicitly found, return null.\n"
            f"- brand: Brand name\n"
            f"- rating_avg: Numerical average rating (e.g., 4.5) - float or null\n"
            f"- rating_count: Number of reviews (e.g., 1250) - integer or null\n"
            f"- offers: Short summary of discounts\n"
            f"- source: Store name or platform\n"
            f"- image_url: Direct image URL\n"
            f"- url: Original product URL\n"
            f"- moq: Minimum Order Quantity (e.g., '100 units' or '1 pc') - string or null\n"
            f"- supplier_years: Number of years active on platform (e.g., '5 yrs') - string or null\n"
            f"- location: Origin location (e.g., 'CN', 'VN', 'IN') - string or null\n"
            f"- details: COMPREHENSIVE product description and features.\n"
            f"- reviews: A list of 3-5 REAL user comments/reviews found in the text. Each review MUST be an object: {{\"user\": \"name\", \"comment\": \"text\", \"rating\": 5}}\n"
            f"\nText:\n{truncated_content}"
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
            print(f"Error loading templates category: {e}")

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

            print(f"RAG: {len(all_docs)} raw → {len(context_docs)} relevant for '{search_query}'")
        except Exception as e:
            print(f"RAG search failed for report: {e}")

        # 2. If local DB has no relevant data, handle gracefully
        if not context_docs:
            print(f"No relevant local data for '{search_query}'. Using empty context.")
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
            print(f"DEBUG: Generating Agent Report for {template_id}")
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
            print(f"Error generating agent report: {e}")
            import traceback
            traceback.print_exc()
            return f"BACKEND_ERROR: {str(e)}"

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
            
    async def search_sources(self, query, intent="shopping", limit=10):
        # 1. Proactive Image Search to get HIGH QUALITY direct product URLs
        image_results = await self.search_images(query)
        image_urls = [r.get("source_url") for r in image_results.get("results", []) if r.get("source_url")]
        
        # 2. LLM Fallback for additional URLs
        system_msg = "You are a shopping expert. Find DIRECT product page URLs, not category or search result pages."
        prompt = (
            f"Find exactly {limit} DIRECT product listing URLs on major retail sites (Amazon, Walmart, Target, etc.) for: {query}.\n"
            f"CRITICAL: Return ONLY direct product pages (e.g. including '/dp/' or '/p/' or '/product/').\n"
            f"Do NOT return search result pages of the form '/s?k=' or '/search'.\n"
            f"Return ONLY a JSON list of strings."
        )
        
        try:
            print(f"DEBUG: Search sources LLM start for {query}")
            response = await self._call_with_retry(
                lambda: self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    system=system_msg,
                    messages=[{"role": "user", "content": prompt}],
                )
            )
            
            extracted_urls = []
            if response:
                text = response.content[0].text
                data = self._safe_json_parse(text, "urls")
                extracted_urls = data if isinstance(data, list) else data.get("urls", [])
                extracted_urls = [self._normalize_url(u) for u in extracted_urls if isinstance(u, str)]
            
            all_urls = []
            seen = set()
            for u in image_urls + extracted_urls:
                norm = self._normalize_url(u)
                if norm and norm not in seen:
                    all_urls.append(norm)
                    seen.add(norm)
            
            print(f"DEBUG: Found {len(all_urls)} combined product URLs.")
            return all_urls[:limit]
            
        except Exception as e:
            print("Search error:", e)
            return [self._normalize_url(u) for u in image_urls[:limit]]

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

    async def cache_and_store_products(self, products, query):
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
                print(f"✅ [BACKGROUND] Successfully cached {len(ingest_items)} products for '{query}'\n")
            
        except Exception as e:
            print(f"❌ [BACKGROUND] Error during caching: {e}")

kimi_service = KimiService()