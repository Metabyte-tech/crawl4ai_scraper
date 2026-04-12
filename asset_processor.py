import httpx
import uuid
from s3_service import s3_service
from kimi_service import kimi_service
class AssetProcessor:
    def __init__(self):
        import os
        import random
        import asyncio
        # List of realistic user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
        ]
        
        self.proxy_url = os.getenv("PROXY_URL")
        self.semaphore = asyncio.Semaphore(50) # Limit concurrent downloads per worker
        
        limits = httpx.Limits(max_connections=500, max_keepalive_connections=50)
        
        # Initialize client without specific headers as we'll set them per request
        if self.proxy_url:
            print(f"DEBUG: AssetProcessor using proxy: {self.proxy_url}")
            self.client = httpx.AsyncClient(timeout=30.0, verify=False, proxy=self.proxy_url, limits=limits)
        else:
            self.client = httpx.AsyncClient(timeout=30.0, verify=False, limits=limits)
    def _get_headers(self, url=None):
        import random
        ua = random.choice(self.user_agents)
        domain = "www.google.com"
        if url:
             from urllib.parse import urlparse
             domain = urlparse(url).netloc
        
        return {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"https://{domain}/",
            "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "image",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-site",
            "Upgrade-Insecure-Requests": "1"
        }
    async def process_product_images(self, products, category="products", subcategory="general", source="other", scrape_date=None):
        """
        Iterates through products, downloads images from external URLs,
        uploads them to S3 using a website/category/date structure,
        and updates the product metadata with S3 URLs.
        Parallelized using asyncio.gather.
        """
        import datetime
        import asyncio
        if not scrape_date:
            scrape_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        safe_source = str(source).replace(" ", "_").lower()
        
        async def process_single(product):
            image_url = product.get("image_url")
            if not image_url:
                return product
                
            # Save as fallback before any modifications
            if "original_image_url" not in product:
                product["original_image_url"] = image_url
            
            # Normalize the URL before processing
            image_url = kimi_service._normalize_url(image_url)
            
            # 1. AWS/Amazon Thumbnail Cleaning - Aggressive Recovery
            if "m.media-amazon.com" in image_url and "._" in image_url:
                import re
                recovered_url = re.sub(r'\._[^/]*\.', '.', image_url)
                if recovered_url != image_url:
                    image_url = recovered_url
            
            # 2. eBay Thumbnail Cleaning - Upgrade to s-l500
            if "ebayimg.com" in image_url and "s-l" in image_url:
                import re
                recovered_url = re.sub(r's-l\d+', 's-l500', image_url)
                if recovered_url != image_url:
                    image_url = recovered_url

            # 3. Ajio Domain Repair
            if "assets.ajio.com" in image_url:
                image_url = image_url.replace("assets.ajio.com", "assets-jiocdn.ajio.com")
            
            product["image_url"] = image_url
            
            # Strict Filtering
            clean_url = image_url.split('?')[0].lower()
            is_image = any(clean_url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.avif']) or "bing.net" in image_url or "m.media-amazon" in image_url or "image" in image_url.lower()
            
            logolike_keywords = ["logo", "sprite", "icon", "banner", "header", "footer", "favicon"]
            is_logolike = any(kw in image_url.lower() for kw in logolike_keywords)
            
            from image_cache import image_cache
            
            # Check cache
            cached_s3 = image_cache.get_s3_url(image_url)
            if cached_s3:
                product["s3_image_url"] = cached_s3
                return product

            if image_url.startswith("http") and is_image and not is_logolike:
                try:
                    async with self.semaphore:
                        headers = self._get_headers(image_url)
                        response = await self.client.get(image_url, timeout=10.0, headers=headers)
                        
                        if response.status_code != 200 and "original_image_url" in product:
                             image_url = product["original_image_url"]
                             headers = self._get_headers(image_url)
                             response = await self.client.get(image_url, timeout=10.0, headers=headers)

                    if response.status_code == 200:
                        content_len = len(response.content)
                        if content_len < 1000:
                            return product
                            
                        content_type = response.headers.get("Content-Type", "").lower()
                        if "gif" in content_type or not content_type.startswith("image/"):
                            return product
                            
                        ext = image_url.split(".")[-1].split("?")[0]
                        if len(ext) > 4: ext = content_type.split("/")[-1] if "/" in content_type else "jpg"
                        
                        import uuid
                        filename = f"{safe_source}/{category}/{scrape_date}/{uuid.uuid4()}.{ext}"
                        
                        from s3_service import s3_service
                        s3_url = await s3_service.upload_image_async(
                            response.content, 
                            filename,
                            content_type=response.headers.get("Content-Type", "image/jpeg")
                        )
                        
                        if s3_url:
                            product["s3_image_url"] = s3_url
                            image_cache.save_s3_url(image_url, s3_url)
                except Exception as e:
                    print(f"DEBUG: Async image processing error for {image_url}: {e}")
            
            return product

        # Process all products in parallel
        tasks = [process_single(p) for p in products]
        processed_products = await asyncio.gather(*tasks)
        return list(processed_products)
        
    async def process_raw_content(self, content, category="uncategorized", subcategory="general", source="other", scrape_date=None):
        """
        Scans raw markdown for images, uploads them to S3, and returns cleaned content and first S3 image.
        Uses categorized folder structure.
        """
        import re
        # Find all markdown images: ![alt](url)
        img_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
        # Find all HTML images: <img src="..." ...> OR <img data-src="..." ...> etc.
        html_matches = re.findall(r'<img.*? (?:src|data-src|data-original|data-lazy)=["\'](.*?)["\']', content, flags=re.IGNORECASE)
        
        all_urls = list(set(img_matches + html_matches))
        if not all_urls:
            return content, None
            
        first_s3_url = None
        for url in all_urls:
            # Skip if already an S3 URL
            if "amazonaws.com" in url:
                if not first_s3_url: first_s3_url = url
                continue
                
            try:
                # Prepare a mini-product for existing logic
                mini_products = [{"image_url": url}]
                processed = await self.process_product_images(mini_products, category, subcategory, source, scrape_date)
                if processed and processed[0].get("s3_image_url"):
                    s3_url = processed[0]["s3_image_url"]
                    content = content.replace(url, s3_url)
                    if not first_s3_url: first_s3_url = s3_url
            except Exception:
                continue
                
        return content, first_s3_url
asset_processor = AssetProcessor()
