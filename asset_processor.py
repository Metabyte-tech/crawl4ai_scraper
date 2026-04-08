import asyncio
import httpx
import uuid
from s3_service import s3_service
from kimi_service import kimi_service
class AssetProcessor:
    def __init__(self):
        import os
        import random
        # List of realistic user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36"
        ]
        
        self.default_placeholder = "https://placehold.co/600x600?text=No+Image"
        self.proxy_url = os.getenv("PROXY_URL")
        # Initialize async client
        if self.proxy_url:
            print(f"DEBUG: AssetProcessor using proxy: {self.proxy_url}", flush=True)
            self.client = httpx.AsyncClient(timeout=30.0, verify=False, proxy=self.proxy_url)
        else:
            self.client = httpx.AsyncClient(timeout=30.0, verify=False)
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
    async def process_product_images(self, products, category="products", subcategory="general"):
        """
        Iterates through products, downloads images from external URLs,
        uploads them to S3, and updates the product metadata with S3 URLs.
        """
        async def process_single_product(product):
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
                    try:
                        # Use a quick HEAD request to verify existence
                        test_res = await self.client.head(recovered_url, timeout=5.0)
                        if test_res.status_code == 200:
                            image_url = recovered_url
                    except Exception:
                        pass
            
            # 2. Ajio Domain Repair
            if "assets.ajio.com" in image_url:
                image_url = image_url.replace("assets.ajio.com", "assets-jiocdn.ajio.com")
            
            product["image_url"] = image_url
            
            IMAGE_CDN_DOMAINS = [
                "th.bing.com", "tse1.mm.bing.net", "tse2.mm.bing.net",
                "tse3.mm.bing.net", "tse4.mm.bing.net",
                "m.media-amazon.com", "images-amazon.com",
                "cdn.shopify.com", "i.imgur.com",
                "images.unsplash.com", "lh3.googleusercontent.com",
            ]
            
            clean_url = image_url.split('?')[0].lower()
            has_image_ext = any(clean_url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif'])
            is_cdn_image = any(cdn in image_url.lower() for cdn in IMAGE_CDN_DOMAINS)
            
            # RELAXED FILTER: If it looks like an image URL, or is from a CDN, or even if it lacks extension
            # we will verify with a HEAD request if no extension is present.
            is_image = has_image_ext or is_cdn_image
            
            # Filter out obvious logos/sprites based on URL
            logolike_keywords = ["logo", "sprite", "icon", "banner", "header", "footer", "favicon"]
            is_logolike = any(kw in image_url.lower() for kw in logolike_keywords)
            
            from image_cache import image_cache
            cached_s3 = image_cache.get_s3_url(image_url)
            if cached_s3:
                product["s3_image_url"] = cached_s3
                return product

            # If it lacks extension but is from a shopping site, let's try a HEAD request to be sure
            if not is_image and not is_logolike and image_url.startswith("http"):
                try:
                    head_res = await self.client.head(image_url, timeout=5.0)
                    content_type = head_res.headers.get("Content-Type", "").lower()
                    if "image" in content_type:
                        is_image = True
                except:
                    pass

            if image_url.startswith("http") and is_image and not is_logolike:
                try:
                    headers = self._get_headers(image_url)
                    response = await self.client.get(image_url, timeout=15.0, headers=headers)
                    
                    content_len = len(response.content)
                    if response.status_code == 200 and content_len < 1000:
                        return product # Skip small
                    
                    if response.status_code != 200 and "original_image_url" in product:
                         image_url = product["original_image_url"]
                         headers = self._get_headers(image_url)
                         response = await self.client.get(image_url, timeout=15.0, headers=headers)

                    if response.status_code == 200:
                        ext = image_url.split(".")[-1].split("?")[0]
                        if len(ext) > 4 or "/" in ext: ext = "jpg"
                        
                        filename = f"products/{category}/{subcategory}/{uuid.uuid4()}.{ext}"
                        s3_url = s3_service.upload_image(
                            response.content, 
                            filename,
                            content_type=response.headers.get("Content-Type", "image/jpeg")
                        )
                        
                        if s3_url:
                            product["s3_image_url"] = s3_url
                            image_cache.save_s3_url(image_url, s3_url)
                except Exception as e:
                    print(f"ERROR: Failed to process image {image_url}: {e}", flush=True)

            return product

        # Use Semaphore to limit parallel downloads to avoid IP blocks or memory spikes
        semaphore = asyncio.Semaphore(5)
        async def sem_process(p):
            async with semaphore:
                return await process_single_product(p)

        tasks = [sem_process(p) for p in products]
        return await asyncio.gather(*tasks)
        
    async def process_raw_content(self, content, category="uncategorized", subcategory="general"):
        """
        Scans raw markdown for images, uploads them to S3, and returns cleaned content and first S3 image.
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
                processed = await self.process_product_images(mini_products, category, subcategory)
                if processed and processed[0].get("s3_image_url"):
                    s3_url = processed[0]["s3_image_url"]
                    content = content.replace(url, s3_url)
                    if not first_s3_url: first_s3_url = s3_url
            except Exception:
                continue
                
        return content, first_s3_url
asset_processor = AssetProcessor()
