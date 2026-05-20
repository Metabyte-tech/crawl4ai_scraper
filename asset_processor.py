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
        
        self.proxy_url = os.getenv("PROXY_URL")
        # Initialize client without specific headers as we'll set them per request
        if self.proxy_url:
            print(f"DEBUG: AssetProcessor using proxy: {self.proxy_url}")
            self.client = httpx.Client(timeout=30.0, verify=False, proxy=self.proxy_url)
        else:
            self.client = httpx.Client(timeout=30.0, verify=False)
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
    def process_product_images(self, products, category="products", subcategory="general"):
        """
        Iterates through products, downloads images from external URLs,
        uploads them to S3, and updates the product metadata with S3 URLs.
        """
        processed_products = []
        for product in products:
            image_url = product.get("image_url")
            if image_url:
                # Save as fallback before any modifications
                if "original_image_url" not in product:
                    product["original_image_url"] = image_url
                
                # Try to use product's source URL to resolve relative images
                source_url = product.get("source_url") or product.get("url")
                if image_url.startswith("/") and source_url:
                    from urllib.parse import urljoin
                    image_url = urljoin(source_url, image_url)
                elif not image_url.startswith("http") and source_url:
                    # Could be something like "assets/img.jpg"
                    from urllib.parse import urljoin
                    image_url = urljoin(source_url, image_url)
                else:
                    image_url = kimi_service._normalize_url(image_url)
                
                # 1. AWS/Amazon Thumbnail Cleaning Removed
                # Previously, we aggressively upscaled Amazon thumbnails by removing `._AC_SY200_.` parameters.
                # This caused 404 Not Found errors on S3 upload, which triggered seed image fallbacks.
                # We now strictly use the reliable raw image url exactly as extracted.
                
                # 2. Ajio Domain Repair - assets.ajio.com is often blocked/404
                # assets-jiocdn.ajio.com is the persistent production CDN
                if "assets.ajio.com" in image_url:
                    image_url = image_url.replace("assets.ajio.com", "assets-jiocdn.ajio.com")
                    print(f"DEBUG: Repaired Ajio URL: {image_url}")
                
                product["image_url"] = image_url
                
                # Strict Filtering: Only process actual image files
                clean_url = image_url.split('?')[0].lower()
                is_image = any(clean_url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif'])
                
                # Filter out obvious logos/sprites/buttons based on URL
                logolike_keywords = ["logo", "sprite", "icon", "banner", "header", "footer", "favicon", "button", "spacer", "nav_", "ui_", "menu"]
                is_logolike = any(kw in image_url.lower() for kw in logolike_keywords)
                
                from image_cache import image_cache
                
                # Check cache before doing any network requests
                cached_s3 = image_cache.get_s3_url(image_url)
                if cached_s3:
                    if cached_s3 == "SKIP":
                        print(f"INFO: IMAGE CACHE HIT (Negative Cache). Ignoring skipped image: {image_url}")
                        processed_products.append(product)
                        continue
                    print(f"INFO: IMAGE CACHE HIT. Skipping download for {image_url}")
                    product["s3_image_url"] = cached_s3
                    product["original_image_url"] = image_url
                    processed_products.append(product)
                    continue

                # Validate hostname - skip malformed/relative URLs that slipped through
                # e.g., https://pub/images/... or https://./something.png
                try:
                    from urllib.parse import urlparse as _urlparse
                    _parsed = _urlparse(image_url)
                    _host = _parsed.netloc or ""
                    # A valid hostname must contain a dot and be longer than 3 chars total
                    _is_valid_host = "." in _host and len(_host) > 3
                except Exception:
                    _is_valid_host = False

                if not _is_valid_host:
                    print(f"SKIP: Malformed/relative URL has invalid hostname: {image_url}")
                    processed_products.append(product)
                    continue

                if image_url.startswith("http") and is_image and not is_logolike:
                    try:
                        print(f"INFO: Attempting to download image: {image_url}")
                        # Use rotating stealth headers for each request
                        headers = self._get_headers(image_url)
                        response = None
                        max_retries = 2
                        for attempt in range(max_retries + 1):
                            try:
                                if attempt > 0:
                                    # Fresh client on retry - avoids stale keep-alive connections
                                    import time as _time
                                    _time.sleep(0.5 * attempt)
                                    with httpx.Client(follow_redirects=True, http2=False) as retry_client:
                                        response = retry_client.get(image_url, timeout=12.0, headers=self._get_headers(image_url))
                                else:
                                    response = self.client.get(image_url, timeout=10.0, headers=headers)
                                break  # Success
                            except Exception as req_e:
                                if attempt < max_retries:
                                    print(f"WARNING: Image download attempt {attempt+1} failed for {image_url}: {req_e}. Retrying...")
                                else:
                                    print(f"WARNING: Image download failed after {max_retries+1} attempts for {image_url}: {req_e}. Skipping.")
                                    processed_products.append(product)
                        if response is None:
                            continue
                        
                        # SIZE FILTER: Skip images under 1KB (likely tiny invisible pixels)
                        content_len = len(response.content)
                        if response.status_code == 200 and content_len < 1000:
                            print(f"SKIP: Image too small ({content_len} bytes), likely a logo or icon: {image_url}")
                            image_cache.save_s3_url(image_url, "SKIP")
                            continue
                        print(f"INFO: Image download status: {response.status_code} ({content_len} bytes)")
                        
                        if response.status_code != 200 and "original_image_url" in product:
                             # Don't split on '?' for Shopify URLs as they might need v=...
                             image_url = product["original_image_url"]
                             print(f"WARNING: Initial URL failed ({response.status_code}). Retrying with original: {image_url}")
                             headers = self._get_headers(image_url)
                             response = self.client.get(image_url, timeout=10.0, headers=headers)
                             print(f"INFO: Original image download status: {response.status_code}")
                        if response.status_code == 200:
                            # Generate a unique file name with category structure
                            ext = image_url.split(".")[-1].split("?")[0]
                            if len(ext) > 4: ext = "jpg" # Fallback
                            
                            # Use categorized structure for S3
                            filename = f"products/{category}/{subcategory}/{uuid.uuid4()}.{ext}"
                            
                            # Upload to S3
                            s3_url = s3_service.upload_image(
                                response.content, 
                                filename,
                                content_type=response.headers.get("Content-Type", "image/jpeg")
                            )
                            
                            if s3_url:
                                product["s3_image_url"] = s3_url
                                # Keep original as backup or reference
                                product["original_image_url"] = image_url
                                # Save to DB Cache
                                image_cache.save_s3_url(image_url, s3_url)
                            else:
                                print(f"WARNING: S3 upload failed for {image_url}")
                        else:
                            print(f"WARNING: All download attempts failed for {image_url}")
                            image_cache.save_s3_url(image_url, "SKIP")
                    except httpx.ConnectError as e:
                        print(f"ERROR: DNS/Connection failure for {image_url}: {e}")
                    except Exception as e:
                        print(f"ERROR: Failed to process image {image_url}: {e}")
                        import traceback
                        traceback.print_exc()
            
            processed_products.append(product)
        
        return processed_products
        
    async def process_raw_content(self, content, base_url=None, category="uncategorized", subcategory="general"):
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
                from urllib.parse import urljoin
                full_url = urljoin(base_url, url) if base_url else url
                
                # Prepare a mini-product for existing logic
                mini_products = [{"image_url": full_url}]
                processed = self.process_product_images(mini_products, category, subcategory)
                if processed and processed[0].get("s3_image_url"):
                    s3_url = processed[0]["s3_image_url"]
                    content = content.replace(url, s3_url)
                    if not first_s3_url: first_s3_url = s3_url
            except Exception:
                continue
                
        return content, first_s3_url
asset_processor = AssetProcessor()
