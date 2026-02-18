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

    def process_product_images(self, products):
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
                
                # Normalize the URL before processing
                image_url = kimi_service._normalize_url(image_url)
                
                # 1. AWS/Amazon Thumbnail Cleaning - Aggressive Recovery
                if "m.media-amazon.com" in image_url and "._" in image_url:
                    import re
                    # Remove all thumbnail tags like ._AC_SY200_., ._SX450_., etc.
                    # Pattern matches everything between ._ and the file extension dot
                    recovered_url = re.sub(r'\._[^/]*\.', '.', image_url)
                    if recovered_url != image_url:
                        print(f"DEBUG: Recovered high-res Amazon image: {recovered_url}")
                        image_url = recovered_url
                
                # 2. Ajio Domain Repair - assets.ajio.com is often blocked/404
                # assets-jiocdn.ajio.com is the persistent production CDN
                if "assets.ajio.com" in image_url:
                    image_url = image_url.replace("assets.ajio.com", "assets-jiocdn.ajio.com")
                    print(f"DEBUG: Repaired Ajio URL: {image_url}")
                
                product["image_url"] = image_url
                
                # Strict Filtering: Only process actual image files
                is_image = any(image_url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.avif'])
                if image_url.startswith("http") and is_image:
                    try:
                        print(f"INFO: Attempting to download image: {image_url}")
                        # Use rotating stealth headers for each request
                        headers = self._get_headers(image_url)
                        response = self.client.get(image_url, timeout=10.0, headers=headers)
                        print(f"INFO: Image download status: {response.status_code}")
                        
                        if response.status_code != 200 and "original_image_url" in product:
                             print(f"WARNING: Cleaned URL failed ({response.status_code}). Retrying original: {product['original_image_url']}")
                             image_url = product["original_image_url"]
                             headers = self._get_headers(image_url)
                             response = self.client.get(image_url, timeout=10.0, headers=headers)
                             print(f"INFO: Original image download status: {response.status_code}")

                        if response.status_code == 200:
                            # Generate a unique file name
                            ext = image_url.split(".")[-1].split("?")[0]
                            if len(ext) > 4: ext = "jpg" # Fallback
                            filename = f"products/{uuid.uuid4()}.{ext}"
                            
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
                            else:
                                print(f"WARNING: S3 upload failed for {image_url}")
                        else:
                            print(f"WARNING: All download attempts failed for {image_url}")
                    except httpx.ConnectError as e:
                        print(f"ERROR: DNS/Connection failure for {image_url}: {e}")
                    except Exception as e:
                        print(f"ERROR: Failed to process image {image_url}: {e}")
                        import traceback
                        traceback.print_exc()
            
            processed_products.append(product)
        
        return processed_products

asset_processor = AssetProcessor()
