import asyncio
from crawler import crawl_site_recursive
from kimi_service import kimi_service
from asset_processor import asset_processor
from ingest import add_multiple_contents_to_store

class RetailCrawler:
    def __init__(self):
        self.max_pages = 50

    def _extract_category_info(self, url):
        """
        Extracts category and subcategory from the URL path.
        """
        try:
            from urllib.parse import urlparse
            path_parts = [p for p in urlparse(url).path.split('/') if p]
            category = "uncategorized"
            subcategory = "general"
            if len(path_parts) >= 1:
                category = path_parts[0]
            if len(path_parts) >= 2:
                subcategory = path_parts[1]
            return category, subcategory
        except Exception:
            return "uncategorized", "general"

    async def sync_store(self, seed_url, max_pages=None, target_category="relevant"):
        """
        Orchestrates the discovery -> crawl -> extract -> upload -> ingest loop.
        """
        limit = max_pages or self.max_pages
        print(f"Starting sync for: {seed_url} (limit: {limit} pages, category: {target_category})")
        
        # 0. Initial Category from Seed URL
        seed_cat, seed_sub = self._extract_category_info(seed_url)

        # 1. Recursive Crawl
        pages = await crawl_site_recursive(seed_url, max_pages=limit)
        if not pages:
            print(f"No pages found for {seed_url}")
            return []
        
        all_retail_data = []

        def _find_image_near_keyword(content: str, keyword: str):
            """Return the first image URL near the keyword in the markdown/HTML content."""
            import re
            keyword = keyword.lower().strip()
            if not keyword:
                return None

            # Find all image URLs in the content
            images = re.findall(r'!\[.*?\]\((.*?)\)', content) + re.findall(r'<img[^>]+(?:src|data-src|data-original|data-lazy)=["\'](.*?)["\']', content, flags=re.IGNORECASE)
            if not images:
                return None

            # Find the first occurrence of the keyword and look nearby for an image URL
            idx = content.lower().find(keyword)
            if idx == -1:
                return images[0]

            window = 500  # characters around the keyword
            start = max(0, idx - window)
            end = min(len(content), idx + window)
            snippet = content[start:end].lower()

            for img in images:
                if img.lower() in snippet:
                    return img

            # If none found nearby, just return first image as fallback
            return images[0]

        def _find_meta_image(content: str):
            """Try to find an OG/Twitter image from page metadata."""
            import re
            # Common tags: <meta property="og:image" content="..." />, <meta name="twitter:image" content="..." />
            match = re.search(r'<meta\s+(?:property|name)=["\'](?:og:image|twitter:image)["\']\s+content=["\'](.*?)["\']', content, flags=re.IGNORECASE)
            if match:
                return match.group(1)
            return None

        # 2. Concurrent Intelligent Extraction & Asset Processing
        async def process_single_page(page):
            current_url = page.get("url")
            content = page.get("content", "")
            
            # Extract category/subcategory for this specific page
            page_cat, page_sub = self._extract_category_info(current_url)
            
            # Extract structured data via Kimi
            products = await kimi_service.extract_product_data(content, target_category=target_category)
            print(f"DEBUG: Kimi found {len(products or [])} raw products on {current_url}")
            if products:
                missing = sum(1 for p in products if not p.get("image_url"))
                print(f"DEBUG: {missing}/{len(products)} products missing image_url")
            
            # NEW: Also ingest the RAW page content to ensure we have a fallback even if structured extraction fails
            # This uses the new async add_content_to_store which processes images globally for the page
            from ingest import add_content_to_store
            await add_content_to_store(content, {
                "source": current_url,
                "category": page_cat,
                "subcategory": page_sub,
                "type": "raw_retail_page"
            })

            # Attempt to assign a likely image URL for products missing one
            page_meta_img = _find_meta_image(content)
            if products:
                for p in products:
                    if not p.get("image_url") and p.get("name"):
                        inferred = _find_image_near_keyword(content, p.get("name"))
                        if inferred:
                            p["image_url"] = inferred
                        elif page_meta_img:
                            # Fallback to a generic page image if keyword-based lookup fails
                            p["image_url"] = page_meta_img

            # Normalize relative image URLs (e.g. '/images/..' or '//cdn...') to absolute URLs
            from urllib.parse import urljoin
            for p in (products or []):
                img = p.get("image_url")
                if img and not img.startswith("http"):
                    p["image_url"] = urljoin(current_url, img)

            # Process images for structured products if any were found
            processed_products = []
            if products:
                processed_products = asset_processor.process_product_images(
                    products, 
                    category=page_cat, 
                    subcategory=page_sub
                )
            
            final_products = []
            for p in processed_products:
                if p.get("s3_image_url"):
                    p["image_url"] = p["s3_image_url"]
                
                # Consistently ensure source_url exists
                raw_url = p.get("url") or current_url
                p["source_url"] = kimi_service._normalize_url(raw_url)
                p["category"] = page_cat
                p["subcategory"] = page_sub
                
                # Cleanup internal fields
                for key in ["url", "s3_image_url", "original_image_url"]:
                    if key in p: del p[key]
                final_products.append(p)
            print(f"DEBUG: Successfully processed {len(final_products)} products with S3 images from {current_url}")
            return final_products

        semaphore = asyncio.Semaphore(5)
        async def sem_process(page):
            async with semaphore:
                try:
                    return await asyncio.wait_for(process_single_page(page), timeout=60)
                except Exception as e:
                    print(f"ERROR: Timed out or failed processing {page.get('url')}: {e}")
                    return []

        tasks = [sem_process(page) for page in pages]
        results = await asyncio.gather(*tasks)
        for product_list in results:
            if product_list:
                all_retail_data.extend(product_list)
        
        if all_retail_data:
            ingest_items = []
            for product in all_retail_data:
                # ... (description formatting)
                description = (
                    f"Product: {product.get('name')}\n"
                    f"Brand: {product.get('brand')}\n"
                    f"Category: {product.get('category')} / {product.get('subcategory')}\n"
                    f"Price: {product.get('price')} {product.get('currency')}\n"
                    f"Details: {product.get('details')}\n"
                    f"Age: {product.get('age_group')}\n"
                    f"Image URL: {product.get('image_url')}\n"
                    f"Source URL: {product.get('source_url')}"
                )
                ingest_items.append({
                    "content": description,
                    "url": product.get("source_url"),
                    "metadata": product
                })
            await add_multiple_contents_to_store(ingest_items)
            print(f"Successfully synced {len(all_retail_data)} products from {seed_url}")
            return all_retail_data
        return []

retail_crawler = RetailCrawler()
