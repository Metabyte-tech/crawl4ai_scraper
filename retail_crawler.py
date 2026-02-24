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
        
        # 2. Concurrent Intelligent Extraction & Asset Processing
        async def process_single_page(page):
            current_url = page.get("url")
            content = page.get("content", "")
            
            # Extract category/subcategory for this specific page
            page_cat, page_sub = self._extract_category_info(current_url)
            
            # Extract structured data via Kimi
            products = await kimi_service.extract_product_data(content, target_category=target_category)
            print(f"DEBUG: Kimi found {len(products or [])} raw products on {current_url}")
            if not products:
                return []
                
            processed_products = asset_processor.process_product_images(
                products, 
                category=page_cat, 
                subcategory=page_sub
            )
            
            final_products = []
            for p in processed_products:
                if p.get("s3_image_url"):
                    p["image_url"] = p["s3_image_url"]
                    raw_url = p.get("url") or current_url
                    p["source_url"] = kimi_service._normalize_url(raw_url)
                    p["category"] = page_cat
                    p["subcategory"] = page_sub
                    
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
