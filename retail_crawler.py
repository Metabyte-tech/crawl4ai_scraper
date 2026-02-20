import asyncio
from crawler import crawl_site_recursive
from kimi_service import kimi_service
from asset_processor import asset_processor
from ingest import add_multiple_contents_to_store

class RetailCrawler:
    def __init__(self):
        self.max_pages = 10

    async def sync_store(self, seed_url, max_pages=None, target_category="relevant"):
        """
        Orchestrates the discovery -> crawl -> extract -> upload -> ingest loop.
        """
        limit = max_pages or self.max_pages
        print(f"Starting sync for: {seed_url} (limit: {limit} pages, category: {target_category})")
        
        # 1. Recursive Crawl
        pages = await crawl_site_recursive(seed_url, max_pages=limit)
        if not pages:
            print(f"No pages found for {seed_url}")
            return []
        
        all_retail_data = []
        
        # 2. Concurrent Intelligent Extraction & Asset Processing
        async def process_single_page(page):
            # Use local variables to avoid scoping issues with async tasks
            current_url = page.get("url")
            content = page.get("content", "")
            
            print(f"DEBUG: Processing page {current_url} (Content length: {len(content)})")
            # Log snippet only to avoid file conflict in parallel runs
            print(f"DEBUG: Content snippet: {content[:100]}...")
            
            # Run blocking extraction in a thread to keep async loop free if needed, 
            # but KimiService uses httpx/anthropic sync client for now.
            # Extract structured data via Kimi (Now async)
            products = await kimi_service.extract_product_data(content, target_category=target_category)
            if not products:
                return []
                
            processed_products = asset_processor.process_product_images(products)
            
            final_products = []
            for p in processed_products:
                # CRITICAL: Only accept products where S3 sync actually worked
                # This prevents "Not Found" images on the frontend.
                if p.get("s3_image_url"):
                    # Move S3 URL to the primary image_url field for the bot
                    p["image_url"] = p["s3_image_url"]
                    
                    # Store original as reference (optional)
                    # p["original_source_image"] = p.get("original_image_url")
                    
                    # Ensure source_url is normalized correctly
                    raw_url = p.get("url") or current_url
                    p["source_url"] = kimi_service._normalize_url(raw_url)
                    
                    # Remove temporary internal keys
                    for key in ["url", "s3_image_url", "original_image_url"]:
                        if key in p: del p[key]
                        
                    final_products.append(p)
                
            return final_products

        # Process all pages in parallel
        tasks = [process_single_page(page) for page in pages]
        results = await asyncio.gather(*tasks)
        
        for product_list in results:
            all_retail_data.extend(product_list)
        
        # 3. Ingestion into ChromaDB
        if all_retail_data:
            # Prepare for ingest tool
            ingest_items = []
            for product in all_retail_data:
                # IMPORTANT: Include EVERYTHING in the content string so the LLM can see it
                description = (
                    f"Product: {product.get('name')}\n"
                    f"Brand: {product.get('brand')}\n"
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
            
            add_multiple_contents_to_store(ingest_items)
            print(f"Successfully synced {len(all_retail_data)} products from {seed_url}")
            return all_retail_data
        
        return []

retail_crawler = RetailCrawler()
