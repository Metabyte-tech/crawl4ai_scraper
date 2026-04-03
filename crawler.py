import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from urllib.parse import urljoin, urlparse

EXCLUDED_KEYWORDS = ["login", "signup", "register", "cart", "checkout", "account", "profile", "wishlist", "help", "contact", "about", "privacy", "terms", "policy", "travel", "flights", "hotels", "bus", "train", "tickets"]

async def crawl_site(url: str, crawler=None):
    """
    Crawls a given URL using configuration compatible with crawl4ai 0.8.0.
    Accepts an optional crawler instance for reuse.
    """
    import os
    proxy_url = os.getenv("PROXY_URL")
    
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
        proxy=proxy_url if proxy_url else None
    )
    
    # Slower, more thorough scroll script with lazy-load attribute swap
    js_scroll = """
    (async () => {
        const swapImages = () => {
            document.querySelectorAll('img').forEach(img => {
                const lazyAttrs = ['data-src', 'data-original', 'data-lazy', 'data-srcset'];
                for (const attr of lazyAttrs) {
                    if (img.getAttribute(attr)) {
                        img.src = img.getAttribute(attr);
                    }
                }
            });
        };
        
        for (let i = 0; i < 5; i++) {
            window.scrollBy(0, window.innerHeight);
            swapImages();
            await new Promise(resolve => setTimeout(resolve, 1500));
        }
        swapImages(); // Final pass
    })();
    """

    # Use PruningContentFilter to strip headers, footers, and nav
    # This is much more effective than manual regex
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, min_word_threshold=50)
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        wait_for="body",
        page_timeout=90000,  # 90 seconds
        wait_for_timeout=60000, # 60s
        js_code=js_scroll,
        markdown_generator=md_generator
    )

    if crawler is None:
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                return await _do_crawl(crawler, url, run_config)
        except Exception as e:
            if "Browser.close" in str(e) or "closed" in str(e).lower():
                print(f"DEBUG: Browser closed unexpectedly during cleanup for {url}. Ignoring.")
                return None, [] # Or re-raise if this is not the desired behavior
            raise # Re-raise other exceptions
    else:
        return await _do_crawl(crawler, url, run_config)

async def _do_crawl(crawler, url, run_config):
    try:
        result = await crawler.arun(url=url, config=run_config)
        if result.success:
            # Return HTML for better structured extraction by LLM
            content = result.html
            if not content or len(content.strip()) < 500:
                print(f"HTML content short, retrying with explicit wait for {url}")
                run_config.wait_for = "js:() => document.body.innerText.length > 500"
                result = await crawler.arun(url=url, config=run_config)
                content = result.html
            print(f"Successfully crawled: {url}")
            return content, result.links
        else:
            print(f"Failed to crawl: {url}. Error: {result.error_message}")
            return None, []
    except Exception as e:
        print(f"Unexpected error crawling {url}: {e}")
        return None, []

async def crawl_site_recursive(base_url: str, max_pages: int = 20):
    """
    Concurrent recursive crawl starting from base_url up to max_pages.
    """
    import os
    proxy_url = os.getenv("PROXY_URL")
    
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
        proxy=proxy_url if proxy_url else None
    )
    
    pages_to_crawl = [base_url]
    crawled_urls = set()
    all_content = []
    
    # Use semaphore to limit concurrency and avoid rate limits
    semaphore = asyncio.Semaphore(3)

    async def crawl_with_semaphore(url, crawler):
        async with semaphore:
            try:
                content, internal_links = await crawl_site(url, crawler=crawler)
                return url, content, internal_links
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                return url, None, []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while pages_to_crawl and len(crawled_urls) < max_pages:
                # Prepare batch of URLs
                batch_size = min(len(pages_to_crawl), 3)
                current_batch = []
                for _ in range(batch_size):
                    u = pages_to_crawl.pop(0)
                    if u not in crawled_urls and not any(kw in u.lower() for kw in EXCLUDED_KEYWORDS):
                        current_batch.append(u)
                
                if not current_batch:
                    continue

                tasks = [crawl_with_semaphore(u, crawler) for u in current_batch]
                results = await asyncio.gather(*tasks)
                
                for url, content, internal_links in results:
                    if content:
                        all_content.append({"url": url, "content": content})
                        crawled_urls.add(url)
                        
                        # Add new internal links to queue
                        for link in internal_links:
                            link_url = link.get("href")
                            if link_url:
                                full_url = urljoin(url, link_url)
                                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                                    if not any(kw in full_url.lower() for kw in EXCLUDED_KEYWORDS):
                                        if full_url not in crawled_urls and full_url not in pages_to_crawl:
                                            pages_to_crawl.append(full_url)
                
                # Small yield to allow event loop to breathe
                await asyncio.sleep(0.1)
                
    except Exception as e:
        if "Browser.close" in str(e) or "closed" in str(e).lower():
            print(f"DEBUG: Browser closed unexpectedly, continuing...")
        else:
            print(f"ERROR: Fatal error in recursive crawl: {e}")

    return all_content

if __name__ == "__main__":
    # Test crawling
    test_url = "https://vite.dev/guide/"
    content = asyncio.run(crawl_site(test_url))
    if content:
        print("\nExtracted Content (First 500 chars):")
        print(content[:500])
