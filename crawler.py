import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

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
    
    # Slower, more thorough scroll script
    js_scroll = """
    (async () => {
        for (let i = 0; i < 5; i++) {
            window.scrollBy(0, window.innerHeight);
            await new Promise(resolve => setTimeout(resolve, 1500));
        }
    })();
    """

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        wait_for="body",
        page_timeout=90000,  # 90 seconds
        wait_for_timeout=60000, # 60s
        js_code=js_scroll
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
            content = result.markdown
            if len(content.strip()) < 100:
                print(f"Content short, retrying with explicit wait for {url}")
                run_config.wait_for = "js:() => document.body.innerText.length > 500"
                result = await crawler.arun(url=url, config=run_config)
                content = result.markdown
            print(f"Successfully crawled: {url}")
            return content, result.links.get("internal", [])
        else:
            print(f"Failed to crawl: {url}. Error: {result.error_message}")
            return None, []
    except Exception as e:
        print(f"Unexpected error crawling {url}: {e}")
        return None, []

async def crawl_site_recursive(base_url: str, max_pages: int = 20):
    """
    Recursive crawl starting from base_url up to max_pages.
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
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while pages_to_crawl and len(crawled_urls) < max_pages:
                url = pages_to_crawl.pop(0)
                if url in crawled_urls:
                    continue
                
                # Skip non-product pages
                if any(kw in url.lower() for kw in EXCLUDED_KEYWORDS):
                    print(f"Skipping non-product URL: {url}")
                    continue

                try:
                    content, internal_links = await crawl_site(url, crawler=crawler)
                    if content:
                        all_content.append({"url": url, "content": content})
                        crawled_urls.add(url)
                        
                        # Add new internal links to queue
                        for link in internal_links:
                            link_url = link.get("href")
                            if link_url:
                                # Normalize URL
                                full_url = urljoin(url, link_url)
                                # Ensure it's the same domain and NOT an excluded URL
                                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                                    if not any(kw in full_url.lower() for kw in EXCLUDED_KEYWORDS):
                                        if full_url not in crawled_urls and full_url not in pages_to_crawl:
                                            pages_to_crawl.append(full_url)
                except Exception as e:
                    print(f"Non-fatal error crawling {url} during recursive sync: {e}. Skipping page.")
                
                # Use a slightly longer sleep to avoid hitting rate limits on retail sites
                await asyncio.sleep(0.5)
    except Exception as e:
        if "Browser.close" in str(e) or "closed" in str(e).lower():
            print(f"DEBUG: Browser closed unexpectedly during cleanup, but we finished our crawl. Continuing...")
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
