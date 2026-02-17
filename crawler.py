import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

from urllib.parse import urljoin, urlparse

EXCLUDED_KEYWORDS = ["login", "signup", "register", "cart", "checkout", "account", "profile", "wishlist", "help", "contact", "about", "privacy", "terms", "policy"]

async def crawl_site(url: str, crawler=None):
    """
    Crawls a given URL using configuration compatible with crawl4ai 0.8.0.
    Accepts an optional crawler instance for reuse.
    """
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        wait_for="body",
        page_timeout=30000  # 30 seconds
    )

    if crawler is None:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            return await _do_crawl(crawler, url, run_config)
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
    browser_config = BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
    )
    
    pages_to_crawl = [base_url]
    crawled_urls = set()
    all_content = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        while pages_to_crawl and len(crawled_urls) < max_pages:
            url = pages_to_crawl.pop(0)
            if url in crawled_urls:
                continue
            
            # Skip non-product pages
            if any(kw in url.lower() for kw in EXCLUDED_KEYWORDS):
                print(f"Skipping non-product URL: {url}")
                continue

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
            
            # Use a slightly longer sleep to avoid hitting rate limits on retail sites
            await asyncio.sleep(0.5)

    return all_content

if __name__ == "__main__":
    # Test crawling
    test_url = "https://vite.dev/guide/"
    content = asyncio.run(crawl_site(test_url))
    if content:
        print("\nExtracted Content (First 500 chars):")
        print(content[:500])
