import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from urllib.parse import urljoin, urlparse
import time


EXCLUDED_KEYWORDS = ["login", "signup", "register", "cart", "checkout", "account", "profile", "wishlist", "help", "contact", "about", "privacy", "terms", "policy", "travel", "flights", "hotels", "bus", "train", "tickets"]

def get_local_browser_config():
    """
    Returns a standard local BrowserConfig.
    """
    import os
    proxy_url = os.getenv("PROXY_URL")
    return BrowserConfig(
        headless=True,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
        proxy=proxy_url if proxy_url else None
    )

def get_browser_config(force_local=False):
    """
    Returns a BrowserConfig. If BROWSERLESS_URL is set in .env, it configures
    a remote browser connection to offload the heavy rendering work.
    """
    import os
    browserless_url = os.getenv("BROWSERLESS_URL")
    
    if not force_local and browserless_url and os.getenv("USE_REMOTE_BROWSER", "true").lower() == "true":
        print(f"DEBUG: Using remote browser at {browserless_url}", flush=True)
        return BrowserConfig(
            browser_type="chromium",
            browser_mode="custom",
            cdp_url=browserless_url,
            headless=True,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage"],
            use_managed_browser=False
        )
    
    return get_local_browser_config()

async def crawl_site(url: str, crawler=None):
    """
    Crawls a given URL using configuration compatible with crawl4ai 0.8.0.
    Accepts an optional crawler instance for reuse.
    Includes an automatic local fallback if remote browser fails.
    """
    import os
    browserless_url = os.getenv("BROWSERLESS_URL")
    
    # Optimized scroll to avoid blocking execution for ~10s while still triggering some lazy loaders.
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
        
        // Fast scroll down
        window.scrollTo(0, document.body.scrollHeight / 2);
        swapImages();
        await new Promise(resolve => setTimeout(resolve, 500));
        
        window.scrollTo(0, document.body.scrollHeight);
        await new Promise(resolve => setTimeout(resolve, 800));
        swapImages(); 
    })();
    """

    # Use PruningContentFilter to strip headers, footers, and nav
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.3, min_word_threshold=15)
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        word_count_threshold=10,
        wait_for="body",
        simulate_user=True,
        page_timeout=90000,
        wait_for_timeout=60000,
        js_code=js_scroll,
        markdown_generator=md_generator
    )

    if crawler is not None:
        return await _do_crawl(crawler, url, run_config)

    # First attempt: Try primary config (might be remote)
    browser_config = get_browser_config()
    is_remote = browser_config.browser_mode == "custom"
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            content, links = await _do_crawl(crawler, url, run_config)
            if content:
                return content, links
            # If No content, and it was remote, trigger fallback
            if is_remote:
                print(f"WARNING: Remote crawl returned no content for {url}. Attempting local fallback...", flush=True)
            else:
                return None, []
    except Exception as e:
        if not is_remote:
            print(f"ERROR: Local crawl failed for {url}: {e}", flush=True)
            return None, []
        print(f"WARNING: Remote crawl failed for {url}: {e}. Attempting local fallback...", flush=True)

    # Second attempt: Local Fallback (only if first was remote)
    if is_remote:
        local_config = get_browser_config(force_local=True)
        try:
            async with AsyncWebCrawler(config=local_config) as crawler:
                return await _do_crawl(crawler, url, run_config)
        except Exception as e:
            print(f"ERROR: Local fallback also failed for {url}: {e}", flush=True)
            return None, []
    
    return None, []

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
            print(f"Successfully crawled: {url}", flush=True)
            return content, result.links
        else:
            print(f"Failed to crawl: {url}. Error: {result.error_message}", flush=True)
            return None, []
    except Exception as e:
        print(f"Unexpected error crawling {url}: {e}", flush=True)
        return None, []

async def crawl_site_fast(url: str, crawler=None):
    """
    Extremely fast crawl for the initial synchronous UI phase.
    Bypasses JS scrolling and wait-for delays to return DOM instantly.
    """
    browser_config = get_browser_config()
    # Speed optimization: disable images
    browser_config.extra_args.append("--blink-settings=imagesEnabled=false")
    
    # Strip headers/footers for faster LLM parsing, but skip complex scrolling
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.3, min_word_threshold=15)
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,  # 30 seconds max
        wait_for_timeout=5000, # 5s max wait
        markdown_generator=md_generator
    )

    if crawler is None:
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                return await _do_crawl(crawler, url, run_config)
        except Exception as e:
            return None, []
    else:
        return await _do_crawl(crawler, url, run_config)

async def _run_recursive_crawl(base_url: str, max_pages: int, browser_config) -> list:
    """
    Internal helper that runs the recursive crawl with a given browser config.
    """
    pages_to_crawl = [base_url]
    crawled_urls = set()
    all_content = []

    # 5 concurrent fetches — safe on m6a.xlarge (16GB RAM)
    semaphore = asyncio.Semaphore(5)

    async def crawl_with_semaphore(url, crawler):
        async with semaphore:
            try:
                print(f"Starting crawl of: {url}...", flush=True)
                content, internal_links = await crawl_site(url, crawler=crawler)
                return url, content, internal_links
            except Exception as e:
                print(f"Error crawling {url}: {e}", flush=True)
                return url, None, []

    MAX_CRAWL_TIME = 600
    start_time = time.time()

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            while pages_to_crawl and len(crawled_urls) < max_pages:
                if time.time() - start_time > MAX_CRAWL_TIME:
                    print(f"DEBUG: Recursive crawl timeout reached ({MAX_CRAWL_TIME}s). Returning partial results.", flush=True)
                    break

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

                        links_list = internal_links.get("internal", []) if isinstance(internal_links, dict) else internal_links
                        for link in links_list:
                            if isinstance(link, dict):
                                link_url = link.get("href")
                            else:
                                continue

                            if link_url:
                                full_url = urljoin(url, link_url)
                                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                                    if not any(kw in full_url.lower() for kw in EXCLUDED_KEYWORDS):
                                        if full_url not in crawled_urls and full_url not in pages_to_crawl:
                                            pages_to_crawl.append(full_url)

                await asyncio.sleep(0.1)

    except Exception as e:
        if "Browser.close" in str(e) or "closed" in str(e).lower():
            print(f"DEBUG: Browser closed unexpectedly, continuing...", flush=True)
        else:
            print(f"ERROR: Fatal error in recursive crawl: {e}", flush=True)

    return all_content


async def crawl_site_recursive(base_url: str, max_pages: int = 100):
    """
    Concurrent recursive crawl starting from base_url up to max_pages.
    Tries remote browser first (if BROWSERLESS_URL is set), then falls back
    to local Chromium automatically if remote returns no content.
    """
    remote_config = get_browser_config()
    is_remote = getattr(remote_config, 'browser_mode', None) == "custom"

    # First attempt: remote or local depending on config
    print(f"DEBUG: Starting recursive crawl of {base_url} (remote={is_remote})...", flush=True)
    results = await _run_recursive_crawl(base_url, max_pages, remote_config)

    if results:
        return results

    # Fallback: if remote was used and returned nothing, retry with local browser
    if is_remote:
        print(f"WARNING: Remote recursive crawl returned no results for {base_url}. Attempting local fallback...", flush=True)
        local_config = get_browser_config(force_local=True)
        results = await _run_recursive_crawl(base_url, max_pages, local_config)

    return results

if __name__ == "__main__":
    # Test crawling
    test_url = "https://vite.dev/guide/"
    content = asyncio.run(crawl_site(test_url))
    if content:
        print("\nExtracted Content (First 500 chars):")
        print(content[:500])
