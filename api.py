from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import re
import json
import random
from crawler import crawl_site, crawl_site_recursive
from ingest import add_content_to_store, add_multiple_contents_to_store
from vector_store import clear_vector_store
from query import fast_query
from bot import chat_with_bot
from retail_crawler import retail_crawler
from kimi_service import kimi_service
from urllib.parse import urlparse

# Track last crawled domain
last_crawled_domain = None


def update_last_domain(url):
    global last_crawled_domain
    try:
        domain = urlparse(url).netloc
        if domain:
            last_crawled_domain = domain
            print(f"DEBUG: Updated last_crawled_domain to: {last_crawled_domain}")
    except Exception:
        pass


# ✅ FORMAT RESPONSE FOR FRONTEND (VERY IMPORTANT)
def format_response(res):
    if isinstance(res, str):
        return res

    # 🚗 Vehicle formatting
    if isinstance(res, dict) and not res.get("type") and "price" in res:
        return f"""
🚗 {res.get('name', 'Vehicle')}
💰 Price: {res.get('price', 'N/A')}
⛽ Mileage: {res.get('mileage', 'N/A')}
🔋 Fuel: {res.get('fuel', 'N/A')}
"""

    # 🖼️ Images (Horizontal Carousel)
    if isinstance(res, dict) and res.get("type") == "images":
        results = res.get("results", [])
        carousel_json = json.dumps(results)
        # Use newlines to ensure markdown renderer treats this as a block
        return f"\n\n<product_carousel>\n{carousel_json}\n</product_carousel>\n\n"

    # 🛒 Products
    if isinstance(res, list):
        if not res: return "No products found."
        return "\n\n".join([
            f"🛍️ {p.get('name', 'Product')} - {p.get('price', '')}\n🔗 {p.get('url', p.get('source_url', ''))}"
            for p in res if isinstance(p, dict)
        ])

    return str(res)


# ✅ UTILITY: ENSURE CAROUSEL JSON IS ROBUST
# We now reconstruct the carousel from a lookup map to prevent bot hallucinations.
def rebuild_carousel_with_map(content, lookup_map):
    if not isinstance(content, str): return content
    
    patterns = [
        r'(<product_carousel>)(.*?)(</product_carousel>)',
        r'(<product_carousel>\s*)(.*?)(\s*</product_carousel>)'
    ]
    
    def reconstruct(match):
        tags_open = match.group(1)
        names_str = match.group(2).strip()
        tags_close = match.group(3)
        
        try:
            # The bot now outputs a list of names: ["Nike Air", "Nike Air Max"]
            names = json.loads(names_str)
            if not isinstance(names, list): names = [names]
            
            rebuilt_data = []
            for name in names:
                name_clean = str(name).strip().lower()
                # Find the best match in our lookup map
                match_data = None
                # Exact match first
                if name_clean in lookup_map:
                    match_data = lookup_map[name_clean]
                else:
                    # Fuzzy match: Is the bot's name contained in any of our real names?
                    for real_name, data in lookup_map.items():
                        if name_clean in real_name or real_name in name_clean:
                            match_data = data
                            break
                
                if match_data:
                    rebuilt_data.append(match_data)
                    
            if not rebuilt_data and names:
                # If fuzzy matching completely failed (bot hallucinated names from URL slugs),
                # but we DO have live products available in the lookup map, forcefully inject them!
                if lookup_map:
                    print(f"DEBUG: Carousel fuzzy match failed for names: {names}. Forcefully injecting top 5 available products.")
                    rebuilt_data = list(lookup_map.values())[:5]
                else:
                    # If we truly have absolutely no data, remove the tag completely to prevent UI gray box
                    return ""
                
            compact = json.dumps(rebuilt_data, separators=(',', ':'), ensure_ascii=False)
            return f"\n\n{tags_open}\n{compact}\n{tags_close}\n\n"
            
        except Exception as e:
            print(f"DEBUG Reconstruct fail: {e}")
            return match.group(0) # Return original if parsing fails

    result = content
    for p in patterns:
        result = re.sub(p, reconstruct, result, flags=re.DOTALL)
    
    return result


async def background_ingest(url: str, max_pages: int = 1):
    try:
        print(f"Background ingestion started for: {url} (max_pages={max_pages})")
        if max_pages <= 1:
            content, _ = await crawl_site(url)
            if content and len(content.strip()) > 10:
                await add_content_to_store(content, {"source": url})
                update_last_domain(url)
        else:
            results = await crawl_site_recursive(url, max_pages=max_pages)
            if results:
                await add_multiple_contents_to_store(results)
                print(f"Background deep ingestion complete for {url}")
    except Exception as e:
        print(f"Error in background ingestion for {url}: {e}")


app = FastAPI(title="Retail AI RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrawlRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    message: str


@app.post("/crawl")
async def crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    background_tasks.add_task(background_ingest, request.url, max_pages=1)
    return {"status": "success", "message": f"Ingestion started for {request.url}"}


@app.post("/crawl/deep")
async def deep_crawl_endpoint(request: CrawlRequest, background_tasks: BackgroundTasks):
    if not request.url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL protocol")
    background_tasks.add_task(background_ingest, request.url, max_pages=15)
    return {"status": "success", "message": "Deep ingestion started"}


@app.post("/clear")
async def clear_endpoint():
    clear_vector_store()
    return {"status": "success", "message": "Memory cleared successfully"}


# -------------------------------
# BACKGROUND TASK HELPER
# -------------------------------
async def background_crawl_and_ingest(query: str, fast_products: list):
    try:
        print(f"🔄 BACKGROUND: Starting deep crawl for '{query}'...")
        deep_results = await kimi_service.run_deep_crawl_process(query, fast_products)
        if deep_results:
            print(f"🔄 BACKGROUND: Deep crawl yielded {len(deep_results)} rich products. Saving to DB...")
            await kimi_service.cache_and_store_products(deep_results, query)
        print(f"✅ BACKGROUND: Completely finished processing '{query}'!")
    except Exception as e:
        print(f"❌ BACKGROUND: Failed deep crawl for '{query}': {e}")
        import traceback
        traceback.print_exc()

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    try:
        query = request.message.strip()
        query_lower = query.lower()
        print(f"\n🔥 Query: {query}")

        # -------------------------------
        # 🧠 INTENT DETECTION
        # -------------------------------
        intent = kimi_service.detect_intent(query_lower)
        print(f"🧠 Intent: {intent} (Took {time.time() - start_time:.2f}s)")

        # -------------------------------
        # 🖼️ IMAGE SEARCH OVERRIDE (PRIORITY)
        # -------------------------------
        # If user explicitly asks for images/photos, use the optimized Bing searcher
        if any(x in query_lower for x in ["image", "photo", "pic", "picture", "show me", "images"]):
            print(f"🖼️ Image Search Override for: {query}")
            img_results = await kimi_service.search_images(query)
            if img_results and img_results.get("results"):
                return {"status": "success", "response": format_response(img_results)}

        # -------------------------------
        # 🚗 VEHICLE FLOW
        # -------------------------------
        if intent == "vehicle":
            print("🚗 Vehicle flow")
            result = await kimi_service.get_vehicle_data(query)
            return {"status": "success", "response": format_response(result)}

        # -------------------------------
        # 🛒 SHOPPING FLOW
        # -------------------------------
        live_products = []
        local_results = []
        if intent == "shopping":
            rag_start = time.time()
            # ONLY search in 'retail' category to avoid pulling generic docs/tutorials
            local_results = fast_query(query, category="retail", threshold=2.2)
            print(f"🛒 RAG Check: Found {len(local_results)} docs (Took {time.time() - rag_start:.2f}s)")
            # Shuffle for variety: every search shows a different mix from the full cached pool
            random.shuffle(local_results)

            # Check if we have at least 2 items with images. 
            # If not, we definitely need live data.
            results_with_images = [r for r in local_results if r[0].metadata.get("image_url") or r[0].metadata.get("s3_image_url")]
            
            # PROACTIVE: Even if we have some RAG hits, if it's a "fresh" shopping query (few visual hits)
            # we fetch fast basic data to show instantly, and do the heavy crawl in the background!
            if len(results_with_images) < 4:
                print(f"🛒 Limited visual local results ({len(results_with_images)}). Fetching fast Bing data...")
                kimi_start = time.time()
                live_products = await kimi_service.get_fast_bing_data(query) # INSTANT RESPONSE!
                print(f"⚡ Fast Search: Found {len(live_products)} basic products (Took {time.time() - kimi_start:.2f}s)")

                if live_products:
                    # Fire-and-forget the heavy crawling and S3 uploading!
                    # It will run transparently after we send the fast UI response.
                    background_tasks.add_task(background_crawl_and_ingest, query, live_products)
            else:
                print(f"🛒 Strong RAG Presence ({len(results_with_images)} visual docs). Using local store.")

        # -------------------------------
        # 🌐 FALLBACK / BOT RESPONSE
        # -------------------------------
        # 🤖 BOT RESPONSE GENERATION
        # -------------------------------
        # Build a lookup map for the re-constructor
        lookup_map = {}
        # From Live Products
        for p in live_products:
            name = str(p.get("name") or "Product").strip().lower()
            if name not in lookup_map:
                lookup_map[name] = {
                    "name": p.get("name"),
                    "price": p.get("price") or "Check Site",
                    "image_url": p.get("image_url"),
                    "source_url": p.get("url") or p.get("source_url")
                }
        # From RAG Results
        for doc, score in local_results:
            name = str(doc.metadata.get("name") or doc.metadata.get("Product Name") or f"Option {len(lookup_map)+1}").strip().lower()
            if name not in lookup_map:
                lookup_map[name] = {
                    "name": doc.metadata.get("name") or "Product",
                    "price": doc.metadata.get("price") or "Market Price",
                    "image_url": doc.metadata.get("image_url") or doc.metadata.get("s3_image_url"),
                    "source_url": doc.metadata.get("source") or doc.metadata.get("source_url")
                }

        bot_start = time.time()
        print(f"🤖 Bot is generating response for: {query}")
        bot_response = await chat_with_bot(
            query=query, 
            live_context=live_products,
            intent_type=intent,
            local_docs=local_results
        )
        
        # ✅ UTILITY: SHARING LOGS FOR DEBUGGING
        print(f"✅ Bot Done. Response length: {len(bot_response)}")
        
        # 4. Final Formatting & Re-construction
        final_response = format_response(bot_response)
        # Rebuild the carousel from our verified map to prevent hallucinations!
        final_response = rebuild_carousel_with_map(final_response, lookup_map)

        # 5. If the carousel still has < 5 products, forcefully pad from the lookup_map
        #    This happens when strong RAG data exists but the bot only mentioned 3 names.
        if intent == "shopping" and lookup_map:
            carousel_match = re.search(r'<product_carousel>\s*(\[.*?\])\s*</product_carousel>', final_response, re.DOTALL)
            if carousel_match:
                try:
                    current_items = json.loads(carousel_match.group(1))
                    if len(current_items) < 5:
                        print(f"DEBUG: Carousel has only {len(current_items)} items. Padding from lookup_map...")
                        # Collect current source_urls to de-duplicate
                        existing_sources = {p.get('source_url') for p in current_items}
                        # Shuffle lookup values so we pick different items each time
                        all_candidates = list(lookup_map.values())
                        random.shuffle(all_candidates)
                        # Pick diverse items from the lookup_map, skipping already shown ones
                        for p in all_candidates:
                            if len(current_items) >= 10:
                                break
                            if p.get('source_url') not in existing_sources and p.get('image_url'):
                                current_items.append(p)
                                existing_sources.add(p.get('source_url'))
                        # Replace the carousel in the final response
                        rebuilt = json.dumps(current_items, separators=(',', ':'), ensure_ascii=False)
                        final_response = final_response[:carousel_match.start(1)] + rebuilt + final_response[carousel_match.end(1):]
                        print(f"DEBUG: Padded carousel to {len(current_items)} items.")
                except Exception as pad_err:
                    print(f"DEBUG: Carousel padding failed: {pad_err}")

        print(f"✅ Bot Done (Took {time.time() - bot_start:.2f}s)")
        print(f"🚀 Total Response Time: {time.time() - start_time:.2f}s")

        return {"status": "success", "response": final_response}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
