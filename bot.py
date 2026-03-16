from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from query import get_cached_retriever
from langchain.prompts import PromptTemplate
from functools import lru_cache
import os
from dotenv import load_dotenv

from ddg_image_search import search_images
import json
import re
from urllib.parse import unquote, urljoin


def _normalize_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return "https://" + url.lstrip("/")
    return url


def _extract_product_url_from_text(text: str, base_url: str | None = None) -> str | None:
    """Try to find a more specific product/detail URL in page text."""
    if not text or not isinstance(text, str):
        return None

    # Find explicit URLs first.
    urls = re.findall(r"https?://[^\s\"')]+", text)

    # Also look for common redirect query params (e.g. url=... or u=...)
    for m in re.findall(r"(?:url|u)=([^&\s]+)", text):
        try:
            decoded = unquote(m)
            if decoded.startswith("http"):
                urls.append(decoded)
            elif base_url:
                urls.append(urljoin(base_url, decoded))
        except Exception:
            pass

    # If any URL matches a product-style path (Amazon dp/gp, Walmart /ip/, etc), prefer it.
    product_patterns = [r"/dp/", r"/gp/product/", r"/product/", r"/item/", r"/ip/", r"/p/", r"/sku/"]
    for u in urls:
        for pat in product_patterns:
            if re.search(pat, u, re.IGNORECASE):
                return _normalize_url(u)

    # Fallback: return first URL found
    if urls:
        return _normalize_url(urls[0])

    return None


@lru_cache
def get_llm():
    load_dotenv()
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=4096,
        api_key=os.getenv("MOONSHOT_API_KEY")
    )

def chat_with_bot(query: str, discovered_stores: list = None, live_context: list = None, intent_type: str = "shopping", local_docs: list = None):
    """
    Sends a query to the chatbot and returns the response.
    """
    # Detect explicit image-search style queries so we can return image results directly
    is_image_query = bool(re.search(r"\b(image|images|photo|photos|picture|pictures|pic|pics)\b", query, re.IGNORECASE))

    def _is_placeholder_image(url: str | None) -> bool:
        if not url or not isinstance(url, str):
            return True
        u = url.lower()
        # Treat common placeholder and crawler/anchor images as invalid so we fall back to a real image search.
        return any(
            x in u
            for x in [
                "placehold.co",
                "via.placeholder",
                "example.com",
                "placeholder",
                "dummyimage",
                "noimage",
                "ai-gent-storage",
                "searchpage.jsp",
                "logo",
                "icon",
            ]
        )

    # Intent-based filtering (e.g. if user asks for phones, avoid showing earphones etc.)
    def _is_phone_query(q: str) -> bool:
        return bool(re.search(r"\b(phone|phones|smartphone|smartphones|cellphone|cellphones|mobile|mobiles)\b", q, re.IGNORECASE))

    def _is_phone_product(name: str) -> bool:
        return bool(
            re.search(
                r"\b(phone|smartphone|cellphone|iphone|galaxy|pixel|oneplus|moto|nokia|xiaomi|redmi|realme|oppo|vivo|huawei|sony|lg|s\d+|note\d+)\b",
                name,
                re.IGNORECASE,
            )
        )

    def _is_product_detail_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        return bool(re.search(r"/dp/|/gp/product/|/product/|/ip/|/p/|/sku/", url, re.IGNORECASE))

    def _is_search_url(url: str) -> bool:
        if not url or not isinstance(url, str):
            return False
        # Common patterns for search/listing pages
        return bool(re.search(r"/s\?|/search|/browse/|/gp/search|\?k=|\?q=", url, re.IGNORECASE))

    def _is_audio_product(name: str) -> bool:
        if not name or not isinstance(name, str):
            return False
        return bool(re.search(r"\b(earbud|earbuds|earphone|earphones|headphone|headphones|airpods|wireless bud|wireless ear)\b", name, re.IGNORECASE))

    if local_docs is None and not is_image_query:
        retriever = get_cached_retriever()
        docs = retriever._get_relevant_documents(query)
    else:
        # local_docs can be tuples of (Document, score) or plain Documents
        docs = [d[0] if isinstance(d, tuple) else d for d in (local_docs or [])]
    
    rag_items = []
    seen_products = set()  # Deduplication set
    carousel_products = []
    has_valid_images = False

    # If this is an image search query, bypass local docs and do an image search for thumbnails.
    if is_image_query:
        ddg_results = search_images(query, max_results=8)
        products = []
        for r in ddg_results:
            products.append({
                "name": r.get("title") or query,
                "brand": "",
                "price": "",
                "image_url": r.get("image_url"),
                "source_url": r.get("source_url") or "",
            })

        # Return a deterministic carousel response without calling the LLM for image-only queries.
        carousel_json = json.dumps(products, indent=2)
        return (
            f"Here are some images I found for '{query}':\n"
            f"<product_carousel>\n{carousel_json}\n</product_carousel>"
        )

    for doc in docs:
        item = f"CONTENT: {doc.page_content}"

        # Standardize to IMAGE_URL and SOURCE_URL for LLM consistency
        img = doc.metadata.get("image_url") or doc.metadata.get("s3_image_url") or doc.metadata.get("Image URL")

        # Prefer a more specific product URL found inside the page content (e.g., /dp/ links on Amazon)
        raw_src = doc.metadata.get("source") or doc.metadata.get("source_url") or doc.metadata.get("Source URL") or ""
        src = _normalize_url(raw_src)
        better_src = _extract_product_url_from_text(doc.page_content, base_url=src)
        if better_src:
            src = better_src

        # Normalize image URLs (ensure https and expand relative paths)
        if img and isinstance(img, str):
            if _is_placeholder_image(img):
                img = None
            else:
                if img.startswith("//"):
                    img = "https:" + img
                if img.startswith("/"):
                    img = urljoin(src or "", img)

        # Fallback: Extraction from content if metadata is missing
        if not img:
            markdown_imgs = re.findall(r'!\[.*?\]\((.*?)\)', doc.page_content)
            html_imgs = re.findall(r"<img.*?src=['\"](.*?)['\"]", doc.page_content, flags=re.IGNORECASE)
            # Try lazy-loaded image attributes too
            html_imgs += re.findall(r"<img.*?data-src=['\"](.*?)['\"]", doc.page_content, flags=re.IGNORECASE)
            all_content_imgs = markdown_imgs + html_imgs
            if all_content_imgs:
                # Basic filter to avoid tiny icons/logos if possible
                for u in all_content_imgs:
                    if u and "logo" not in u.lower() and "icon" not in u.lower():
                        img = u
                        break
                if not img:
                    img = all_content_imgs[0]

            if img and isinstance(img, str):
                if img.startswith("//"):
                    img = "https:" + img
                if img.startswith("/"):
                    img = urljoin(src or "", img)

        name = doc.metadata.get("name") or doc.metadata.get("Product Name") or ""
        brand = doc.metadata.get("brand") or doc.metadata.get("Brand") or ""
        price = doc.metadata.get("price") or doc.metadata.get("Price") or ""

        # If we still don't have a useful image, fallback to a quick DuckDuckGo image lookup for the product name
        if (not img or _is_placeholder_image(img)) and name:
            ddg_results = search_images(name, max_results=1)
            if ddg_results:
                candidate = ddg_results[0].get("image_url")
                if candidate and not _is_placeholder_image(candidate):
                    img = candidate

        # Deduplication based on Source URL and Name
        product_key = (src, name)
        if product_key in seen_products:
            continue
        seen_products.add(product_key)

        item = f"CONTENT: {doc.page_content}"
        if img:
            item += f"\nIMAGE_URL: {img}"
            if not _is_placeholder_image(img):
                has_valid_images = True
        if src:
            item += f"\nSOURCE_URL: {src}"
        rag_items.append(item)

        # If the user is asking about phones, only keep products that look like phones and that link to a product detail page
        if _is_phone_query(query):
            if not name or not _is_phone_product(name):
                continue
            if _is_audio_product(name):
                continue
            if src and _is_search_url(src):
                continue

        carousel_products.append({
            "name": name or query,
            "brand": brand,
            "price": price,
            "image_url": img or "",
            "source_url": src or "",
        })

    # If no valid images were found in the retrieved docs, fall back to an image search (DuckDuckGo)
    if not has_valid_images:
        ddg_results = search_images(query, max_results=6)
        if ddg_results:
            for r in ddg_results:
                title = r.get("title") or query
                img = r.get("image_url")
                src = r.get("source_url")
                if img and not _is_placeholder_image(img):
                    has_valid_images = True
                    # Add these images to the carousel products too, so we always have something to show.
                    carousel_products.append({
                        "name": title,
                        "brand": "",
                        "price": "",
                        "image_url": img,
                        "source_url": src or "",
                    })
                rag_items.append(f"CONTENT: Image search result for '{title}'\nIMAGE_URL: {img}\nSOURCE_URL: {src}")

    rag_context = "\n\n---\n\n".join(rag_items)
    
    # Format LIVE products into context
    live_context_str = ""
    if live_context:
        live_items = []
        for p in live_context:
            live_items.append(
                f"LIVE_PRODUCT_S3: {p.get('name')}\n"
                f"Brand: {p.get('brand')}\n"
                f"Price: {p.get('price')} {p.get('currency')}\n"
                f"IMAGE_URL: {p.get('image_url')}\n"
                f"SOURCE_URL: {p.get('source_url')}"
            )
        live_context_str = "\n\n".join(live_items)
    context = f"{live_context_str}\n\n{rag_context}".strip()

    # If we have no retrieved context, the model should still answer using general knowledge.
    has_context = bool(context)
    if not has_context:
        context = (
            "NOTE: There is no retrieved document content available for this query. "
            "Answer based on general knowledge and suggest ingesting a relevant website if it would help."
        )

    # If this is an explicit image search, force shopping/carousel behavior so we return images.
    if is_image_query:
        intent_type = "shopping"

    # If the retrieved context has no product images, don't force shopping/carousel output.
    has_product_images = any("IMAGE_URL:" in item for item in rag_items)
    use_shopping_prompt = (intent_type == "shopping") and has_context and has_product_images

    if use_shopping_prompt:
        template = """You are a Hybrid AI Shopping Assistant.
        
        CONTEXT:
        {context}
        INSTRUCTIONS:
        1. **Direct Answer**: Summarize the best options found.
        2. **Carousel**: You MUST provide a product carousel for the items found.
        3. **Carousel Format**: 
           <product_carousel>
           [ {{ "name": "...", "brand": "...", "price": "...", "image_url": "...", "source_url": "..." }} ]
           </product_carousel>
        
        CRITICAL PRODUCT RULES:
        1. **IMAGE ACCURACY**: ONLY use the URL labeled `IMAGE_URL` for the `image_url` field. NEVER use `SOURCE_URL` as an image.
        2. **IMAGE NECESSITY**: ONLY include products in the carousel if they have a valid, non-empty `IMAGE_URL` that starts with http or https.
        3. **STRICT SOURCE**: If the user is asking about a specific store (e.g., Baby Brands Direct), ONLY show products from that store in the carousel. 
        4. **ZERO HALLUCINATION**: NEVER guess or invent image URLs. If a product in the CONTEXT lacks a clear `IMAGE_URL`, do NOT put it in the carousel.
        5. **NO RAW JSON**: NEVER output raw JSON blocks or internal tags like `<product_carousel>` inside your natural language explanation text. They must be separate.
        Question: {question}
        Answer:"""
    else:
        template = """You are an Intelligent Informational Assistant.
        
        CONTEXT:
        {context}
        INSTRUCTIONS:
        1. **Deep Explanation**: Provide a comprehensive and clear explanation of the topic.
        2. **Real-World Examples**: Include at least one practical example or scenario to illustrate the concept.
        3. **No Carousels**: Do NOT use the <product_carousel> tag or show products as cards. Use text only.
        4. **Structured Layout**: Use headers and bullet points for readability.
        
        Question: {question}
        Answer:"""
    
    prompt = template.format(context=context, question=query)
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content
    
    # URL prefixing logic - ensuring all domains have https:// protocol
    base_retail_domains = [
        "amazon.in", "flipkart.com", "ajio.com", "myntra.com", 
        "firstcry.com", "nykaafashion.com", "m.media-amazon.com", 
        "static.ajio.com", "ai-gent-storage.s3.ap-south-1.amazonaws.com",
        "brightminds.co.uk", "mxwholesale.co.uk", "babybrandsdirect.co.uk", "puckator-dropship.co.uk",
        "apple.com", "store.apple.com", "amazon.com", "walmart.com", "bestbuy.com", "ebay.com", "target.com"
    ]
    for domain in base_retail_domains:
        # Match "domain or //domain and replace with https://domain
        # Only replace if it's NOT already prefixed by https:// or http://
        content = content.replace(f'"{domain}', f'"https://{domain}')
        content = content.replace(f'"//{domain}', f'"https://{domain}')
        
    # Final cleanup for protocol errors
    content = content.replace("https://https://", "https://")
    content = content.replace("https://http://", "https://")
    content = content.replace("https:////", "https://")
    
    # Remove any carousel the LLM emitted; we will create our own deterministic carousel.
    if "<product_carousel>" in content:
        content = re.sub(r"<\s*product_carousel\s*>[\s\S]*?<\s*/\s*product_carousel\s*>", "", content, flags=re.IGNORECASE).strip()

    # Ensure we include a deterministic carousel if we have good product items.
    if use_shopping_prompt and carousel_products:
        valid_carousel = [
            p for p in carousel_products
            if p.get("image_url") and p.get("source_url")
            and p.get("image_url").startswith("http")
            and not _is_placeholder_image(p.get("image_url"))
        ]
        if valid_carousel:
            content = content + "\n<product_carousel>\n" + json.dumps(valid_carousel, indent=2) + "\n</product_carousel>"

    return content
if __name__ == "__main__":
    # Test chat
    query = "What is this project about?"
    try:
        response = chat_with_bot(query)
        print(f"Bot: {response}")
    except Exception as e:
        print(f"Error communicating with bot: {e}")
        print("Make sure Ollama is running and 'llama3' model is pulled.")
