import json
from bs4 import BeautifulSoup
import re

class ProductExtractor:
    @staticmethod
    def extract_from_html(html: str, url: str) -> dict:
        """
        Extracts product metadata from raw HTML.
        Targets JSON-LD, Meta tags, and Title.
        """
        if not html:
            return {}

        soup = BeautifulSoup(html, 'html.parser')
        symbols = {"$": "$", "USD": "$", "RS": "₹", "INR": "₹", "GBP": "£", "EUR": "€"}
        
        data = {
            "name": None,
            "price": None,
            "currency": None,
            "price_numeric": None,
            "rating_avg": None,
            "rating_count": None,
            "availability": "In Stock", # Default
            "brand": None,
            "description": None,
            "reviews": []
        }

        # 1. Title/Name
        title_tag = soup.find("title")
        if title_tag:
            data["name"] = title_tag.get_text().strip()

        # 2. Structured Data (JSON-LD)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                content = script.string
                if not content: continue
                ld = json.loads(content)
                
                items = ld if isinstance(ld, list) else [ld]
                if isinstance(ld, dict) and "@graph" in ld: items = ld["@graph"]
                
                for item in items:
                    if not isinstance(item, dict): continue
                    
                    # AggregateRating
                    rate = item.get("aggregateRating")
                    if isinstance(rate, dict):
                        data["rating_avg"] = rate.get("ratingValue") or rate.get("value")
                        data["rating_count"] = rate.get("reviewCount") or rate.get("ratingCount")
                    
                    # Brand
                    brand = item.get("brand")
                    if isinstance(brand, dict):
                        data["brand"] = brand.get("name")
                    elif isinstance(brand, str):
                        data["brand"] = brand

                    # Name (Often better than Title)
                    if item.get("@type") == "Product" and item.get("name"):
                        data["name"] = item.get("name")
                        
                    # Description
                    if item.get("description"):
                        doc_desc = item.get("description")
                        if isinstance(doc_desc, str) and (not data["description"] or len(doc_desc) > len(data["description"])):
                            data["description"] = doc_desc
                            
                    # Reviews
                    reviews_data = item.get("review") or item.get("reviews")
                    if reviews_data:
                        if not isinstance(reviews_data, list): reviews_data = [reviews_data]
                        for r in reviews_data:
                            if isinstance(r, dict):
                                author = r.get("author", {}).get("name", "User") if isinstance(r.get("author"), dict) else "User"
                                body = r.get("reviewBody") or r.get("text")
                                rev_rating = r.get("reviewRating", {}).get("ratingValue") if isinstance(r.get("reviewRating"), dict) else None
                                if body:
                                    data["reviews"].append({"user": author, "comment": body[:150], "rating": rev_rating})

                    # Offers
                    offers = item.get("offers")
                    if offers:
                        if isinstance(offers, list): offers = offers[0]
                        if isinstance(offers, dict):
                            price = offers.get("price") or offers.get("lowPrice")
                            curr = offers.get("priceCurrency")
                            if price:
                                data["price_numeric"] = float(str(price).replace(",", ""))
                                data["currency"] = curr
                                data["price"] = f"{symbols.get(curr, '$')}{price}" if curr else str(price)
                            
                            # Availability
                            avail = offers.get("availability")
                            if avail:
                                if "OutOfStock" in str(avail):
                                    data["availability"] = "Out of Stock"
                                elif "InStock" in str(avail):
                                    data["availability"] = "In Stock"
            except: pass

        # 3. Meta Tags
        if not data["price"]:
            meta_price = [
                ("property", "product:price:amount"),
                ("property", "og:price:standard_amount"),
                ("name", "twitter:data1"),
                ("property", "price")
            ]
            
            # Find currency first
            currency_symbol = "$" 
            currency_meta = soup.find("meta", property="product:price:currency") or \
                            soup.find("meta", property="og:price:currency") or \
                            soup.find("meta", attrs={"name": "currency"})
            if currency_meta and currency_meta.get("content"):
                data["currency"] = currency_meta.get("content").upper()
                currency_symbol = symbols.get(data["currency"], symbols.get("USD"))

            for attr, val in meta_price:
                tag = soup.find("meta", {attr: val})
                if tag and tag.get("content"):
                    raw_p = tag.get("content")
                    try:
                        data["price_numeric"] = float(re.sub(r'[^\d.]', '', raw_p))
                    except: pass
                    data["price"] = f"{currency_symbol}{raw_p}"
                    break

        # 4. Meta Availability
        meta_avail = soup.find("meta", property="product:availability") or \
                     soup.find("meta", property="og:availability") or \
                     soup.find("meta", attrs={"name": "availability"})
        if meta_avail and meta_avail.get("content"):
            c = meta_avail.get("content").lower()
            if any(x in c for x in ["instock", "in stock", "available"]):
                data["availability"] = "In Stock"
            elif any(x in c for x in ["outofstock", "out of stock", "preorder"]):
                data["availability"] = "Out of Stock"

        # Regex Fallback for Price (If still missing)
        if not data["price"]:
            text = soup.get_text(separator=" ", strip=True)
            price_pattern = r'([£$€₹])\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
            matches = list(re.finditer(price_pattern, text))
            
            if matches:
                # Heuristic: First price near money-related words
                keywords = ["price", "now", "sale", "only", "offer"]
                sorted_matches = sorted(matches, key=lambda m: any(kw in text[max(0, m.start()-30):min(len(text), m.end()+30)].lower() for kw in keywords), reverse=True)
                m = sorted_matches[0]
                data["currency"] = next((k for k, v in symbols.items() if v == m.group(1)), "USD")
                data["price"] = f"{m.group(1)}{m.group(2)}"
                try: data["price_numeric"] = float(m.group(2).replace(",", ""))
                except: pass

        # 6. Source URL Enforcement
        data["source_url"] = url

        # 7. Safe Serialization for ChromaDB
        if data.get("reviews"):
            data["reviews"] = json.dumps(data["reviews"])
        else:
            data["reviews"] = None

        return {k: v for k, v in data.items() if v is not None}

product_extractor = ProductExtractor()
