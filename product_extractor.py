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
            "rating_avg": None,
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
                        if isinstance(doc_desc, str) and len(doc_desc) > len(data["description"] or ""):
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
                                data["price"] = f"{symbols.get(curr, '$')}{price}" if curr else str(price)
                                if data["price"] and not any(s in str(data["price"]) for s in symbols.values()):
                                    data["price"] = f"${data['price']}"
            except: pass

        # 3. Meta Tags
        if not data["price"]:
            meta_price = [
                ("property", "product:price:amount"),
                ("property", "og:price:standard_amount"),
                ("name", "twitter:data1"),
                ("property", "price")
            ]
            
            # Find currency first to format price correctly
            currency_symbol = "$" # Default
            currency_meta = soup.find("meta", property="product:price:currency") or \
                            soup.find("meta", property="og:price:currency") or \
                            soup.find("meta", attrs={"name": "currency"})
            if currency_meta and currency_meta.get("content"):
                curr = currency_meta.get("content").upper()
                currency_symbol = symbols.get(curr, symbols.get("USD"))

            for attr, val in meta_price:
                tag = soup.find("meta", {attr: val})
                if tag and tag.get("content"):
                    data["price"] = tag.get("content")
                    if data["price"] and not any(s in str(data["price"]) for s in symbols.values()):
                        data["price"] = f"{currency_symbol}{data['price']}"
                    break

        if not data["rating_avg"]:
            meta_r = soup.find("meta", property="og:rating") or soup.find("meta", attrs={"name": "rating"})
            if meta_r: data["rating_avg"] = meta_r.get("content")

        if not data["brand"]:
            meta_b = soup.find("meta", property="product:brand") or soup.find("meta", attrs={"name": "brand"})
            if meta_b: data["brand"] = meta_b.get("content")

        if not data["name"]:
            meta_n = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "title"})
            if meta_n: data["name"] = meta_n.get("content")
            
        if not data["description"]:
            meta_d = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
            if meta_d and meta_d.get("content"): data["description"] = meta_d.get("content")[:500]

        # 4. Regex Fallback for Price (If still missing)
        if not data["price"]:
            text = soup.get_text(separator=" ", strip=True)
            # Match currency symbols followed by numbers: £499.00, $50, ₹1,200.50
            price_pattern = r'([£$€₹])\s?(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
            matches = re.finditer(price_pattern, text)
            
            # Heuristic: Find prices near keywords or just take the first prominent one
            keywords = ["price", "now", "sale", "only", "offer"]
            best_match = None
            
            for match in matches:
                symbol = match.group(1)
                amount = match.group(2)
                full_match = f"{symbol}{amount}"
                
                # If we find a price near a keyword, it's likely the right one
                start, end = match.span()
                context = text[max(0, start-30):min(len(text), end+30)].lower()
                if any(kw in context for kw in keywords):
                    best_match = full_match
                    break
                if not best_match:
                    best_match = full_match
            
            if best_match:
                data["price"] = best_match

        # 5. Rating/Review Aggressive Hunt
        if not data.get("rating_avg"):
            # Look for "X out of 5 stars" or "Rating: X"
            text = soup.get_text(separator=" ", strip=True)
            r_match = re.search(r'(\d+\.?\d*)\s*out of 5', text, re.IGNORECASE) or \
                      re.search(r'Rating:\s*(\d+\.?\d*)', text, re.IGNORECASE)
            if r_match:
                data["rating_avg"] = r_match.group(1)

        # 6. Source URL Enforcement
        data["source_url"] = url

        # 7. Safe Serialization for ChromaDB (Requires flat structures)
        if data["reviews"]:
            data["reviews"] = json.dumps(data["reviews"])
        else:
            data["reviews"] = None

        return {k: v for k, v in data.items() if v is not None}

product_extractor = ProductExtractor()
