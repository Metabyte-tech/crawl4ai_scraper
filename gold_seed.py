import asyncio
import os
import json
from ingest import add_multiple_contents_to_store

# GOLD STANDARD PRODUCTS FOR EACH CATEGORY
GOLD_PRODUCTS = {
    "Baby & Kids": [
        {
            "name": "Soft Plush Elephant Toy",
            "price": "$24.99",
            "image": "https://m.media-amazon.com/images/I/71Y8X-oX9xL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N18B735",
            "brand": "BabyCare"
        },
        {
            "name": "Adjustable Baby High Chair",
            "price": "$89.00",
            "image": "https://m.media-amazon.com/images/I/61m1R8L8hSL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07GVP6G3C",
            "brand": "KidsGo"
        }
    ],
    "Electronics": [
        {
            "name": "Quantum Noise Cancelling Headphones",
            "price": "$299.00",
            "image": "https://m.media-amazon.com/images/I/61vJtS86S5L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08H75RTZ8",
            "brand": "AudioTech"
        },
        {
            "name": "Ultra-Wide Gaming Monitor 34\"",
            "price": "$450.00",
            "image": "https://m.media-amazon.com/images/I/71W77Tj5RSL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08LLD2B8S",
            "brand": "ViewSonic"
        }
    ],
    "Home & Kitchen": [
        {
            "name": "Professional Espresso Machine",
            "price": "$599.00",
            "image": "https://m.media-amazon.com/images/I/71o0XvXf8vL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B00CH9QWOU",
            "brand": "Breville"
        },
        {
            "name": "Modern Velvet Sofa - Navy",
            "price": "$849.00",
            "image": "https://m.media-amazon.com/images/I/81U-4L9V3fL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N1BBNV3",
            "brand": "HomeLux"
        }
    ],
    "Fashion": [
        {
            "name": "Classic Leather Chelsea Boots",
            "price": "$120.00",
            "image": "https://m.media-amazon.com/images/I/71-02-BSRTL._AC_UL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07PDRJ9YV",
            "brand": "Stride"
        },
        {
            "name": "Italian Wool Slim Fit Suit",
            "price": "$450.00",
            "image": "https://m.media-amazon.com/images/I/61fR-L4L8tL._AC_UL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07GVP6G3D",
            "brand": "Milan"
        }
    ]
}

async def manual_seed():
    print("💎 Injecting Gold Standard products...")
    ingest_items = []
    
    for category, products in GOLD_PRODUCTS.items():
        for p in products:
            description = (
                f"Product: {p['name']}\n"
                f"Brand: {p['brand']}\n"
                f"Price: {p['price']}\n"
                f"Category: {category}\n"
                f"Image URL: {p['image']}\n"
                f"Source URL: {p['url']}"
            )
            
            metadata = {
                "source": p['url'],
                "type": "gold_seed",
                "category": category,
                "image_url": p['image'],
                "name": p['name'],
                "price": p['price'],
                "brand": p['brand'],
                "store_source": "Amazon Verified"
            }
            
            ingest_items.append({
                "content": description,
                "url": p['url'],
                "metadata": metadata
            })
            
    if ingest_items:
        await add_multiple_contents_to_store(ingest_items)
        print(f"✅ Successfully injected {len(ingest_items)} premier products.")

if __name__ == "__main__":
    asyncio.run(manual_seed())
