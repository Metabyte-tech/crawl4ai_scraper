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
            "name": "Men's Lightweight Running Shoes",
            "brand": "Nivia",
            "price": "₹1299",
            "image_url": "https://m.media-amazon.com/images/I/71Yy8I5H+KL._AC_UY1100_.jpg",
            "source": "Amazon India",
            "details": "High performance running shoes",
            "category": "sports-outdoors"
        },
        {
            "name": "Nivia Aero Unisex Sports Cap",
            "brand": "Nivia",
            "price": "₹99",
            "image_url": "https://m.media-amazon.com/images/I/61s7O5x5N7L._AC_UL320_.jpg",
            "source_url": "https://www.amazon.in/dp/B00K5T2O9G",
            "source": "Amazon India",
            "details": "Breathable sports cap for running and training",
            "category": "sports-outdoors"
        },
        {
            "name": "Premium Cotton Sports T-Shirt",
            "brand": "Adidas",
            "price": "₹149",
            "image_url": "https://m.media-amazon.com/images/I/71z3A9kZ+1L._AC_UL320_.jpg",
            "source_url": "https://www.amazon.in/dp/B07N18B736",
            "source": "Walmart",
            "details": "Quick-dry fabric for intense workouts",
            "category": "sports-outdoors"
        },
        {
            "name": "Casual Canvas Sneakers - White",
            "price": "$45.00",
            "image": "https://m.media-amazon.com/images/I/61vJtS86S5L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08H75RTZ2",
            "brand": "Vans"
        },
        {
            "name": "Italian Wool Slim Fit Suit",
            "price": "$450.00",
            "image": "https://m.media-amazon.com/images/I/61fR-L4L8tL._AC_UL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07GVP6G3D",
            "brand": "Milan"
        }
    ],
    "Beauty & Health": [
        {
            "name": "Advanced Night Repair Serum",
            "price": "$75.00",
            "image": "https://m.media-amazon.com/images/I/61O2hW6G6CL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08G8Y5G5G",
            "brand": "Estee"
        },
        {
            "name": "Sonic Electric Toothbrush",
            "price": "$129.99",
            "image": "https://m.media-amazon.com/images/I/71nI5B+lT1L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07H8W8H8H",
            "brand": "Philips"
        }
    ],
    "Sports & Outdoors": [
        {
            "name": "Lightweight Camping Tent 4-Person",
            "price": "$149.00",
            "image": "https://m.media-amazon.com/images/I/71X8X-oX9xL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N18B736",
            "brand": "Coleman"
        },
        {
            "name": "Premium Yoga Mat - Non-Slip",
            "price": "$65.00",
            "image": "https://m.media-amazon.com/images/I/81U-4L9V3fL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N1BBNV4",
            "brand": "Lulu"
        }
    ],
    "Grocery": [
        {
            "name": "Organic Extra Virgin Olive Oil",
            "price": "$28.00",
            "image": "https://m.media-amazon.com/images/I/71o0XvXf8vL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B00CH9QWOV",
            "brand": "Bertolli"
        },
        {
            "name": "Artisanal Whole Bean Coffee",
            "price": "$19.50",
            "image": "https://m.media-amazon.com/images/I/81U-4L9V3fL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N1BBNV5",
            "brand": "Stumptown"
        }
    ],
    "Industrial & Scientific": [
        {
            "name": "Digital Laser Distance Meter",
            "price": "$55.00",
            "image": "https://m.media-amazon.com/images/I/61vJtS86S5L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08H75RTZ9",
            "brand": "Bosch"
        },
        {
            "name": "Professional Lab Microscope",
            "price": "$380.00",
            "image": "https://m.media-amazon.com/images/I/71W77Tj5RSL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08LLD2B8T",
            "brand": "AmScope"
        }
    ],
    "Pet Supplies": [
        {
            "name": "Automatic Pet Feeder - WiFi",
            "price": "$89.99",
            "image": "https://m.media-amazon.com/images/I/61m1R8L8hSL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07GVP6G3E",
            "brand": "PetLibro"
        },
        {
            "name": "Orthopedic Dog Bed - Large",
            "price": "$120.00",
            "image": "https://m.media-amazon.com/images/I/81U-4L9V3fL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N1BBNV6",
            "brand": "BarkBox"
        }
    ],
    "Automotive": [
        {
            "name": "Portable Car Jump Starter",
            "price": "$99.00",
            "image": "https://m.media-amazon.com/images/I/71Y8X-oX9xL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N18B737",
            "brand": "NOCO"
        },
        {
            "name": "OBD2 Scanner Bluetooth",
            "price": "$45.00",
            "image": "https://m.media-amazon.com/images/I/61vJtS86S5L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08H75RTZ0",
            "brand": "BlueDriver"
        }
    ],
    "Office Products": [
        {
            "name": "Ergonomic Office Chair - Mesh",
            "price": "$350.00",
            "image": "https://m.media-amazon.com/images/I/71o0XvXf8vL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B00CH9QWOW",
            "brand": "Herman"
        },
        {
            "name": "Dual Monitor Stand Mount",
            "price": "$75.00",
            "image": "https://m.media-amazon.com/images/I/81U-4L9V3fL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B07N1BBNV7",
            "brand": "Fully"
        }
    ],
    "Video Games": [
        {
            "name": "Wireless Pro Controller",
            "price": "$69.99",
            "image": "https://m.media-amazon.com/images/I/61vJtS86S5L._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08H75RTZ1",
            "brand": "Nintendo"
        },
        {
            "name": "Next-Gen Gaming Console",
            "price": "$499.00",
            "image": "https://m.media-amazon.com/images/I/71W77Tj5RSL._AC_SL1500_.jpg",
            "url": "https://www.amazon.com/dp/B08LLD2B8U",
            "brand": "Sony"
        }
    ]
}

async def manual_seed():
    from vector_store import clear_vector_store
    print("🧹 Clearing vector store for fresh seed...", flush=True)
    clear_vector_store()
    
    print("💎 Injecting Gold Standard products...")
    ingest_items = []
    
    for category, products in GOLD_PRODUCTS.items():
        for p in products:
            img = p.get('image_url') or p.get('image')
            url = p.get('source_url') or p.get('url')
            name = p.get('name')
            brand = p.get('brand')
            price = p.get('price')
            
            description = (
                f"Product: {name}\n"
                f"Brand: {brand}\n"
                f"Price: {price}\n"
                f"Category: {category}\n"
                f"Image URL: {img}\n"
                f"Source URL: {url}"
            )
            
            metadata = {
                "source": url,
                "type": "gold_seed",
                "category": category,
                "image_url": img,
                "name": name,
                "price": price,
                "brand": brand,
                "store_source": "Amazon Verified"
            }
            
            # Legacy Accio compatibility
            p_compat = {
                "name": name,
                "price": price,
                "image": img,
                "url": url,
                "brand": brand
            }
            
            ingest_items.append({
                "content": description,
                "url": url,
                "metadata": metadata
            })
            
    if ingest_items:
        await add_multiple_contents_to_store(ingest_items)
        print(f"✅ Successfully injected {len(ingest_items)} premier products.")

if __name__ == "__main__":
    asyncio.run(manual_seed())
