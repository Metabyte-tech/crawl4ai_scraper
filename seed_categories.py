import asyncio
import os
from db_service import db_service

async def seed():
    print("Seeding universal retail categories...")
    await db_service.init_db()
    pool = await db_service.get_pool()
    
    categories = [
        ("Baby & Kids", "baby-kids", None),
        ("Electronics", "electronics", None),
        ("Home & Kitchen", "home-kitchen", None),
        ("Fashion", "fashion", None),
        ("Beauty & Health", "beauty-health", None),
        ("Sports & Outdoors", "sports-outdoors", None),
        ("Grocery", "grocery", None),
        ("Industrial & Scientific", "industrial-scientific", None),
        ("Pet Supplies", "pet-supplies", None),
        ("Automotive", "automotive", None),
        ("Office Products", "office-products", None),
        ("Video Games", "video-games", None)
    ]
    
    subcategories = [
        ("Toys", "toys", "baby-kids"),
        ("Clothing", "kids-clothing", "baby-kids"),
        ("Mobiles", "mobiles", "electronics"),
        ("Laptops", "laptops", "electronics"),
        ("Furniture", "furniture", "home-kitchen"),
        ("Appliances", "appliances", "home-kitchen"),
        ("Men", "fashion-men", "fashion"),
        ("Women", "fashion-women", "fashion"),
        ("Lab Supplies", "lab-supplies", "industrial-scientific"),
        ("Power Tools", "power-tools", "industrial-scientific"),
        ("Dog Food", "dog-food", "pet-supplies"),
        ("Cat Accessories", "cat-accessories", "pet-supplies"),
        ("Tires", "tires", "automotive"),
        ("Interior Accessories", "car-interior", "automotive"),
        ("Stationery", "stationery", "office-products"),
        ("Consoles", "consoles", "video-games")
    ]
    
    async with pool.acquire() as conn:
        # Seed Top Level
        for name, slug, _ in categories:
            await conn.execute("""
                INSERT INTO global_categories (name, slug, location_code)
                VALUES ($1, $2, $3)
                ON CONFLICT (slug) DO NOTHING
            """, name, slug, "Global")
        
        # Seed Subcategories
        for name, slug, parent_slug in subcategories:
            parent_id = await conn.fetchval("SELECT id FROM global_categories WHERE slug = $1", parent_slug)
            if parent_id:
                await conn.execute("""
                    INSERT INTO global_categories (name, slug, parent_id, location_code)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (slug) DO NOTHING
                """, name, slug, parent_id, "Global")

    print("Seeding complete.")

if __name__ == "__main__":
    asyncio.run(seed())
