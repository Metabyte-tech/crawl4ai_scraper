import os
import boto3
from dotenv import load_dotenv

load_dotenv()
s3 = boto3.client('s3')
bucket = os.getenv('S3_BUCKET_NAME')
try:
    s3.head_object(Bucket=bucket, Key='products/brands/rainbow-designs-distributor/4898594f-f91b-4372-8566-7640dfd5e513.jpg')
    print("Rainbow Designs Image Exists!")
except Exception as e:
    print("Rainbow Designs Image Missing:", e)
    
try:
    s3.head_object(Bucket=bucket, Key='products/brands/playmobil-distributor/3b32ff43-2a33-47c9-a267-4d90cd031a86.jpg')
    print("Playmobil Image Exists!")
except Exception as e:
    print("Playmobil Image Missing:", e)
