import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

class S3Service:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        self.bucket_name = os.getenv("S3_BUCKET_NAME")

    def upload_image(self, file_content, file_name, content_type='image/jpeg'):
        """
        Uploads an image to S3 and returns the public URL.
        """
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=file_content,
                ContentType=content_type
            )
            
            # Construct URL using the specified region (recommended for Mumbai and others)
            region = os.getenv("AWS_REGION", "us-east-1")
            if region == "us-east-1":
                url = f"https://{self.bucket_name}.s3.amazonaws.com/{file_name}"
            else:
                url = f"https://{self.bucket_name}.s3.{region}.amazonaws.com/{file_name}"
                
            print(f"Successfully uploaded to S3: {url}")
            return url
        except NoCredentialsError:
            print("S3 Error: Credentials not available")
            return None
        except Exception as e:
            print(f"Error uploading to S3 ({file_name}): {e}")
            return None

s3_service = S3Service()
