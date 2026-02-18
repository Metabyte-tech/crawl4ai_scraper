import urllib.request
from s3_service import s3_service

def test_s3_upload():
    print("--- Testing S3 Upload (No ACL) ---")
    
    # Small test payload
    test_content = b"This is a test image content for verification."
    test_filename = "tests/verify-upload.txt"
    
    print(f"Uploading to bucket: {s3_service.bucket_name}...")
    url = s3_service.upload_image(test_content, test_filename, content_type='text/plain')
    
    if url:
        print(f"✅ Upload successful. Generated URL: {url}")
        
        # Test if the URL is accessible
        print("Testing public access via HTTP...")
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    print("✅ Public access test passed!")
                else:
                    print(f"❌ Public access failed. Status: {response.status}")
        except Exception as e:
            print(f"❌ Error testing accessibility: {e}")
            print("NOTE: You likely need to add a Public Read bucket policy because ACLs are disabled.")
    else:
        print("❌ Upload failed.")

if __name__ == "__main__":
    test_s3_upload()
