import os
import requests
import zipfile

def get_gdrive_direct_url(share_url):
    # Extract the file ID from the shared Google Drive URL
    try:
        file_id = share_url.split('/d/')[1].split('/')[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    except Exception as e:
        raise ValueError("Invalid Google Drive URL format") from e

def download_zip_if_needed(zip_url, zip_path):
    if os.path.exists(zip_path):
        print(f"{zip_path} already exists locally. Skipping download.")
        return
    print(f"Downloading {zip_url} ...")
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {zip_path}.")

def extract_zip_if_needed(zip_path, extract_dir):
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"{extract_dir} already contains data. Skipping extraction.")
        return
    print(f"Extracting {zip_path} to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extraction complete.")

if __name__ == "__main__":
    # Your shared Google Drive link
    GDRIVE_SHARE_URL = "https://drive.google.com/file/d/1E3byA-tBtjVy58hPlFhSdaWmXkxXl3pS/view?usp=drivesdk"
    ZIP_PATH = "data.zip"
    DATA_DIR = "data"

    # Get the direct download URL
    ZIP_URL = get_gdrive_direct_url(GDRIVE_SHARE_URL)

    download_zip_if_needed(ZIP_URL, ZIP_PATH)
    extract_zip_if_needed(ZIP_PATH, DATA_DIR)

    print("Data is ready in:", DATA_DIR)
