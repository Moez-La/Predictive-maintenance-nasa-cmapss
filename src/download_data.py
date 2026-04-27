"""
Download NASA C-MAPSS Turbofan Engine Degradation Dataset
"""
import urllib.request
import zipfile
import os

def download_cmapss_data():
    """Download and extract NASA C-MAPSS dataset"""
    
    data_dir = 'data'
    url = 'https://ti.arc.nasa.gov/c/6/'
    filename = 'CMAPSSData.zip'
    filepath = os.path.join(data_dir, filename)
    
    print("Downloading NASA C-MAPSS dataset...")
    print(f"URL: {url}")
    
    # Download
    if not os.path.exists(filepath):
        urllib.request.urlretrieve(url, filepath)
        print(f"✓ Downloaded to {filepath}")
    else:
        print(f"✓ File already exists: {filepath}")
    
    # Extract
    print("Extracting files...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("✓ Extraction complete!")
    
    # List extracted files
    print("\nExtracted files:")
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            print(f"  - {file}")

if __name__ == "__main__":
    download_cmapss_data()
