import os
import zipfile
import shutil

# Paths to your zip files
zips = {
    "YawDD": r"C:\Users\garva\Downloads\archive (10).zip",
    "DriverDrowsiness": r"C:\Users\garva\Downloads\archive (11).zip",
    "MRLEye": r"C:\Users\garva\Downloads\archive (12).zip"
}

# Destination root in AIML_LAB_EL folder (parent of Drowsiness-Landmark-Detection)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gets AIML_LAB_EL
dest_root = os.path.join(project_root, 'Data')

# Create Data directory if it doesn't exist
os.makedirs(dest_root, exist_ok=True)

for name, zip_path in zips.items():
    dest_dir = os.path.join(dest_root, name)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Extracting {zip_path} to {dest_dir} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f"Done extracting {name}.")

print("All datasets extracted successfully.")
