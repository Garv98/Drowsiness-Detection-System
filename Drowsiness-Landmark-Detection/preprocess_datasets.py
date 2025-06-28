import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Configuration
CONFIG = {
    'target_size': (224, 224),
    'video_sample_fps': 2,  # Extract 2 frames per second from videos
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'seed': 42
}

# Dataset paths
PATHS = {
    'mrl': r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\MRLEye\mrleyedataset",
    'ddd': r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\DriverDrowsiness\Driver Drowsiness Dataset (DDD)",
    'yawdd': r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\YawDD",
    'output': r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\Processed_Dataset"
}

def create_output_dirs():
    """Create output directory structure"""
    splits = ['train', 'val', 'test']
    states = ['drowsy', 'alert']
    
    for split in splits:
        for state in states:
            path = os.path.join(PATHS['output'], split, state)
            os.makedirs(path, exist_ok=True)
    print("Created output directory structure")

def parse_mrl_filename(filename):
    """Parse MRL Eye Dataset filename metadata"""
    parts = filename.split('_')
    if len(parts) != 8:
        return None
    return {
        'eye_state': 'open' if parts[4] == '1' else 'closed',
        'lighting': 'good' if parts[6] == '1' else 'bad',
        'reflections': parts[5]  # '0'=none, '1'=small, '2'=big
    }

def preprocess_image(img, target_size):
    """Preprocess image to target size with padding if needed"""
    if img is None:
        return None
        
    # Calculate scaling factor to maintain aspect ratio
    h, w = img.shape[:2]
    scale = min(target_size[0]/h, target_size[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    
    # Resize image
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create blank canvas
    processed = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    
    # Place resized image on canvas
    processed[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return processed

def process_mrl_dataset():
    """Process MRL Eye Dataset"""
    print("\nProcessing MRL Eye Dataset...")
    good_images = []
    
    # Process both Open and Closed eyes
    for eye_state_dir in ['Open-Eyes', 'Close-Eyes']:
        dir_path = os.path.join(PATHS['mrl'], eye_state_dir)
        is_open = 'Open' in eye_state_dir
        
        for img_path in tqdm(list(Path(dir_path).rglob('*.*'))):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            # Parse metadata
            metadata = parse_mrl_filename(img_path.stem)
            if metadata is None:
                continue
            
            # Filter for good quality images
            if metadata['lighting'] == 'good' and metadata['reflections'] == '0':
                img = cv2.imread(str(img_path))
                processed = preprocess_image(img, CONFIG['target_size'])
                if processed is not None:
                    good_images.append({
                        'image': processed,
                        'state': 'alert' if is_open else 'drowsy'
                    })
    
    print(f"Processed {len(good_images)} good quality images from MRL dataset")
    return good_images

def process_ddd_dataset():
    """Process Driver Drowsiness Dataset"""
    print("\nProcessing Driver Drowsiness Dataset...")
    processed_images = []
    
    # Process both Drowsy and Non-Drowsy
    for state in ['Drowsy', 'Non Drowsy']:
        dir_path = os.path.join(PATHS['ddd'], state)
        is_drowsy = state == 'Drowsy'
        
        for img_path in tqdm(list(Path(dir_path).rglob('*.png'))):
            img = cv2.imread(str(img_path))
            processed = preprocess_image(img, CONFIG['target_size'])
            if processed is not None:
                processed_images.append({
                    'image': processed,
                    'state': 'drowsy' if is_drowsy else 'alert'
                })
    
    print(f"Processed {len(processed_images)} images from DDD dataset")
    return processed_images

def extract_video_frames(video_path, sample_fps):
    """Extract frames from video at given sample rate"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return frames
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / sample_fps)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            processed = preprocess_image(frame, CONFIG['target_size'])
            if processed is not None:
                frames.append(processed)
                
        frame_count += 1
        
    cap.release()
    return frames

def process_yawdd_dataset():
    """Process YawDD Dataset videos"""
    print("\nProcessing YawDD Dataset...")
    processed_frames = []
    
    # Process both Dash and Mirror cameras
    for camera in ['Dash', 'Mirror']:
        camera_path = os.path.join(PATHS['yawdd'], camera)
        
        for video_path in tqdm(list(Path(camera_path).rglob('*.avi'))):
            frames = extract_video_frames(video_path, CONFIG['video_sample_fps'])
            # For now, mark all video frames as alert (you might want to adjust this based on annotations)
            for frame in frames:
                processed_frames.append({
                    'image': frame,
                    'state': 'alert'
                })
    
    print(f"Extracted {len(processed_frames)} frames from YawDD videos")
    return processed_frames

def save_processed_images(images, split_type, output_dir):
    """Save processed images to appropriate directories"""
    for idx, img_data in enumerate(images):
        state_dir = os.path.join(output_dir, split_type, img_data['state'])
        out_path = os.path.join(state_dir, f"{split_type}_{idx:06d}.png")
        cv2.imwrite(out_path, img_data['image'])

def main():
    # Set random seed for reproducibility
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create output directories
    create_output_dirs()
    
    # Process all datasets
    all_images = []
    all_images.extend(process_mrl_dataset())
    all_images.extend(process_ddd_dataset())
    all_images.extend(process_yawdd_dataset())
    
    # Shuffle the dataset
    random.shuffle(all_images)
    
    # Split into train/val/test
    train_val_images, test_images = train_test_split(
        all_images, 
        test_size=CONFIG['test_split'], 
        random_state=CONFIG['seed']
    )
    
    val_size = CONFIG['val_split'] / (CONFIG['train_split'] + CONFIG['val_split'])
    train_images, val_images = train_test_split(
        train_val_images,
        test_size=val_size,
        random_state=CONFIG['seed']
    )
    
    # Save splits
    print("\nSaving processed images...")
    save_processed_images(train_images, 'train', PATHS['output'])
    save_processed_images(val_images, 'val', PATHS['output'])
    save_processed_images(test_images, 'test', PATHS['output'])
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total images: {len(all_images)}")
    print(f"Training: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")
    
    # Count class distribution
    train_drowsy = sum(1 for img in train_images if img['state'] == 'drowsy')
    print(f"\nTraining set class distribution:")
    print(f"Drowsy: {train_drowsy} ({train_drowsy/len(train_images)*100:.1f}%)")
    print(f"Alert: {len(train_images)-train_drowsy} ({(1-train_drowsy/len(train_images))*100:.1f}%)")

if __name__ == "__main__":
    main()
