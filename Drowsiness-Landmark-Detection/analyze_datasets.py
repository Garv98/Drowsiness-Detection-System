import os
import cv2
from collections import defaultdict
import numpy as np
from pathlib import Path
import random

def parse_mrl_filename(filename):
    # Parse MRL Eye Dataset filename format: s0014_03559_0_0_0_0_1_02
    parts = filename.split('_')
    if len(parts) != 8:
        return None
    
    return {
        'subject_id': parts[0],
        'image_id': parts[1],
        'gender': 'woman' if parts[2] == '1' else 'man',
        'glasses': 'yes' if parts[3] == '1' else 'no',
        'eye_state': 'open' if parts[4] == '1' else 'closed',
        'reflections': {'0': 'none', '1': 'small', '2': 'big'}[parts[5]],
        'lighting': 'good' if parts[6] == '1' else 'bad',
        'sensor': {
            '01': 'RealSense (640x480)',
            '02': 'IDS (1280x1024)',
            '03': 'Aptina (752x480)'
        }[parts[7]]
    }

def analyze_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return {
        'shape': img.shape,
        'size_mb': os.path.getsize(img_path) / (1024 * 1024),
        'mean_rgb': tuple(img.mean(axis=(0,1)).astype(int)),
        'format': Path(img_path).suffix
    }

def analyze_mrl_dataset(root_dir, sample_size=1000):
    stats = defaultdict(lambda: defaultdict(int))
    image_stats = defaultdict(list)
    
    # Get list of all image files
    image_files = []
    for path in Path(root_dir).rglob('*.*'):
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.append(path)
    
    total_files = len(image_files)
    print(f"\nTotal files found in {root_dir}: {total_files}")
    
    # Take a random sample
    if total_files > sample_size:
        print(f"Analyzing random sample of {sample_size} images...")
        image_files = random.sample(image_files, sample_size)
    else:
        print(f"Analyzing all {total_files} images...")
    
    for path in image_files:
        # Parse filename
        metadata = parse_mrl_filename(path.stem)
        if metadata:
            for key, value in metadata.items():
                stats[key][value] += 1
        
        # Analyze image properties
        result = analyze_image(path)
        if result:
            image_stats['shapes'].append(result['shape'])
            image_stats['sizes'].append(result['size_mb'])
    
    print("\nMetadata Distribution (from sample):")
    print("-" * 50)
    
    sample_size = len(image_files)  # actual number analyzed
    # Skip image_id in the output and show only relevant metadata
    important_categories = ['gender', 'glasses', 'eye_state', 'reflections', 'lighting', 'sensor']
    for category in important_categories:
        if category in stats:
            print(f"\n{category.replace('_', ' ').title()}:")
            for value, count in stats[category].items():
                print(f"  {value}: {count} ({count/sample_size*100:.1f}%)")
    
    if image_stats['shapes']:
        shapes = np.array([s[:2] for s in image_stats['shapes']])
        print("\nImage Statistics (from sample):")
        print("-" * 50)
        print(f"Average dimensions (h×w): {shapes.mean(axis=0).astype(int)}")
        print(f"Min dimensions (h×w): {shapes.min(axis=0)}")
        print(f"Max dimensions (h×w): {shapes.max(axis=0)}")
        print(f"Average file size: {np.mean(image_stats['sizes']):.2f} MB")

def analyze_regular_dataset(root_dir):
    stats = defaultdict(list)
    total_files = 0
    formats = defaultdict(int)
    sizes = []
    
    print(f"\nAnalyzing {root_dir}...")
    
    for path in Path(root_dir).rglob('*.*'):
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            total_files += 1
            formats[path.suffix.lower()] += 1
            
            result = analyze_image(path)
            if result:
                stats['shapes'].append(result['shape'])
                stats['sizes'].append(result['size_mb'])
                sizes.append(result['shape'][:2])
    
    if not stats['shapes']:
        print("No valid images found!")
        return
    
    shapes = np.array(sizes)
    mean_shape = shapes.mean(axis=0).astype(int)
    min_shape = shapes.min(axis=0)
    max_shape = shapes.max(axis=0)
    
    print(f"Total files: {total_files}")
    print(f"Image formats: {dict(formats)}")
    print(f"Average dimensions (h×w): {mean_shape}")
    print(f"Min dimensions (h×w): {min_shape}")
    print(f"Max dimensions (h×w): {max_shape}")
    print(f"Average file size: {np.mean(stats['sizes']):.2f} MB")

def analyze_video(video_path):
    """Analyze a video file and return its properties"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Read first frame for additional analysis
    ret, frame = cap.read()
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration_seconds': duration,
        'size_mb': os.path.getsize(video_path) / (1024 * 1024)
    }

def analyze_yawdd_dataset(root_dir):
    """Analyze YawDD dataset including its video files"""
    video_stats = defaultdict(list)
    gender_stats = defaultdict(int)
    formats = defaultdict(int)
    
    print(f"\nAnalyzing YawDD videos in {root_dir}...")
    
    # Recursively find all video files
    video_files = []
    for path in Path(root_dir).rglob('*.*'):
        if path.suffix.lower() in ['.avi', '.mp4', '.mov']:
            video_files.append(path)
            formats[path.suffix.lower()] += 1
            
            # Count gender based on directory structure
            if 'Female' in str(path):
                gender_stats['Female'] += 1
            elif 'Male' in str(path):
                gender_stats['Male'] += 1
    
    total_videos = len(video_files)
    print(f"\nTotal videos found: {total_videos}")
    print(f"Video formats: {dict(formats)}")
    print(f"\nGender distribution:")
    for gender, count in gender_stats.items():
        print(f"  {gender}: {count} ({count/total_videos*100:.1f}%)")
    
    # Analyze video properties
    print("\nAnalyzing video properties...")
    for video_path in video_files:
        props = analyze_video(video_path)
        if props:
            for key, value in props.items():
                video_stats[key].append(value)
    
    if video_stats:
        print("\nVideo Statistics:")
        print("-" * 50)
        print(f"Resolution range: {min(video_stats['width'])}x{min(video_stats['height'])} to {max(video_stats['width'])}x{max(video_stats['height'])}")
        print(f"Average FPS: {np.mean(video_stats['fps']):.2f}")
        print(f"Average duration: {np.mean(video_stats['duration_seconds']):.2f} seconds")
        print(f"Average file size: {np.mean(video_stats['size_mb']):.2f} MB")
        print(f"Total duration: {sum(video_stats['duration_seconds'])/3600:.2f} hours")

# Paths to datasets
datasets = {
    "MRL Eye": r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\MRLEye\mrleyedataset",
    "Driver Drowsiness": r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\DriverDrowsiness\Driver Drowsiness Dataset (DDD)",
    "YawDD": r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\YawDD"
}

print("Dataset Analysis")
print("=" * 50)

for name, path in datasets.items():
    print(f"\n{name} Dataset:")
    print("-" * 50)
    
    if name == "MRL Eye":
        print("\nAnalyzing Close-Eyes...")
        analyze_mrl_dataset(os.path.join(path, "Close-Eyes"))
        print("\nAnalyzing Open-Eyes...")
        analyze_mrl_dataset(os.path.join(path, "Open-Eyes"))
    
    elif name == "Driver Drowsiness":
        print("\nAnalyzing Drowsy...")
        analyze_regular_dataset(os.path.join(path, "Drowsy"))
        print("\nAnalyzing Non Drowsy...")
        analyze_regular_dataset(os.path.join(path, "Non Drowsy"))
    
    elif name == "YawDD":
        print("\nAnalyzing Dash camera videos...")
        analyze_yawdd_dataset(os.path.join(path, "Dash"))
        print("\nAnalyzing Mirror camera videos...")
        analyze_yawdd_dataset(os.path.join(path, "Mirror"))

print("\nAnalysis Complete!")
