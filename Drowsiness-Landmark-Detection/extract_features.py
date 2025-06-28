import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import deque
import warnings
import argparse
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class FacialFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # For head pose estimation
        self.NOSE_TIP = 4
        self.CHIN = 152
        self.LEFT_EYE_LEFT = 33
        self.LEFT_EYE_RIGHT = 133
        self.RIGHT_EYE_LEFT = 362
        self.RIGHT_EYE_RIGHT = 263
        
        # Constants
        self.EAR_THRESHOLD = 0.2
        self.history_size = 30  # Number of frames to keep in history
        
        # Initialize history
        self.ear_history = deque(maxlen=self.history_size)
        self.closed_frames = 0
    
    def get_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio"""
        try:
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            return np.clip(ear, 0.0, 1.0)  # Clamp between 0 and 1
        except:
            return 0.0

    def estimate_head_rotation(self, face_landmarks):
        """Estimate head rotation using geometric approach"""
        try:
            # Get eye midpoints
            left_eye_mid = (face_landmarks[self.LEFT_EYE_LEFT] + face_landmarks[self.LEFT_EYE_RIGHT]) / 2
            right_eye_mid = (face_landmarks[self.RIGHT_EYE_LEFT] + face_landmarks[self.RIGHT_EYE_RIGHT]) / 2
            
            # Get nose and chin points
            nose = face_landmarks[self.NOSE_TIP]
            chin = face_landmarks[self.CHIN]
            
            # Calculate angles
            eye_line = right_eye_mid - left_eye_mid
            yaw = np.arctan2(eye_line[2], eye_line[0])
            
            nose_chin = chin - nose
            pitch = np.arctan2(nose_chin[1], np.sqrt(nose_chin[0]**2 + nose_chin[2]**2))
            
            roll = np.arctan2(right_eye_mid[1] - left_eye_mid[1], 
                            right_eye_mid[0] - left_eye_mid[0])
            
            # Convert to degrees and normalize
            angles = np.degrees([pitch, yaw, roll])
            return np.clip(angles, -90, 90)  # Clamp between -90 and 90 degrees
        except:
            return np.array([0.0, 0.0, 0.0])
    
    def update_ear_stats(self, ear):
        """Update EAR statistics"""
        self.ear_history.append(ear)
        
        if ear <= self.EAR_THRESHOLD:
            self.closed_frames += 1
    
    def calculate_perclos(self):
        """Calculate PERCLOS based on frame history"""
        if not self.ear_history:
            return 0.0
        
        closed_count = sum(1 for ear in self.ear_history if ear <= self.EAR_THRESHOLD)
        return closed_count / len(self.ear_history)
    
    def extract_features(self, image):
        """Extract all facial features from an image"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            results = self.face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            face_landmarks = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks.landmark])
            
            # Calculate EAR
            left_eye = np.array([face_landmarks[i] for i in self.LEFT_EYE])
            right_eye = np.array([face_landmarks[i] for i in self.RIGHT_EYE])
            left_ear = self.get_ear(left_eye)
            right_ear = self.get_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Update statistics
            self.update_ear_stats(avg_ear)
            
            # Calculate head rotation
            pitch, yaw, roll = self.estimate_head_rotation(face_landmarks)
            
            # Calculate PERCLOS
            perclos = self.calculate_perclos()
            
            return {
                'ear': float(avg_ear),
                'left_ear': float(left_ear),
                'right_ear': float(right_ear),
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll),
                'perclos': float(perclos)
            }
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

def process_dataset(data_dir, test_mode=False, resume=False, sample_size=100):
    """Process all images in the dataset and extract features
    
    Args:
        data_dir: Path to dataset
        test_mode: If True, only process sample_size images per category
        resume: If True, try to resume from last saved state
        sample_size: Number of images to process in test mode
    """
    extractor = FacialFeatureExtractor()
    features = []
    
    # Load existing features if resuming
    output_path = Path(data_dir).parent / 'features.csv'
    processed_files = set()
    if resume and output_path.exists():
        print("Resuming from existing features.csv...")
        df = pd.read_csv(output_path)
        features = df.to_dict('records')
        processed_files = set(df['image_path'].values)
        print(f"Loaded {len(processed_files)} previously processed files")
    
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        for state in ['drowsy', 'alert']:
            state_dir = split_dir / state
            print(f"\nProcessing {split}/{state} images...")
            
            # Reset statistics for each new category
            extractor.ear_history.clear()
            extractor.closed_frames = 0
            
            # Get list of files to process
            image_files = list(state_dir.glob('*.png'))
            if test_mode:
                image_files = image_files[:sample_size]
                print(f"Test mode: processing {len(image_files)} images")
            
            for img_path in tqdm(image_files):
                try:
                    # Skip if already processed
                    if str(img_path) in processed_files:
                        continue
                        
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    features_dict = extractor.extract_features(img)
                    if features_dict is None:
                        continue
                    
                    features_dict.update({
                        'split': split,
                        'state': state,
                        'image_path': str(img_path)
                    })
                    
                    features.append(features_dict)
                    
                    # Save progress periodically
                    if len(features) % 1000 == 0:
                        save_features(features, output_path)
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue
    
    # Final save
    save_features(features, output_path)
    print_statistics(features)

def save_features(features, output_path):
    """Save features to CSV"""
    df = pd.DataFrame(features)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(features)} features to {output_path}")

def print_statistics(features):
    """Print feature statistics"""
    df = pd.DataFrame(features)
    
    print("\nFeature Statistics:")
    print(df.groupby(['split', 'state']).size())
    print("\nFeature Averages by State:")
    print(df.groupby('state')[['ear', 'perclos', 'pitch', 'yaw', 'roll']].mean())
    
    print("\nFeature Ranges:")
    for col in ['ear', 'perclos', 'pitch', 'yaw', 'roll']:
        print(f"{col}:")
        print(f"  Min: {df[col].min():.3f}")
        print(f"  Max: {df[col].max():.3f}")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Std: {df[col].std():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract facial features from dataset')
    parser.add_argument('--test', action='store_true', help='Run in test mode with sample of images')
    parser.add_argument('--resume', action='store_true', help='Resume from last saved state')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of images per category in test mode')
    args = parser.parse_args()
    
    PROCESSED_DATA_DIR = r"C:\Users\garva\OneDrive\Desktop\AIML_LAB_EL\Data\Processed_Dataset"
    
    process_dataset(
        PROCESSED_DATA_DIR,
        test_mode=args.test,
        resume=args.resume,
        sample_size=args.sample_size
    )
