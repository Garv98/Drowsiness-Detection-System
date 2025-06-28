import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
import threading
import time
import queue
from collections import deque
import pygame  # For alert sound
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pygame for audio
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Constants for drowsiness detection
        self.EAR_THRESHOLD = 0.2  # Threshold for eye closure (lower value)
        self.PERCLOS_THRESHOLD = 0.15  # 15% threshold for PERCLOS
        self.DROWSY_TIME = 1.0  # Seconds of eye closure for drowsiness
        
        # Constants for normalization
        self.EAR_NORMAL_MAX = 0.35  # Maximum normal EAR value
        self.EAR_NORMAL_MIN = 0.1   # Minimum normal EAR value
        self.HEAD_POSE_LIMIT = 90.0  # Maximum head pose angle
        
        # Initialize metrics tracking
        self.ear_history = deque(maxlen=30)  # 1 second at 30 fps
        self.perclos_window = deque(maxlen=90)  # 3 seconds at 30 fps
        self.closed_eyes_start = None
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds
        
        # Load ML model and scaler
        try:
            self.model = joblib.load('model/best_model.pkl')
            self.scaler = joblib.load('model/scaler.pkl')
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or scaler: {e}")
            self.model = None
            self.scaler = None
        
        # Load alert sound
        try:
            pygame.mixer.music.load("alert.wav")
            pygame.mixer.music.set_volume(0.5)  # Set volume to 50%
        except:
            logger.error("Could not load alert sound")

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate the Eye Aspect Ratio (EAR) for one eye"""
        p1 = landmarks[eye_indices[1]]
        p2 = landmarks[eye_indices[2]]
        p3 = landmarks[eye_indices[3]]
        p4 = landmarks[eye_indices[4]]
        p5 = landmarks[eye_indices[5]]
        p6 = landmarks[eye_indices[0]]

        # Calculate Euclidean distances
        height1 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
        height2 = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
        width = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))
        
        # Avoid division by zero
        if width == 0:
            return 0.0
            
        # Calculate EAR
        ear = (height1 + height2) / (2.0 * width)
        
        # Normalize EAR value
        ear = min(max(ear, self.EAR_NORMAL_MIN), self.EAR_NORMAL_MAX)
        
        return ear

    def process_frame(self, frame):
        """Process a single frame and return the results"""
        if frame is None:
            return frame, None, 0
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        features = None
        prediction = 0  # Default to alert state
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Draw face mesh
            self.mp_draw.draw_landmarks(
                frame, 
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            # Calculate EAR
            left_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [33, 160, 158, 133, 153, 144]
            
            left_ear = self.calculate_ear(face_landmarks.landmark, left_eye_indices)
            right_ear = self.calculate_ear(face_landmarks.landmark, right_eye_indices)
            ear = (left_ear + right_ear) / 2
            self.ear_history.append(ear)
            
            # Update drowsiness detection
            current_time = time.time()
            
            # Detect eye closure and update PERCLOS
            if ear < self.EAR_THRESHOLD:
                if self.closed_eyes_start is None:
                    self.closed_eyes_start = current_time
                eyes_closed_time = current_time - self.closed_eyes_start
                self.perclos_window.append(1)
                
                # Check for prolonged eye closure
                if eyes_closed_time >= self.DROWSY_TIME:
                    prediction = 1  # Set drowsy state
            else:
                self.closed_eyes_start = None
                self.perclos_window.append(0)
            
            # Calculate PERCLOS
            perclos = sum(self.perclos_window) / len(self.perclos_window) if self.perclos_window else 0
            
            # Calculate head pose
            size = frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            dist_coeffs = np.zeros((4,1))
            
            # 3D model points
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),   # Right eye corner
                (-150.0, -150.0, -125.0), # Left mouth corner
                (150.0, -150.0, -125.0)   # Right mouth corner
            ]) / 4.5
            
            # Get 2D points
            landmarks = face_landmarks.landmark
            image_points = np.array([
                (landmarks[1].x * size[1], landmarks[1].y * size[0]),    # Nose tip
                (landmarks[152].x * size[1], landmarks[152].y * size[0]),  # Chin
                (landmarks[226].x * size[1], landmarks[226].y * size[0]),  # Left eye corner
                (landmarks[446].x * size[1], landmarks[446].y * size[0]),  # Right eye corner
                (landmarks[57].x * size[1], landmarks[57].y * size[0]),   # Left mouth corner
                (landmarks[287].x * size[1], landmarks[287].y * size[0])  # Right mouth corner
            ], dtype=np.float32)
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs)
            
            if success:
                # Get rotation matrix and angles
                rotation_mat, _ = cv2.Rodrigues(rotation_vec)
                
                # Extract Euler angles
                sy = np.sqrt(rotation_mat[0, 0] * rotation_mat[0, 0] + rotation_mat[1, 0] * rotation_mat[1, 0])
                singular = sy < 1e-6

                if not singular:
                    pitch = np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]) * 180 / np.pi
                    yaw = np.arctan2(-rotation_mat[2, 0], sy) * 180 / np.pi
                    roll = np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]) * 180 / np.pi
                else:
                    pitch = np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1]) * 180 / np.pi
                    yaw = np.arctan2(-rotation_mat[2, 0], sy) * 180 / np.pi
                    roll = 0
                
                # Normalize angles to -90 to +90 range
                pitch = np.clip(pitch, -self.HEAD_POSE_LIMIT, self.HEAD_POSE_LIMIT)
                yaw = np.clip(yaw, -self.HEAD_POSE_LIMIT, self.HEAD_POSE_LIMIT)
                roll = np.clip(roll, -self.HEAD_POSE_LIMIT, self.HEAD_POSE_LIMIT)
                
                # Create feature vector
                features = {
                    'ear': ear,
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'perclos': perclos
                }
                
                # Check if PERCLOS indicates drowsiness
                if perclos > self.PERCLOS_THRESHOLD:
                    prediction = 1
                
                # Use ML model if available and no clear drowsiness detected
                if prediction == 0 and self.model is not None and self.scaler is not None:
                    feature_vector = np.array([[
                        features['ear'],
                        features['left_ear'],
                        features['right_ear'],
                        features['pitch'],
                        features['yaw'],
                        features['roll'],
                        features['perclos']
                    ]])
                    scaled_features = self.scaler.transform(feature_vector)
                    prediction = self.model.predict(scaled_features)[0]
            
            # Draw status on frame
            label = "DROWSY" if prediction == 1 else "ALERT"
            color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
            cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Handle alert sound
            if prediction == 1:
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    if not pygame.mixer.music.get_busy():
                        pygame.mixer.music.play()
                    self.last_alert_time = current_time
            else:
                pygame.mixer.music.stop()
        
        return frame, features, prediction

def main():
    st.set_page_config(page_title="Drowsiness Detection", layout="wide")
    
    # Title and description
    st.title("Real-time Drowsiness Detection")
    st.markdown("""
    This system monitors:
    - Eye Aspect Ratio (EAR)
    - PERCLOS (Percentage of eye closure)
    - Head pose (Pitch, Yaw, Roll)
    - ML-based drowsiness prediction
    """)
    
    # Create detector instance
    detector = DrowsinessDetector()

    # Initialize session state
    if 'ear' not in st.session_state:
        st.session_state.ear = 0.0
    if 'perclos' not in st.session_state:
        st.session_state.perclos = 0.0
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
    if 'head_pose' not in st.session_state:
        st.session_state.head_pose = (0.0, 0.0, 0.0)

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        # Placeholder for video feed
        video_placeholder = st.empty()

    with col2:
        # Placeholders for metrics
        status_placeholder = st.empty()
        ear_placeholder = st.empty()
        perclos_placeholder = st.empty()
        head_pose_placeholder = st.empty()
        
        # Add threshold indicators
        st.markdown("### Thresholds")
        st.write(f"EAR Threshold: {detector.EAR_THRESHOLD:.2f}")
        st.write(f"PERCLOS Threshold: {detector.PERCLOS_THRESHOLD:.2f}")
        st.write(f"Drowsy Time Threshold: {detector.DROWSY_TIME:.1f}s")

    # Add stop button
    stop_button = st.button("Stop")

    # Camera selection
    available_cameras = []
    for i in range(5):  # Check first 5 possible camera indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if not available_cameras:
        st.error("No cameras found!")
        return
    
    # Camera selection dropdown
    camera_index = st.selectbox(
        "Select Camera",
        available_cameras,
        format_func=lambda x: f"Camera {x}"
    )
    
    # Initialize video capture with selected camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Try different resolutions in order of preference
    resolutions = [
        (640, 480),
        (1280, 720),
        (848, 480),
        (800, 600),
        (320, 240)
    ]
    
    success = False
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify if settings were applied
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if abs(actual_width - width) < 100 and abs(actual_height - height) < 100:
            success = True
            st.info(f"Camera initialized at {int(actual_width)}x{int(actual_height)}")
            break
    
    if not success:
        st.warning("Could not set preferred resolution. Using default camera settings.")
    
    # Set FPS
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify camera is working
    if not cap.isOpened():
        st.error("Error: Could not open camera!")
        return
    
    # Read test frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        st.error("Error: Could not read from camera!")
        cap.release()
        return
        
    # Add camera controls
    st.sidebar.markdown("### Camera Controls")
    brightness = st.sidebar.slider("Brightness", -64, 64, 0)
    contrast = st.sidebar.slider("Contrast", -64, 64, 0)
    
    # Apply camera controls
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness + 64)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast + 64)

    try:
        frame_counter = 0
        retry_count = 0
        max_retries = 3
        
        while not stop_button:
            ret, frame = cap.read()
            
            if not ret or frame is None:
                retry_count += 1
                st.warning(f"Camera read failed. Attempt {retry_count} of {max_retries}")
                time.sleep(1)  # Wait before retrying
                
                if retry_count >= max_retries:
                    st.error("Failed to get frame from camera after multiple attempts")
                    break
                    
                # Try to reinitialize camera
                cap.release()
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                continue
                
            # Reset retry count on successful frame
            retry_count = 0
            frame_counter += 1

            # Process frame
            processed_frame, features, prediction = detector.process_frame(frame)

            if features:
                # Update metrics
                status_placeholder.metric(
                    "Status",
                    "DROWSY" if prediction == 1 else "ALERT",
                    delta=None,
                    delta_color="inverse"
                )
                ear_placeholder.metric(
                    "Eye Aspect Ratio (EAR)",
                    f"{features['ear']:.3f}",
                    delta=None
                )
                perclos_placeholder.metric(
                    "PERCLOS",
                    f"{features['perclos']:.3f}",
                    delta=None
                )
                head_pose_placeholder.markdown(f"""
                **Head Pose**
                - Pitch: {features['pitch']:.1f}°
                - Yaw: {features['yaw']:.1f}°
                - Roll: {features['roll']:.1f}°
                """)

            # Display frame
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

            # Small delay to prevent overwhelming the CPU
            time.sleep(0.01)

    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            st.info("Camera and resources released successfully")
        except Exception as e:
            st.error(f"Error during cleanup: {str(e)}")
        
        if retry_count >= max_retries:
            st.error("""
            Camera access failed. Please try:
            1. Checking if another application is using the camera
            2. Selecting a different camera from the dropdown
            3. Disconnecting and reconnecting the camera
            4. Restarting the application
            """)
        elif frame_counter == 0:
            st.error("No frames were processed. Please check your camera connection.")

if __name__ == "__main__":
    main()
