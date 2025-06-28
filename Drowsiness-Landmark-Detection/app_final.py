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
import pygame
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drowsiness_detection_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize pygame for audio
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Use refined landmarks for better accuracy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Constants for drowsiness detection
        self.EAR_THRESHOLD = 0.21  # Slightly increased threshold for more reliable detection
        self.PERCLOS_THRESHOLD = 0.12  # Reduced to 12% for earlier warning
        self.DROWSY_TIME = 1.5  # Seconds of continuous eye closure for drowsiness
        self.BLINK_THRESHOLD = 0.25  # Seconds for a normal blink
        
        # Constants for normalization
        self.HEAD_POSE_LIMIT = 45.0  # Maximum head pose angle
        
        # Initialize metrics tracking
        self.fps = 30  # Assumed FPS
        self.history_size = int(self.fps * 3)  # 3 seconds history
        self.ear_history = deque(maxlen=self.history_size)
        self.closed_frames = 0
        
        # Drowsiness state tracking
        self.eyes_closed_start = None
        self.last_blink_end = None
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds
        self.current_state = "ALERT"
        self.state_confidence = 1.0
        self.drowsy_start = None
        
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
            pygame.mixer.music.set_volume(0.5)
        except Exception as e:
            logger.error(f"Could not load alert sound: {e}")

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate the Eye Aspect Ratio (EAR) for one eye"""
        try:
            # Convert landmarks to 2D points
            eye_points = []
            for idx in eye_indices:
                lm = landmarks[idx]
                eye_points.append(np.array([lm.x, lm.y]))
            eye_points = np.array(eye_points)
            
            # Calculate vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
            return np.clip(ear, 0.0, 1.0)  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return None

    def update_perclos(self, ear, timestamp):
        """Update PERCLOS calculation"""
        try:
            # Update EAR history and closed frames count
            self.ear_history.append(ear)
            
            if ear <= self.EAR_THRESHOLD:
                self.closed_frames += 1
                if self.eyes_closed_start is None:
                    self.eyes_closed_start = timestamp
            else:
                if self.eyes_closed_start is not None:
                    self.last_blink_end = timestamp
                    self.eyes_closed_start = None
            
            # Calculate PERCLOS based on frame history
            if not self.ear_history:
                return 0.0
            
            closed_count = sum(1 for ear in self.ear_history if ear <= self.EAR_THRESHOLD)
            return closed_count / len(self.ear_history)
            
        except Exception as e:
            logger.error(f"Error updating PERCLOS: {e}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error updating PERCLOS: {e}")
            return 0.0

    def calculate_head_pose(self, face_landmarks, frame_shape):
        """Calculate head pose angles using geometric approach"""
        try:
            h, w = frame_shape[:2]
            landmarks = face_landmarks.landmark
            face_3d = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in landmarks])
            
            # Get eye midpoints
            left_eye_mid = (face_3d[33] + face_3d[133]) / 2  # LEFT_EYE left and right corners
            right_eye_mid = (face_3d[362] + face_3d[263]) / 2  # RIGHT_EYE left and right corners
            
            # Get nose and chin points
            nose = face_3d[4]  # NOSE_TIP
            chin = face_3d[152]  # CHIN
            
            # Calculate angles
            eye_line = right_eye_mid - left_eye_mid
            yaw = np.arctan2(eye_line[2], eye_line[0])
            
            nose_chin = chin - nose
            # Adjusted pitch calculation to be more stable
            pitch = np.arctan2(-nose_chin[1], np.sqrt(nose_chin[0]**2 + nose_chin[2]**2))
            
            roll = np.arctan2(right_eye_mid[1] - left_eye_mid[1], 
                            right_eye_mid[0] - left_eye_mid[0])
            
            # Convert to degrees and normalize to reasonable ranges
            angles = np.degrees([pitch, yaw, roll])
            # More reasonable ranges for head pose
            angles[0] = np.clip(angles[0], -45, 45)  # pitch
            angles[1] = np.clip(angles[1], -45, 45)  # yaw
            angles[2] = np.clip(angles[2], -45, 45)  # roll
            return angles
            
        except Exception as e:
            logger.error(f"Error calculating head pose: {e}")
            return [0.0, 0.0, 0.0]
            
        except Exception as e:
            logger.error(f"Error calculating head pose: {e}")
            return (0.0, 0.0, 0.0)

    def _rotation_matrix_to_angles(self, rotation_matrix):
        """Convert rotation matrix to Euler angles"""
        try:
            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] +
                        rotation_matrix[1, 0] * rotation_matrix[1, 0])
            
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
                yaw = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
            else:
                pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]) * 180 / np.pi
                yaw = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
                roll = 0

            return [pitch, yaw, roll]
            
        except Exception as e:
            logger.error(f"Error converting rotation matrix to angles: {e}")
            return [0.0, 0.0, 0.0]

    def process_frame(self, frame):
        """Process a single frame and return the results"""
        if frame is None:
            return frame, None, 0, 0.0
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            features = None
            prediction = 0  # Default to alert state
            confidence = 1.0
            
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
                
                if left_ear is None or right_ear is None:
                    return frame, None, 0, 0.0
                    
                ear = (left_ear + right_ear) / 2
                self.ear_history.append(ear)
                
                # Update drowsiness detection
                current_time = time.time()
                
                # Update PERCLOS
                perclos = self.update_perclos(ear, current_time)
                
                # Get head pose
                pitch, yaw, roll = self.calculate_head_pose(face_landmarks, frame.shape)
                
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
                
                # Determine drowsiness state
                drowsy_confidence = 0.0
                
                # Check immediate drowsiness indicators
                if ear < self.EAR_THRESHOLD:
                    if self.eyes_closed_start is not None:
                        eyes_closed_time = current_time - self.eyes_closed_start
                        if eyes_closed_time >= self.DROWSY_TIME:
                            drowsy_confidence = min(1.0, eyes_closed_time / (self.DROWSY_TIME * 1.5))
                            prediction = 1
                
                # Weight different factors for alertness/drowsiness detection
                weights = {
                    'eye_closure': 0.3,   # Immediate eye closure
                    'perclos': 0.3,       # PERCLOS
                    'head_pose': 0.3,     # Head pose (increased weight)
                    'model': 0.1          # ML model
                }
                
                # Calculate head pose confidence
                pitch, yaw, roll = features['pitch'], features['yaw'], features['roll']
                
                # Define thresholds and weights for each angle
                pose_params = {
                    'pitch': {'warn': 25, 'critical': 45, 'weight': 0.4},
                    'yaw': {'warn': 25, 'critical': 45, 'weight': 0.3},
                    'roll': {'warn': 25, 'critical': 45, 'weight': 0.3}
                }
                
                # Calculate dynamic confidence for each angle
                angle_confidences = {}
                pose_warning_count = 0
                
                for angle, value in [('pitch', abs(pitch)), ('yaw', abs(yaw)), ('roll', abs(roll))]:
                    params = pose_params[angle]
                    abs_value = abs(value)
                    
                    if abs_value <= params['warn']:
                        # Normal range: linear reduction from 1.0 to 0.8
                        angle_confidences[angle] = 1.0 - (0.2 * abs_value / params['warn'])
                    elif abs_value <= params['critical']:
                        # Warning range: linear reduction from 0.8 to 0.4
                        warning_fraction = (abs_value - params['warn']) / (params['critical'] - params['warn'])
                        angle_confidences[angle] = 0.8 - (0.4 * warning_fraction)
                        pose_warning_count += 1
                    else:
                        # Critical range: 0.4 or lower
                        over_critical = (abs_value - params['critical']) / params['critical']
                        angle_confidences[angle] = max(0.2, 0.4 - (0.2 * over_critical))
                        pose_warning_count += 2
                
                # Calculate weighted head pose confidence
                head_pose_confidence = sum(
                    conf * pose_params[angle]['weight']
                    for angle, conf in angle_confidences.items()
                )
                
                # More gradual reduction for multiple warnings
                if pose_warning_count >= 2:
                    # Reduce by 5% for each warning, but maintain minimum of 40%
                    reduction_factor = min(0.6, 0.05 * pose_warning_count)
                    head_pose_confidence = max(0.4, head_pose_confidence * (1.0 - reduction_factor))
                
                # Check PERCLOS
                perclos_confidence = 0.0
                if perclos > self.PERCLOS_THRESHOLD:
                    perclos_confidence = min(1.0, perclos / (self.PERCLOS_THRESHOLD * 1.5))
                    if perclos_confidence > 0.8:
                        prediction = 1
                
                # Use ML model if available
                model_confidence = 0.0
                if self.model is not None and self.scaler is not None:
                    feature_vector = np.array([[
                        features['ear'],
                        features['left_ear'],
                        features['right_ear'],
                        features['pitch'],
                        features['yaw'],
                        features['roll'],
                        features['perclos']
                    ]])
                    
                    # Scale features
                    scaled_features = self.scaler.transform(feature_vector)
                    
                    # Get prediction and probability
                    model_pred = self.model.predict(scaled_features)[0]
                    proba = self.model.predict_proba(scaled_features)[0]
                    model_confidence = proba[1]  # Confidence for drowsy class
                    
                    if model_confidence > 0.9:
                        prediction = 1
                
                # Calculate alert confidence with stronger head pose influence
                alert_confidence = min(
                    1.0,
                    weights['eye_closure'] * (1.0 - drowsy_confidence) +
                    weights['perclos'] * (1.0 - perclos_confidence) +
                    weights['head_pose'] * head_pose_confidence +
                    weights['model'] * (1.0 - model_confidence)
                )
                  # Calculate overall pose severity (0 to 1)
                overall_pose_severity = max(
                    abs(pitch) / pose_params['pitch']['critical'],
                    abs(yaw) / pose_params['yaw']['critical'],
                    abs(roll) / pose_params['roll']['critical']
                )
                
                # 1. Calculate base confidence from head pose
                base_pose_confidence = 1.0
                
                # Apply penalties for each angle exceeding thresholds
                for angle, value in [('pitch', abs(pitch)), ('yaw', abs(yaw)), ('roll', abs(roll))]:
                    if value > pose_params[angle]['critical']:
                        # Beyond critical: heavy penalty
                        penalty = 0.3 * (value / pose_params[angle]['critical'])
                        base_pose_confidence -= min(0.3, penalty)
                    elif value > pose_params[angle]['warn']:
                        # Warning zone: medium penalty
                        warn_fraction = (value - pose_params[angle]['warn']) / (pose_params[angle]['critical'] - pose_params[angle]['warn'])
                        base_pose_confidence -= min(0.2, 0.2 * warn_fraction)
                
                # 2. Calculate normalized metrics
                current_time = time.time()
                
                # EAR normalization with full range
                ear_factor = 0.0
                if ear >= self.EAR_THRESHOLD + 0.1:  # Fully alert
                    ear_factor = 1.0
                elif ear <= self.EAR_THRESHOLD - 0.05:  # Very drowsy
                    ear_factor = 0.1
                else:  # Linear interpolation in between
                    ear_factor = 0.1 + 0.9 * (ear - (self.EAR_THRESHOLD - 0.05)) / 0.15
                
                # PERCLOS with full range
                perclos_factor = 0.0
                if perclos <= self.PERCLOS_THRESHOLD * 0.5:  # Good state
                    perclos_factor = 1.0
                elif perclos >= self.PERCLOS_THRESHOLD * 2.0:  # Very drowsy
                    perclos_factor = 0.1
                else:  # Linear decrease
                    perclos_factor = 1.0 - 0.9 * (perclos - self.PERCLOS_THRESHOLD * 0.5) / (self.PERCLOS_THRESHOLD * 1.5)
                
                # Head pose confidence with full range
                pose_factor = 0.0
                if base_pose_confidence >= 0.8:  # Good pose
                    pose_factor = 1.0
                elif base_pose_confidence <= 0.3:  # Bad pose
                    pose_factor = 0.1
                else:  # Linear interpolation
                    pose_factor = 0.1 + 0.9 * (base_pose_confidence - 0.3) / 0.5
                
                # 3. Calculate weighted alert confidence with adjusted weights
                alert_weights = {'ear': 0.45, 'perclos': 0.3, 'pose': 0.25}
                alert_confidence = (
                    ear_factor * alert_weights['ear'] +
                    perclos_factor * alert_weights['perclos'] +
                    pose_factor * alert_weights['pose']
                )
                
                # 4. Determine state and final confidence
                if ear < self.EAR_THRESHOLD and self.eyes_closed_start:
                    eyes_closed_time = current_time - self.eyes_closed_start
                    if eyes_closed_time >= self.DROWSY_TIME:
                        prediction = 1
                        # More aggressive confidence drop for prolonged eye closure
                        confidence = min(0.3, max(0.1, 0.3 - (eyes_closed_time - self.DROWSY_TIME) * 0.1))
                elif perclos > self.PERCLOS_THRESHOLD:
                    prediction = 1
                    # Stronger influence of PERCLOS
                    confidence = max(0.1, min(0.5, 0.7 - perclos))
                elif pose_factor < 0.5:
                    prediction = 1
                    # More noticeable confidence drop for poor pose
                    confidence = max(0.1, min(0.6, pose_factor))
                else:
                    prediction = 0
                    # Alert confidence can reach 100% when all metrics are perfect
                    confidence = max(0.1, min(1.0, alert_confidence))
                
                # Update state and handle alerts
                if prediction == 1:
                    # Update drowsy state timing
                    if self.drowsy_start is None:
                        self.drowsy_start = current_time
                    
                    # Handle alert sound with cooldown
                    if (current_time - self.last_alert_time) > self.alert_cooldown:
                        if not pygame.mixer.music.get_busy():
                            pygame.mixer.music.play()
                        self.last_alert_time = current_time
                else:
                    # Reset state for alert condition
                    self.drowsy_start = None
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                
                # Draw status on frame
                status_text = f"{'DROWSY' if prediction == 1 else 'ALERT'} ({confidence*100:.0f}%)"
                color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Draw EAR and PERCLOS
                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            return frame, features, prediction, confidence
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, None, 0, 0.0

def main():
    st.set_page_config(page_title="Drowsiness Detection v4", layout="wide")
    
    # Title and description
    st.title("Real-time Drowsiness Detection")
    st.markdown("""
    Advanced monitoring system using:
    - Eye Aspect Ratio (EAR)
    - PERCLOS (Percentage of eye closure)
    - Head pose (Pitch, Yaw, Roll)
    - ML-based drowsiness prediction
    """)
    
    # Create detector instance
    detector = DrowsinessDetector()
    
    # Initialize metrics state
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'ear': 0.0,
            'perclos': 0.0,
            'prediction': 0,
            'confidence': 0.0,
            'head_pose': (0.0, 0.0, 0.0),
            'start_time': datetime.now(),
            'alert_count': 0,
            'total_drowsy_time': 0
        }

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Camera selection
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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

        # Video feed placeholder
        video_placeholder = st.empty()

    with col2:
        # Status indicators
        status_container = st.container()
        with status_container:
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            session_info_placeholder = st.empty()
        
        # Thresholds display
        st.markdown("### Detection Thresholds")
        st.write(f"EAR Threshold: {detector.EAR_THRESHOLD:.2f}")
        st.write(f"PERCLOS Threshold: {detector.PERCLOS_THRESHOLD:.2f}")
        st.write(f"Drowsy Time Threshold: {detector.DROWSY_TIME:.1f}s")
        
        # Volume control
        st.markdown("### Alert Settings")
        volume = st.slider("Alert Volume", 0.0, 1.0, 0.5, 0.1)
        pygame.mixer.music.set_volume(volume)

    # Stop button
    stop_button = st.button("Stop")

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        st.error("Error: Could not open camera!")
        return

    try:
        last_alert_time = time.time()
        frame_count = 0
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get frame from camera")
                break

            # Process frame
            processed_frame, features, prediction, confidence = detector.process_frame(frame)
            frame_count += 1

            # Update session metrics
            if features:
                st.session_state.metrics['ear'] = features['ear']
                st.session_state.metrics['perclos'] = features['perclos']
                st.session_state.metrics['prediction'] = prediction
                st.session_state.metrics['confidence'] = confidence
                st.session_state.metrics['head_pose'] = (
                    features['pitch'],
                    features['yaw'],
                    features['roll']
                )
                
                if prediction == 1:
                    if time.time() - last_alert_time > 3.0:  # New alert after 3 seconds
                        st.session_state.metrics['alert_count'] += 1
                        last_alert_time = time.time()
                    st.session_state.metrics['total_drowsy_time'] += 1/30  # Assuming 30 fps

            # Update status display
            status_color = "#FF0000" if prediction == 1 else "#00FF00"
            status_placeholder.markdown(
                f"""
                <div style="padding:20px;border-radius:10px;background-color:{status_color}20">
                    <h1 style="color:{status_color};text-align:center;margin:0">
                        {"DROWSY" if prediction == 1 else "ALERT"}
                    </h1>
                    <p style="text-align:center;margin:5px 0 0 0">
                        Confidence: {confidence*100:.1f}%
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Update metrics
            metrics_placeholder.markdown(
                f"""
                ### Current Metrics
                - **EAR:** {st.session_state.metrics['ear']:.3f}
                - **PERCLOS:** {st.session_state.metrics['perclos']:.3f}
                - **Head Pose:**
                  - Pitch: {st.session_state.metrics['head_pose'][0]:.1f}° {'⚠️' if abs(st.session_state.metrics['head_pose'][0]) > 45 else ''}
                  - Yaw: {st.session_state.metrics['head_pose'][1]:.1f}° {'⚠️' if abs(st.session_state.metrics['head_pose'][1]) > 35 else ''}
                  - Roll: {st.session_state.metrics['head_pose'][2]:.1f}° {'⚠️' if abs(st.session_state.metrics['head_pose'][2]) > 35 else ''}
                """
            )

            # Update session information
            session_duration = datetime.now() - st.session_state.metrics['start_time']
            session_info_placeholder.markdown(
                f"""
                ### Session Information
                - **Duration:** {str(session_duration).split('.')[0]}
                - **Drowsy Events:** {st.session_state.metrics['alert_count']}
                - **Total Drowsy Time:** {timedelta(seconds=int(st.session_state.metrics['total_drowsy_time']))}
                - **Average FPS:** {frame_count/session_duration.total_seconds():.1f}
                """
            )

            # Display frame
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)

            # Small delay to prevent overwhelming the CPU
            time.sleep(0.001)

    finally:
        cap.release()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        st.success("Session ended. Camera released.")

if __name__ == "__main__":
    main()
