import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import queue
from collections import deque
import threading
import time
import playsound
import logging
from datetime import datetime
from typing import List, NamedTuple
import json

# Setup logging with file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drowsiness_detection_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for facial landmarks
FACE_LANDMARKS = {
    'LEFT_EYE': [362, 385, 387, 263, 373, 380],
    'RIGHT_EYE': [33, 160, 158, 133, 153, 144],
    'NOSE_TIP': 1,
    'CHIN': 152,
    'LEFT_EYE_CORNER': 226,
    'RIGHT_EYE_CORNER': 446,
    'LEFT_MOUTH': 57,
    'RIGHT_MOUTH': 287
}

# Feature names for the model (must match training data)
FEATURE_NAMES = [
    'ear', 'left_ear', 'right_ear', 'pitch', 'yaw', 'roll', 'perclos'
]

# Global state management
if 'global_state' not in st.session_state:
    st.session_state.global_state = {
        'current_ear': 0.0,
        'current_perclos': 0.0,
        'current_prediction': 0,
        'prediction_confidence': 0.0,
        'head_pose': (0.0, 0.0, 0.0),
        'face_detected': False,
        'alert_volume': 0.7,
        'emergency_contact': "",
        'emergency_phone': "",
        'lock': threading.Lock()
    }

# Load the ML model and scaler
try:
    model_path = Path('model/best_model.pkl')
    scaler_path = Path('model/scaler.pkl')
    
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler file not found")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

class Alert:
    def __init__(self, sound_path="alert.wav"):
        self.sound_path = Path(sound_path)
        if not self.sound_path.exists():
            logger.error(f"Alert sound file not found: {sound_path}")
            self.sound_path = None
        self.alert_thread = None
        self.stop_alert = False

    def play_alert(self):
        while not self.stop_alert:
            try:
                if self.sound_path:
                    playsound.playsound(str(self.sound_path))
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error playing alert: {e}")
                break

    def start(self):
        self.stop_alert = False
        if self.alert_thread is None or not self.alert_thread.is_alive():
            self.alert_thread = threading.Thread(target=self.play_alert)
            self.alert_thread.daemon = True
            self.alert_thread.start()

    def stop(self):
        self.stop_alert = True
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        # Initialize metrics tracking
        self.ear_history = deque(maxlen=30)  # 1 second at 30 fps
        self.perclos_window = deque(maxlen=300)  # 10 seconds at 30 fps
        self.metrics_log = deque(maxlen=1000)  # Store last 1000 measurements
        self.last_save_time = time.time()
        self.save_interval = 60  # Save metrics every 60 seconds
        
        # Initialize thresholds
        self.EAR_THRESHOLD = 0.2
        self.PERCLOS_THRESHOLD = 0.2
        self.HEAD_POSE_THRESHOLD = 30.0  # degrees
        
        # Initialize alert system
        self.alert = Alert()
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds

    def calculate_ear(self, landmarks, eye_indices):
        """Calculate the Eye Aspect Ratio (EAR) for one eye"""
        try:
            p1 = landmarks[eye_indices[1]]
            p2 = landmarks[eye_indices[2]]
            p3 = landmarks[eye_indices[3]]
            p4 = landmarks[eye_indices[4]]
            p5 = landmarks[eye_indices[5]]
            p6 = landmarks[eye_indices[0]]

            # Calculate Euclidean distances
            ear = (self._calculate_distance(p2, p6) + self._calculate_distance(p3, p5)) / \
                  (2 * self._calculate_distance(p1, p4))
            return max(0.0, min(1.0, ear))  # Clamp between 0 and 1
        except Exception as e:
            logger.error(f"Error calculating EAR: {e}")
            return None

    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _save_metrics(self):
        """Save accumulated metrics to a JSON file"""
        try:
            current_time = time.time()
            if current_time - self.last_save_time >= self.save_interval and self.metrics_log:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_v2_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(list(self.metrics_log), f)
                
                logger.info(f"Saved metrics to {filename}")
                self.metrics_log.clear()
                self.last_save_time = current_time
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _calculate_euler_angles(self, rotation_matrix):
        """Calculate Euler angles (pitch, yaw, roll) from rotation matrix"""
        try:
            # Pitch (x-axis rotation)
            pitch = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )
            
            # Yaw (y-axis rotation)
            yaw = np.arctan2(
                -rotation_matrix[2, 0],
                np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2)
            )
            
            # Roll (z-axis rotation)
            roll = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )
            
            # Convert radians to degrees
            pitch = np.degrees(pitch)
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)
            
            return pitch, yaw, roll
        except Exception as e:
            logger.error(f"Error calculating Euler angles: {e}")
            return 0.0, 0.0, 0.0

    def _extract_features(self, frame, face_landmarks):
        """Extract all features from the face landmarks"""
        try:
            if not face_landmarks:
                return None

            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(face_landmarks.landmark, FACE_LANDMARKS['LEFT_EYE'])
            right_ear = self.calculate_ear(face_landmarks.landmark, FACE_LANDMARKS['RIGHT_EYE'])
            
            if left_ear is None or right_ear is None:
                return None
                
            ear = (left_ear + right_ear) / 2
            self.ear_history.append(ear)

            # Calculate head pose
            size = frame.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            dist_coeffs = np.zeros((4,1), dtype=np.float32)

            # 3D model points (normalized)
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),   # Right eye corner
                (-150.0, -150.0, -125.0), # Left mouth corner
                (150.0, -150.0, -125.0)   # Right mouth corner
            ], dtype=np.float32) / 4.5

            # Get corresponding 2D points
            image_points = np.array([
                [face_landmarks.landmark[FACE_LANDMARKS['NOSE_TIP']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['NOSE_TIP']].y * size[0]],
                [face_landmarks.landmark[FACE_LANDMARKS['CHIN']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['CHIN']].y * size[0]],
                [face_landmarks.landmark[FACE_LANDMARKS['LEFT_EYE_CORNER']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['LEFT_EYE_CORNER']].y * size[0]],
                [face_landmarks.landmark[FACE_LANDMARKS['RIGHT_EYE_CORNER']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['RIGHT_EYE_CORNER']].y * size[0]],
                [face_landmarks.landmark[FACE_LANDMARKS['LEFT_MOUTH']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['LEFT_MOUTH']].y * size[0]],
                [face_landmarks.landmark[FACE_LANDMARKS['RIGHT_MOUTH']].x * size[1],
                 face_landmarks.landmark[FACE_LANDMARKS['RIGHT_MOUTH']].y * size[0]]
            ], dtype=np.float32)

            # Ensure image_points has the right shape (Nx2)
            image_points = image_points.reshape(-1, 2)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Calculate Euler angles directly
            pitch, yaw, roll = self._calculate_euler_angles(rotation_mat)

            # Calculate PERCLOS
            if ear < self.EAR_THRESHOLD:
                self.perclos_window.append(1)
            else:
                self.perclos_window.append(0)
            
            perclos = sum(self.perclos_window) / len(self.perclos_window) if self.perclos_window else 0

            # Create feature vector
            features = {
                'ear': float(ear),
                'left_ear': float(left_ear),
                'right_ear': float(right_ear),
                'pitch': float(pitch),
                'yaw': float(yaw),
                'roll': float(roll),
                'perclos': float(perclos),
                'timestamp': time.time()
            }

            # Log metrics
            self.metrics_log.append(features)
            self._save_metrics()

            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process each frame from the video stream"""
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Reduce resolution for processing
            scale_factor = 0.5
            small_frame = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Scale landmarks back to original size
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        landmark.x /= scale_factor
                        landmark.y /= scale_factor
                
                face_landmarks = results.multi_face_landmarks[0]
                features = self._extract_features(img, face_landmarks)
                
                if features and model is not None and scaler is not None:
                    try:
                        # Prepare features in the correct order
                        feature_vector = pd.DataFrame([{
                            'ear': features['ear'],
                            'left_ear': features['left_ear'],
                            'right_ear': features['right_ear'],
                            'pitch': features['pitch'],
                            'yaw': features['yaw'],
                            'roll': features['roll'],
                            'perclos': features['perclos']
                        }])[FEATURE_NAMES]  # Ensure correct feature order
                        
                        # Scale features and predict
                        scaled_features = scaler.transform(feature_vector)
                        prediction = model.predict(scaled_features)[0]
                        prediction_proba = model.predict_proba(scaled_features)[0]
                        
                        # Update global state
                        st.session_state.global_state['current_ear'] = float(features['ear'])
                        st.session_state.global_state['current_perclos'] = float(features['perclos'])
                        st.session_state.global_state['current_prediction'] = int(prediction)
                        st.session_state.global_state['prediction_confidence'] = float(prediction_proba.max())
                        st.session_state.global_state['head_pose'] = (
                            float(features['pitch']),
                            float(features['yaw']),
                            float(features['roll'])
                        )
                        st.session_state.global_state['face_detected'] = True
                        
                        # Draw face mesh
                        mp.solutions.drawing_utils.draw_landmarks(
                            image=img,
                            landmark_list=face_landmarks,
                            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        
                        # Add prediction text with confidence
                        status_text = f"{'DROWSY' if prediction == 1 else 'ALERT'} ({prediction_proba.max()*100:.1f}%)"
                        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
                        cv2.putText(
                            img,
                            status_text,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2
                        )
                        
                        # Handle alerts
                        if prediction == 1 and prediction_proba.max() > 0.8:
                            current_time = time.time()
                            if current_time - self.last_alert_time > self.alert_cooldown:
                                self.alert.start()
                                self.last_alert_time = current_time
                        else:
                            self.alert.stop()
                    except Exception as e:
                        logger.error(f"Error in prediction step: {e}")
            else:
                # No face detected
                st.session_state.global_state['face_detected'] = False
                st.session_state.global_state['current_ear'] = 0.0
                st.session_state.global_state['current_perclos'] = 0.0
                st.session_state.global_state['current_prediction'] = 0
                st.session_state.global_state['prediction_confidence'] = 0.0
                st.session_state.global_state['head_pose'] = (0.0, 0.0, 0.0)
                
                cv2.putText(
                    img,
                    "No Face Detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Set page config
    st.set_page_config(
        page_title="Drowsiness Detection v2",
        page_icon="👁️",
        layout="wide"
    )
    
    # Reduce logging noise
    logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

    # Application header with custom styling
    st.markdown("""
        <style>
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 24px;
            color: #424242;
            margin-bottom: 30px;
        }
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="title">Real-time Drowsiness Detection v2</p>', unsafe_allow_html=True)
    st.markdown("""
        <p class="subtitle">Advanced monitoring system using computer vision and machine learning</p>
        """, unsafe_allow_html=True)

    # Create three columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("Live Feed")
        
        # Camera selection
        try:
            camera_list = [
                index for index in range(10)
                if cv2.VideoCapture(index).read()[0]
            ]
            if camera_list:
                selected_camera = st.selectbox(
                    "Select Camera",
                    camera_list,
                    format_func=lambda x: f"Camera {x}"
                )
            else:
                selected_camera = 0
        except Exception as e:
            logger.error(f"Error listing cameras: {e}")
            selected_camera = 0
        
        # WebRTC Configuration
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        
        # Start video stream
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection-v2",
            video_processor_factory=DrowsinessDetector,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {
                    "deviceId": {"exact": str(selected_camera)},
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )

    with col2:
        st.subheader("Real-time Metrics")
        
        # Status indicator
        status_placeholder = st.empty()
        
        # Create metric cards with custom styling
        with st.container():
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            ear_metric = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            perclos_metric = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            head_pose_container = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Volume control
        st.subheader("Alert Settings")
        new_volume = st.slider(
            "Alert Volume",
            0.0,
            1.0,
            st.session_state.global_state['alert_volume'],
            0.1,
            help="Adjust the volume of the drowsiness alert"
        )
        if new_volume != st.session_state.global_state['alert_volume']:
            st.session_state.global_state['alert_volume'] = new_volume
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.alert.set_volume(new_volume)

    with col3:
        st.subheader("Emergency Contact")
        new_contact = st.text_input(
            "Contact Name",
            value=st.session_state.global_state['emergency_contact'],
            help="Name of emergency contact"
        )
        new_phone = st.text_input(
            "Contact Phone",
            value=st.session_state.global_state['emergency_phone'],
            help="Phone number of emergency contact"
        )
        if new_contact != st.session_state.global_state['emergency_contact']:
            st.session_state.global_state['emergency_contact'] = new_contact
        if new_phone != st.session_state.global_state['emergency_phone']:
            st.session_state.global_state['emergency_phone'] = new_phone

        # Metrics download section
        st.subheader("Session Metrics")
        metrics_files = list(Path('.').glob('metrics_v2_*.json'))
        if metrics_files:
            metrics_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            selected_file = st.selectbox(
                "Select Metrics File",
                metrics_files,
                format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})"
            )
            
            if st.button("Download Metrics"):
                try:
                    with open(selected_file, 'r') as f:
                        st.download_button(
                            "Download JSON",
                            f.read(),
                            file_name=selected_file.name,
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"Error reading metrics file: {e}")

    # Update metrics in real-time
    if webrtc_ctx.state.playing:
        while True:
            # Update status with confidence
            confidence = st.session_state.global_state['prediction_confidence'] * 100
            status_color = "#FF0000" if st.session_state.global_state['current_prediction'] == 1 else "#00FF00"
            status_placeholder.markdown(
                f"""
                <div style="padding: 15px; background-color: {status_color}33; border-radius: 10px; text-align: center;">
                    <h2 style="color: {status_color}; margin: 0;">
                        {"DROWSY" if st.session_state.global_state['current_prediction'] == 1 else "ALERT"}
                    </h2>
                    <p style="margin: 5px 0 0 0;">Confidence: {confidence:.1f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Update EAR with trend
            ear_metric.metric(
                "Eye Aspect Ratio (EAR)",
                f"{st.session_state.global_state['current_ear']:.3f}",
                delta=None
            )

            # Update PERCLOS with warning
            perclos_warning = st.session_state.global_state['current_perclos'] > 0.15
            perclos_metric.metric(
                "PERCLOS",
                f"{st.session_state.global_state['current_perclos']:.3f}",
                delta="Warning" if perclos_warning else None,
                delta_color="inverse"
            )

            # Update head pose with visual indicators
            pitch, yaw, roll = st.session_state.global_state['head_pose']
            head_pose_container.markdown(f"""
            **Head Pose**
            - Pitch: {pitch:.1f}° {'⚠️' if abs(pitch) > 30 else ''}
            - Yaw: {yaw:.1f}° {'⚠️' if abs(yaw) > 30 else ''}
            - Roll: {roll:.1f}° {'⚠️' if abs(roll) > 30 else ''}
            """)

            time.sleep(0.1)

    # Add information about the ML model
    st.sidebar.title("Model Information")
    st.sidebar.info("""
    Using XGBoost classifier trained on:
    - 7,971 alert samples
    - 4,454 drowsy samples
    
    Model Performance:
    - Accuracy: 99%
    - Precision: 98-99%
    - Recall: 99%
    """)

    # Instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Select your camera from the dropdown
    2. Click 'Start' to begin video stream
    3. Position yourself in front of the camera
    4. Ensure good lighting conditions
    5. Watch the metrics panel for:
        - Current drowsiness status
        - Eye aspect ratio
        - PERCLOS value
        - Head position
    6. Adjust alert volume as needed
    7. Add emergency contact details
    8. Monitor saved metrics for analysis
    """)

    # Troubleshooting section
    st.sidebar.title("Troubleshooting")
    st.sidebar.markdown("""
    If you experience issues:
    1. Check your camera connection
    2. Ensure good lighting
    3. Position yourself 0.5-1m from camera
    4. Clear browser cache if video is laggy
    5. Check logs for technical issues
    """)

if __name__ == "__main__":
    main()
