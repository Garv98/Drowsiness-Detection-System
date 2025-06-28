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
from typing import List, NamedTuple
import time
from collections import deque
import threading
import playsound
import logging
from datetime import datetime
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('drowsiness_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
LANDMARK_POINTS_68 = list(range(468))
EYES_INDICES_68 = [
    # Left eye
    [362, 385, 387, 263, 373, 380],
    # Right eye
    [33, 160, 158, 133, 153, 144],
]
MOUTH_INDICES = [78, 81, 13, 311, 308, 402, 14, 178]
FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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

class FaceMeshResult(NamedTuple):
    multi_face_landmarks: List
    frame: np.ndarray
    success: bool

class Alert:
    def __init__(self, sound_path="alert.wav"):
        self.sound_path = Path(sound_path)
        if not self.sound_path.exists():
            logger.error(f"Alert sound file not found: {sound_path}")
            self.sound_path = None
        self.alert_thread = None
        self.stop_alert = False
        self.volume = 1.0  # Default volume
        
        # Initialize audio devices
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_control = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:
            logger.error(f"Error initializing audio control: {e}")
            self.volume_control = None

    def set_volume(self, volume: float):
        """Set alert volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        if self.volume_control:
            try:
                # Convert to dB range (-65.25 to 0.0)
                db = -65.25 * (1.0 - self.volume)
                self.volume_control.SetMasterVolumeLevel(db, None)
            except Exception as e:
                logger.error(f"Error setting volume: {e}")

    def play_alert(self):
        while not self.stop_alert and self.sound_path:
            try:
                playsound.playsound(str(self.sound_path))
                time.sleep(0.5)  # Reduced sleep time for more responsive alerts
            except Exception as e:
                logger.error(f"Error playing alert sound: {e}")
                break

    def start(self):
        self.stop_alert = False
        if self.sound_path and (self.alert_thread is None or not self.alert_thread.is_alive()):
            self.alert_thread = threading.Thread(target=self.play_alert)
            self.alert_thread.daemon = True  # Ensure thread stops when main program exits
            self.alert_thread.start()

    def stop(self):
        self.stop_alert = True
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=1.0)  # Wait up to 1 second for thread to stop

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.process_every_n_frames = 2
        self.frame_counter = 0
        self.last_features = None
        self.ear_history = deque(maxlen=30)  # 1 second at 30 fps
        self.perclos_window = deque(maxlen=300)  # 10 seconds at 30 fps
        self.alert = Alert()
        self.last_alert_time = 0
        self.alert_cooldown = 3  # seconds
        self.metrics_log = deque(maxlen=1000)  # Store last 1000 measurements
        self.last_save_time = time.time()
        self.save_interval = 60  # Save metrics every 60 seconds

    def _calculate_ear(self, landmarks, eye_indices):
        """Calculate the Eye Aspect Ratio (EAR) for one eye"""
        try:
            p1 = landmarks[eye_indices[1]]
            p2 = landmarks[eye_indices[2]]
            p3 = landmarks[eye_indices[3]]
            p4 = landmarks[eye_indices[4]]
            p5 = landmarks[eye_indices[5]]
            p6 = landmarks[eye_indices[0]]

            ear = (self._calculate_distance(p2, p6) + self._calculate_distance(p3, p5)) / (2 * self._calculate_distance(p1, p4))
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
                filename = f"metrics_{timestamp}.json"
                
                with open(filename, 'w') as f:
                    json.dump(list(self.metrics_log), f)
                
                logger.info(f"Saved metrics to {filename}")
                self.metrics_log.clear()
                self.last_save_time = current_time
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def _extract_features(self, frame, face_landmarks):
        if not face_landmarks:
            return None

        try:
            # Convert landmarks to numpy array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # Calculate EAR
            left_ear = self._calculate_ear(face_landmarks.landmark, EYES_INDICES_68[0])
            right_ear = self._calculate_ear(face_landmarks.landmark, EYES_INDICES_68[1])
            
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
            
            dist_coeffs = np.zeros((4,1))

            # 3D model points (normalized)
            model_points = np.array([
                (0.0, 0.0, 0.0),          # Nose tip
                (0.0, -330.0, -65.0),     # Chin
                (-225.0, 170.0, -135.0),  # Left eye corner
                (225.0, 170.0, -135.0),   # Right eye corner
                (-150.0, -150.0, -125.0), # Left mouth corner
                (150.0, -150.0, -125.0)   # Right mouth corner
            ]) / 4.5

            # Get corresponding 2D points
            image_points = np.array([
                (landmarks[1, :2]),     # Nose tip
                (landmarks[152, :2]),   # Chin
                (landmarks[226, :2]),   # Left eye corner
                (landmarks[446, :2]),   # Right eye corner
                (landmarks[57, :2]),    # Left mouth corner
                (landmarks[287, :2])    # Right mouth corner
            ], dtype=np.float32)
            
            image_points = np.multiply(image_points, [size[1], size[0]])

            success, rotation_vec, translation_vec = cv2.solvePnP(
                model_points, 
                image_points,
                camera_matrix, 
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                np.hstack((pose_mat, [[0], [0], [0], [1]]))
            )
            
            pitch, yaw, roll = [float(angle) for angle in euler_angles]

            # Calculate PERCLOS
            if ear < 0.2:  # Threshold for closed eyes
                self.perclos_window.append(1)
            else:
                self.perclos_window.append(0)
            
            perclos = sum(self.perclos_window) / len(self.perclos_window) if self.perclos_window else 0

            # Create feature vector
            features = {
                'ear': ear,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'perclos': perclos,
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
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Reduce resolution for processing
            scale_factor = 0.5
            small_frame = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
            
            # Only process every nth frame
            self.frame_counter += 1
            process_this_frame = self.frame_counter % self.process_every_n_frames == 0
            
            if process_this_frame:
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
                
                    for face_landmarks in results.multi_face_landmarks:
                        features = self._extract_features(img, face_landmarks)
                        
                        if features and model is not None and scaler is not None:
                            # Prepare features for prediction
                            feature_vector = np.array([
                                [
                                    features['ear'],
                                    features['left_ear'],
                                    features['right_ear'],
                                    features['pitch'],
                                    features['yaw'],
                                    features['roll'],
                                    features['perclos']
                                ]
                            ])
                            
                            # Scale features
                            scaled_features = scaler.transform(feature_vector)
                            
                            # Make prediction
                            try:
                                prediction = model.predict(scaled_features)[0]
                                prediction_proba = model.predict_proba(scaled_features)[0]
                                
                                # Update UI elements
                                with st.session_state.lock:
                                    st.session_state.current_ear = features['ear']
                                    st.session_state.current_perclos = features['perclos']
                                    st.session_state.current_prediction = prediction
                                    st.session_state.prediction_confidence = prediction_proba.max()
                                    st.session_state.head_pose = (
                                        features['pitch'],
                                        features['yaw'],
                                        features['roll']
                                    )
                                
                                # Draw landmarks and status
                                mp.solutions.drawing_utils.draw_landmarks(
                                    image=img,
                                    landmark_list=face_landmarks,
                                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                                )
                                
                                # Add prediction text with confidence
                                status_text = f"{'DROWSY' if prediction == 1 else 'ALERT'} ({prediction_proba.max()*100:.1f}%)"
                                cv2.putText(
                                    img,
                                    status_text,
                                    (20, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255) if prediction == 1 else (0, 255, 0),
                                    2
                                )
                                
                                # Handle alerts
                                if prediction == 1 and prediction_proba.max() > 0.8:  # Only alert if confidence > 80%
                                    current_time = time.time()
                                    if current_time - self.last_alert_time > self.alert_cooldown:
                                        self.alert.start()
                                        self.last_alert_time = current_time
                                else:
                                    self.alert.stop()
                                    
                            except Exception as e:
                                logger.error(f"Error in prediction: {e}")
                else:
                    # No face detected
                    with st.session_state.lock:
                        st.session_state.face_detected = False
                    
                    # Draw "No Face Detected" text
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
            logger.error(f"Error in frame processing: {e}")
            return frame

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    # Set page config
    st.set_page_config(
        page_title="Drowsiness Detection System",
        page_icon="👁️",
        layout="wide"
    )
    
    # Reduce logging noise
    logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

    # Initialize session state
    if 'lock' not in st.session_state:
        st.session_state.lock = threading.Lock()
    if 'current_ear' not in st.session_state:
        st.session_state.current_ear = 0.0
    if 'current_perclos' not in st.session_state:
        st.session_state.current_perclos = 0.0
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = 0
    if 'prediction_confidence' not in st.session_state:
        st.session_state.prediction_confidence = 0.0
    if 'head_pose' not in st.session_state:
        st.session_state.head_pose = (0.0, 0.0, 0.0)
    if 'face_detected' not in st.session_state:
        st.session_state.face_detected = False
    if 'alert_volume' not in st.session_state:
        st.session_state.alert_volume = 0.7
    if 'emergency_contact' not in st.session_state:
        st.session_state.emergency_contact = ""
    if 'emergency_phone' not in st.session_state:
        st.session_state.emergency_phone = ""

    # Application header
    st.title("Real-time Drowsiness Detection")
    st.markdown("""
    This application uses computer vision and machine learning to detect drowsiness in real-time.
    It monitors:
    - Eye Aspect Ratio (EAR)
    - PERCLOS (Percentage of eye closure)
    - Head pose (Pitch, Yaw, Roll)
    """)

    # Create three columns for layout
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("Live Feed")
        
        # Camera selection (if multiple cameras are available)
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
        
        # WebRTC streamer with selected camera
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=DrowsinessDetector,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
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
        st.subheader("Metrics")
        
        # Create metrics displays with improved styling
        alert_status = st.empty()
        ear_metric = st.empty()
        perclos_metric = st.empty()
        head_pose_container = st.empty()
        
        # Volume control slider
        st.subheader("Alert Settings")
        new_volume = st.slider(
            "Alert Volume",
            0.0,
            1.0,
            st.session_state.alert_volume,
            0.1,
            help="Adjust the volume of the drowsiness alert"
        )
        if new_volume != st.session_state.alert_volume:
            st.session_state.alert_volume = new_volume
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.alert.set_volume(new_volume)

    with col3:
        st.subheader("Emergency Contact")
        new_contact = st.text_input(
            "Contact Name",
            value=st.session_state.emergency_contact,
            help="Name of emergency contact"
        )
        new_phone = st.text_input(
            "Contact Phone",
            value=st.session_state.emergency_phone,
            help="Phone number of emergency contact"
        )
        if new_contact != st.session_state.emergency_contact:
            st.session_state.emergency_contact = new_contact
        if new_phone != st.session_state.emergency_phone:
            st.session_state.emergency_phone = new_phone

        # Display saved metrics files
        st.subheader("Saved Metrics")
        metrics_files = list(Path('.').glob('metrics_*.json'))
        if metrics_files:
            selected_file = st.selectbox(
                "Select Metrics File",
                metrics_files,
                format_func=lambda x: x.name
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
            with st.session_state.lock:
                # Update alert status with confidence
                status_color = "#FF0000" if st.session_state.current_prediction == 1 else "#00FF00"
                confidence = st.session_state.prediction_confidence * 100
                alert_status.markdown(
                    f"""
                    <div style="padding: 10px; background-color: {status_color}33; border-radius: 5px;">
                        <h3 style="color: {status_color}; margin: 0;">
                            {"DROWSY" if st.session_state.current_prediction == 1 else "ALERT"}
                        </h3>
                        <p style="margin: 0;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Update EAR with trend indicator
                ear_metric.metric(
                    "Eye Aspect Ratio (EAR)",
                    f"{st.session_state.current_ear:.3f}",
                    delta=None
                )

                # Update PERCLOS with warning threshold
                perclos_metric.metric(
                    "PERCLOS",
                    f"{st.session_state.current_perclos:.3f}",
                    delta="Warning" if st.session_state.current_perclos > 0.15 else None,
                    delta_color="inverse"
                )

                # Update head pose with visual indicators
                head_pose_container.markdown(f"""
                **Head Pose**
                - Pitch: {st.session_state.head_pose[0]:.1f}° {'⚠️' if abs(st.session_state.head_pose[0]) > 30 else ''}
                - Yaw: {st.session_state.head_pose[1]:.1f}° {'⚠️' if abs(st.session_state.head_pose[1]) > 30 else ''}
                - Roll: {st.session_state.head_pose[2]:.1f}° {'⚠️' if abs(st.session_state.head_pose[2]) > 30 else ''}
                """)

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

    # Add usage instructions
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

    # Add troubleshooting section
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
