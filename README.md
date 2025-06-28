# Real-time Drowsiness Detection System

A robust, real-time drowsiness detection system that uses facial landmarks to monitor driver alertness. The system analyzes various facial features including Eye Aspect Ratio (EAR), PERCLOS (Percentage of Eye Closure), head pose, and blink rate to determine drowsiness levels.

## Features

- Real-time facial landmark detection using MediaPipe Face Mesh
- Multiple drowsiness indicators:
  - Eye Aspect Ratio (EAR)
  - PERCLOS (Percentage of Eye Closure)
  - Head Pose Estimation
  - Blink Rate Analysis
- Dynamic drowsiness confidence calculation (10-100%)
- Real-time alerts for drowsy state
- Modern Streamlit-based user interface
- Trained on multiple datasets for robust performance

## Project Structure

```
.
├── Data/
│   └── features.csv            # Extracted features from all datasets
├── Drowsiness-Landmark-Detection/
│   ├── analyze_datasets.py     # Dataset analysis scripts
│   ├── app_final.py           # Final production app
│   ├── drowsiness_detector/   # Core detection module
│   ├── extract_datasets.py    # Dataset extraction utilities
│   ├── extract_features.py    # Feature extraction pipeline
│   ├── model/                 # Trained models directory
│   ├── preprocess_datasets.py # Data preprocessing scripts
│   ├── requirements.txt       # Project dependencies
│   └── train_model.py        # Model training pipeline
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Garv98/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app_final.py
```

2. Allow camera access when prompted
3. The app will display:
   - Live video feed with facial landmarks
   - Real-time drowsiness metrics
   - Alert messages when drowsiness is detected

## How it Works

1. **Feature Extraction**:
   - Facial landmarks are detected using MediaPipe Face Mesh
   - EAR (Eye Aspect Ratio) is calculated from eye landmarks
   - PERCLOS measures the percentage of eye closure over time
   - Head pose is estimated using facial geometry

2. **Drowsiness Detection**:
   - Multiple features are combined with appropriate weights
   - Drowsiness confidence is calculated on a scale of 10-100%
   - Alerts are triggered based on confidence thresholds

3. **Model Training**:
   - Features extracted from multiple datasets (MRL Eye, Driver Drowsiness, YawDD)
   - Model trained using ensemble methods for robust performance
   - Regular validation and testing to ensure accuracy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- MediaPipe for facial landmark detection
- MRL Eye Dataset
- Driver Drowsiness Dataset
- YawDD (Yawning Detection Dataset)
- Streamlit for the web interface
