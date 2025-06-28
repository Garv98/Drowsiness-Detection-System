# Drowsiness-Detection-System
# Real-Time Drowsiness Detection System

![Drowsiness Detection System](Assets/Logo.png)

## Overview
This project implements a real-time drowsiness detection system using facial landmark detection and machine learning. The system monitors a person's eye state, head pose, and blink patterns to detect signs of drowsiness and provide timely alerts.

## Features
- Real-time facial landmark detection using MediaPipe Face Mesh
- Multiple drowsiness indicators:
  - Eye Aspect Ratio (EAR)
  - PERCLOS (Percentage of Eye Closure)
  - Head Pose Analysis
  - Blink Rate Monitoring
- Dynamic drowsiness confidence scoring (10-100%)
- Audio-visual alerts when drowsiness is detected
- Clean and intuitive Streamlit interface

## Tech Stack
- Python
- MediaPipe
- OpenCV
- Streamlit
- PyTorch
- scikit-learn
- NumPy
- Pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Garv98/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
streamlit run main_app.py
```

The application will open in your default web browser, accessing your webcam for real-time drowsiness detection.

## How It Works

1. **Face and Landmark Detection**: Uses MediaPipe Face Mesh to detect 468 facial landmarks in real-time

2. **Feature Extraction**:
   - Eye Aspect Ratio (EAR) calculation
   - PERCLOS measurement
   - Head pose estimation
   - Blink rate analysis

3. **Drowsiness Detection**:
   - Machine learning model trained on multiple datasets
   - Real-time feature analysis
   - Dynamic confidence scoring
   - Alert system activation based on drowsiness threshold

## Model Training

The system uses a model trained on multiple datasets:
- MRL Eye Dataset
- Driver Drowsiness Dataset
- YawDD Dataset

Features are extracted using MediaPipe Face Mesh and processed to train the final model.

## Project Structure
```
├── main_app.py           # Main Streamlit application
├── requirements.txt      # Project dependencies
├── Assets/              # Project assets
├── drowsiness_detector/ # Core detection modules
│   ├── core.py         # Core detection logic
│   ├── app.py          # Application interface
│   └── model/          # Trained models
└── Data/               # Dataset files
    └── features.csv    # Extracted features
```

## Contributing
Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss the proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- MediaPipe team for their face mesh implementation
- Streamlit for their amazing web app framework
- The open-source community for various drowsiness detection datasets
