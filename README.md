# Driver Fatigue Detection using Ensemble Learning

A real-time driver fatigue detection system that analyzes facial cues using computer vision and machine learning to enhance road safety.

## Features
Real-time fatigue detection using webcam
Eye Aspect Ratio (EAR) for eye closure detection
Mouth Aspect Ratio (MAR) for yawn detection
Blink frequency and head pose estimation
Stacked ensemble model for high accuracy prediction
GUI interface with live monitoring
Alert system (audio warning) for drowsy drivers
## Tools Used
Python
OpenCV
MediaPipe
Scikit-learn
XGBoost
LightGBM
CatBoost
Tkinter
## Dataset Details

Custom real-time dataset generated using webcam input.

## Features:
EAR (Eye Aspect Ratio)
MAR (Mouth Aspect Ratio)
Blink Count (30-frame window)
EAR Mean & Variance
Head Pose (Yaw, Pitch, Roll)
Label (0 = Alert, 1 = Fatigued)
## Model Overview
Stacked Ensemble Learning Approach:
Base Models: XGBoost, LightGBM, CatBoost
Meta Model: Logistic Regression
Data preprocessing using StandardScaler
Temporal feature engineering for improved accuracy
## Performance
Accuracy: ~92–96%
Precision, Recall, F1-score evaluated using test dataset
Robust performance compared to single models
## How It Works
Capture live video using webcam
Detect face landmarks using MediaPipe
Extract EAR, MAR, blink rate, and head pose
Pass features to trained ensemble model
Predict fatigue probability
Trigger alert if drowsiness detected
### Usage
Step 1: Collect data
python main.py → Choose option 1

Step 2: Train model
python main.py → Choose option 2

Step 3: Run real-time detection
python main.py → Choose option 3
## Project Structure
fatigue_features.csv     # Dataset  
fatigue_ensemble.pkl     # Trained model  
main.py                  # Main script  
shap_summary.png         # Model insights (optional)  
## Project by

S Prithika
Aspiring Software Engineer / Machine Learning Enthusiast
