# ASL to Text

A real-time American Sign Language (ASL) fingerspelling-to-text converter that runs entirely in the browser. Point your webcam at your hand, sign ASL letters, and watch them appear as text.

## Features

- **Real-time hand tracking** via Google MediaPipe HandLandmarker (GPU with CPU fallback)
- **Full A–Z alphabet recognition** using a logistic regression classifier trained on normalized 3D hand landmarks
- **Dwell-time character locking** (700ms hold) to prevent false registrations from transitional hand movements
- **Rolling vote window** (7 frames) for temporal smoothing
- **Visual feedback**: detected letter overlay, confidence percentage, dwell progress bar
- **Data collection mode**: record your own landmark samples and export as CSV for re-training
- **Text controls**: clear, backspace, add space

## Run Locally

1. Install Python 3
2. From this project directory, start a local web server:
   ```bash
   python -m http.server 8000
   ```
3. Open [http://localhost:8000](http://localhost:8000) in Chrome or Edge and select **Start camera**.
   Allow camera access when prompted. Webcam access requires `localhost` or HTTPS.

## Training Pipeline

To retrain the classifier (e.g., with real collected data):

```bash
cd training
pip install -r requirements.txt

# Option 1: Generate synthetic data
python generate_synthetic_data.py

# Option 2: Use data collected from the browser's data collection panel
# (Export CSV from the app, save as training/data/asl_landmarks.csv)

# Train and export model
python train.py
```

The training script exports `model.json` to the project root, which the browser app loads automatically on page load.

## How It Works

1. **MediaPipe HandLandmarker** extracts 21 3D hand landmarks from the webcam feed
2. Landmarks are **normalized** relative to the wrist and scaled to unit distance, producing a 63-dimensional feature vector
3. A **Logistic Regression classifier** (trained on landmark data) predicts the most likely letter
4. A **rolling vote window** smooths predictions across frames
5. **Dwell-time locking** requires the user to hold a sign stable for 700ms before committing a character
6. Committed characters appear in the text output buffer

## How to Retrain with Real Data

1. Use the **Training Data Collection** panel in the app to record samples
2. Export CSV → save as `training/data/asl_landmarks.csv`
3. Run `python training/train.py`
4. Refresh the browser to load the updated `model.json`
