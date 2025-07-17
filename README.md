# Election Watch AI System
![Election Watch AI Demo](<Ballot box monitoring.png>) Example of ballot box monitoring in action

## Overview
This project implements a comprehensive computer vision system to monitor ballot boxes and detect potential electoral fraud or irregularities. The system combines multiple deep learning models to analyze video feeds in real-time, detecting suspicious activities while preserving voter privacy.

### System Tasks
1. Ballot Drop Detection
Model: YOLOv8 (CNN-based object detection)

* Datasets:

. Leap Hand Gesture Dataset

. Synthetic Ballot Dataset (generated)

. Purpose: Detect when hands place ballot papers into the box and count valid drops

2. Tampering Detection
Model: CNN + LSTM (for video anomaly recognition)

* Datasets:


. Synthetic Ballot Dataset

. Purpose: Identify suspicious activities like:

. Box shaking

. Unauthorized opening

. Ballot stuffing

. Other unusual events

3. Voter Re-Entry/Repetition Detection
Model: YOLOv8 + DeepSort (object tracking)

* Dataset: Synthetic Ballot Dataset (with different outfits)

Purpose: Track individuals to detect if the same person votes multiple times

4. Voting Spike Pattern Detection
Model: LSTM (for time-series anomaly detection)

* Dataset: Synthetic CSV Logs (generated from events)

Purpose: Identify unusually high ballot drop rates that may indicate fraud

5. Privacy Protection (Face Blurring)
Model: MTCNN (face detection) + OpenCV Gaussian Blur

* Dataset: LFW Face Dataset

Purpose: Automatically detect and blur faces in footage to protect voter privacy

#### Technical Implementation
Models Used
Model	Purpose	Dataset
YOLOv8	Hand and ballot detection	Leap Hand Gesture + Synthetic Ballot Images
Siamese Network (optional)	Re-entry verification	DukeMTMC-reID
CNN + LSTM	Tampering detection	UCF Crime + Simulated Clips
LSTM	Time-series anomaly detection	Generated CSV logs
MTCNN	Face detection for blurring	LFW Face Dataset
Datasets
Dataset	Source

LFW Face Dataset	Kaggle Link
Hand Gesture Dataset	Kaggle Link
Synthetic Ballot Dataset	Generated using OpenCV/Blender
#####  System Architecture
![alt text](<System Architecture.png>)

####  Getting Started
* Prerequisites
. Python 3.8+

. NVIDIA GPU (recommended)

. CUDA/cuDNN (for GPU acceleration)

##### Installation
bash
* git clone https://github.com/lily-kwamboka/Election_Watch_AI.git
* cd Election_Watch_AI
* pip install -r requirements.txt
Usage
python
python main.py --input video.mp4 --output processed.mp4 --blur-faces True
######  License
* This project is licensed under the MIT License - see the LICENSE file for details.

######  Contact
* For questions or suggestions, please contact: bernicewakarindi@gmail.com - ProjectLead
##### Next Steps:
* View full analysis in the notebook

