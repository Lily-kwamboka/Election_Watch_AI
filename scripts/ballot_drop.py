# modules/ballot_drop.py

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent hangs on import

print("[DEBUG] Importing cv2...")
import cv2
print("[DEBUG] Importing numpy...")
import numpy as np
print("[DEBUG] Importing time...")
import time
print("[DEBUG] Importing deque...")
from collections import deque
print("[DEBUG] Importing tensorflow.keras.models.load_model...")
from tensorflow.keras.models import load_model

# Load model and gesture classes
model = load_model("models/gesture_model.keras")
class_labels = ['Two_Fingers_Edge', 'Hands_Down', 'Holding_Ballot', 'Moved_Palm', 'Open_Palm']

# Buffer for last gestures
gesture_buffer = deque(maxlen=5)
drop_logged = False

# Preprocess camera frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# Drop detection logic
def detect_ballot_drop(predicted_class):
    global drop_logged
    gesture_buffer.append(predicted_class)
    pattern = list(gesture_buffer)

    if 'Holding_Ballot' in pattern and 'Hands_Down' in pattern and 'Open_Palm' in pattern:
        if not drop_logged:
            print("✅ Ballot drop detected!")
            with open("drop_log.txt", "a") as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"Drop detected at {timestamp}\n")
            drop_logged = True
    else:
        drop_logged = False

# Main run function
def run_ballot_drop_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Region of interest
        roi = frame[100:300, 100:300]
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

        img = preprocess_frame(roi)

        # Predict
        prediction = model.predict(img, verbose=0)
        predicted_class = class_labels[np.argmax(prediction)]

        # Display result
        cv2.putText(frame, f"Gesture: {predicted_class}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        detect_ballot_drop(predicted_class)

        cv2.imshow("Ballot Drop Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()





