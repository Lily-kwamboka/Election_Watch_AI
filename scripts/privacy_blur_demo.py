import cv2
from mtcnn import MTCNN
import numpy as np


def blur_faces(frame, faces, blur_strength=35):
    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        # Ensure the box stays within frame bounds
        x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
        face_region = frame[y:y2, x:x2]
        # Apply Gaussian blur to the face region
        if face_region.size > 0:
            blurred = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
            frame[y:y2, x:x2] = blurred
    return frame


def run_privacy_blur_demo():
    print("[INFO] Starting live privacy blur demo. Press 'q' to quit.")
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        frame_blurred = blur_faces(frame, faces)
        cv2.imshow('Privacy Blur (Press q to exit)', frame_blurred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_privacy_blur_demo()
