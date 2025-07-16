# Import all necessary libraries
import cv2
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
import time
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Face Blur Processor Class
class FaceBlurProcessor:
    '''
    Interactive face detection and blurring system
    '''

    def __init__(self, confidence_threshonld: float = 0.9, blur_strength: int = 71):
        '''
        Args:
            confidence_threshold: Minimum confidence for face detection (0-1)
            blur_strength: Gaussian blur kernel size
        '''

        print('Initialising MTCNN Face Detector ...')
        self.detector = MTCNN()
        self.confidence_threshold = confidence_threshonld
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

        print(f'Face detector initialised with a confidence threshold: {confidence_threshonld}')
        print(f'Blur strength set to: {self.blur_strength}')

    def detect_faces(self, image: np.ndarray) -> List[dict]:
        '''
        Detect faces in an image using MTCNN
        '''

        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect Faces
        faces = self.detector.detect_faces(rgb_image)

        # Filter by confidence threshold
        filtered_faces = [face for face in faces if face['confidence'] >= self.confidence_threshold]

        return filtered_faces
    
    def visualize_detections(self, image: np.ndarray, faces: List[dict], title: str = 'Face Detection Results'):
        '''
        Visualise face detection results
        '''

        plt.figure(figsize = (12, 8))

        # Convert BGR to RGB for display
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw bounding boxes around detected faces
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            confidence = face['confidence']

            # Draw a rectangle
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Add confidence label
            label = f'Face {i + 1}: {confidence:.2f}'
            cv2.putText(display_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 0, 0), 2)

        plt.imshow(display_image)
        plt.title(f'{title}\nDetected {len(faces)} faces')
        plt.axis('off')
        plt.show()

    def blur_faces(self, image:np.ndarray, faces: List[dict]) -> np.ndarray:
        '''
        Apply a Gaussian blur to detected faces
        '''

        result_image = image.copy()

        for face in faces:
            # Extract bounding box coordinates
            x, y, w, h = face['box']

            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[1] - y)

            if w > 0 and h > 0:
                # Extract face region
                face_region = result_image[y : y + h, x : x + w]

                # Apply Gaussian blur
                blurred_face = cv2.GaussianBlur(face_region, (self.blur_strength, self.blur_strength), 0)

                # Replace original face with blurred version
                result_image[y : y + h, x : x + w] = blurred_face

        return result_image
    
    def process_and_display(self, image_path: str, show_detection: bool = True):
        '''
        Process an image and display results
        '''

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f'Could not read image from {image_path}')
            return None
        
        print(f'Processing Image: {image_path}')
        print(f'Image Dimensions: {image.shape[1]}x{image.shape[0]}')

        # Detect faces
        start_time = time.time()
        faces = self.detect_faces(image)
        detection_time = time.time() - start_time

        print(f'Detected {len(faces)} faces in {detection_time:.2f} seconds.')

        # Show detection results if requested
        if show_detection and faces:
            self.visualize_detections(image, faces)

        # Apply blur
        start_time = time.time()
        blurred_image = self.blur_faces(image, faces)
        blur_time = time.time() - start_time

        print(f'Applied blur in {blur_time:.2f} seconds')

        # Display before/after comparison
        self.display_comparison(image, blurred_image)

        return blurred_image
    
    def display_comparison(self, original: np.ndarray, blurred: np.ndarray):
        '''
        Display the before/after comparison
        '''

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8))

        # Original image
        ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize = 14)
        ax1.axis('off')

        # Blurred image
        ax2.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        ax2.set_title('Privacy Protected (Blurred Faces)', fontsize = 14)
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    def save_result(self, blurred_image: np.ndarray, output_path: str):
        '''Save the processed image'''

        cv2.imwrite(output_path, blurred_image)
        print(f'Saved blurred image to: {output_path}')

    def process_live_video(self, camera_index: int = 0, window_name: str = 'Face Blur - Live Feed'):
        '''
        Process live video feed with face blurring

        Args:
            camera_index: Camera index (0 for default camera)
            window_name: Name of the display window
        '''

        # Imitialising the camera
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f'Error: Could not open camera {camera_index}')
            return
        
        # Set Camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting live video feed...")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'd' to toggle detection display")

        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        show_detection_boxes = False

        while True:
            # Capture Frame
            ret, frame = cap.read()

            if not ret:
                print('Error: Could not find read frame.')
                break

            frame_start_time = time.time()

            # Detect faces in the frame
            faces = self.detect_faces(frame)

            # Create processed frame
            processed_frame = frame.copy()

            if faces:
                # Apply blur to faces
                processed_frame = self.blur_faces(frame, faces)

                # Draw detection boxes (optional)
                if show_detection_boxes:
                    for i, face in enumerate(faces):
                        x, y, w, h = face['box']
                        confidence = face['confidence']

                        # Draw rectangle around the detected face(s)
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Adding a confidence label
                        label = f'Face {i + 1}: {confidence:.2f}'
                        cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Calculate and display the fps
            frame_time = time.time() - frame_start_time
            fps_counter += 1


            if fps_counter % 10 == 0: 
                fps = fps_counter / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0

                # Display FPS and face count on frame:
                cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(processed_frame, f'Faces: {len(faces)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow(window_name, processed_frame)

            # Handling the key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print('Quitting...')
                break
            elif key == ord('s'):
                # Save the current frame
                time_stamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f'blurred_frame_{time_stamp}.jpg'
                cv2.imwrite(filename, processed_frame)
                print(f'File saved as: {filename}')
            elif key == ord('d'):
                # Toggling the detection box
                show_detection_boxes = not show_detection_boxes
                status = 'ON' if show_detection_boxes else 'OFF'
                print(f'Detection boxes: {status}')

            frame_count += 1

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f'Processed {frame_count} frames.')

class LiveFaceBlurApp:
    '''
    Live face blurring app main class
    '''

    def __init__(self, confidence_threshold: float = 0.9, blur_strength: int = 71):
        self.processor = FaceBlurProcessor(confidence_threshold, blur_strength)

    def run(self, camera_index: int = 0):
        '''
        Running the face blurring app
        '''

        try:
            self.processor.process_live_video(camera_index)
        except KeyboardInterrupt:
            print('\nApplication interrupted by user')
        except Exception as e:
            print(f'Error occurred: {e}')

# Usage example
if __name__ == "__main__":
    # Create the application
    app = LiveFaceBlurApp(confidence_threshold=0.8, blur_strength=51)
    
    # Run the application
    print("Starting Live Face Blur Application...")
    app.run(camera_index=0) 
            