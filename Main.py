from scripts.ballot_drop import run_ballot_drop_detection
from scripts.face_recognition import run_face_recognition_demo
from scripts.Tampering_detection import run_tampering_detection_demo

from scripts.privacy import run_privacy_blur_demo
def main():
    print("Starting AI Voting Surveillance System...\n")
    run_ballot_drop_detection()
    run_tampering_detection_demo()
    run_privacy_blur_demo()
    run_face_recognition_demo()
    
    
    
    
if __name__ == "__main__":
    main()
