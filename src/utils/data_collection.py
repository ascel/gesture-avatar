"""
Data collection utility for gesture-controlled avatar project.
Records gesture samples using webcam and MediaPipe for training custom models.
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import mediapipe as mp
from pathlib import Path


class GestureDataCollector:
    """Collects gesture data using webcam and MediaPipe for training."""
    
    def __init__(self, output_dir: str = "data/raw", gestures: List[str] = None):
        """
        Initialize the gesture data collector.
        
        Args:
            output_dir: Directory to save collected data
            gestures: List of gesture names to collect
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default gestures from PRD
        self.gestures = gestures or ["wave", "thumbs_up", "point", "clap", "fist"]
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Data storage
        self.current_gesture = None
        self.sample_count = 0
        self.samples_per_gesture = 100  # As specified in PRD
        
    def collect_gesture_data(self, gesture_name: str, num_samples: int = None):
        """
        Collect samples for a specific gesture.
        
        Args:
            gesture_name: Name of the gesture to collect
            num_samples: Number of samples to collect (default: samples_per_gesture)
        """
        if num_samples is None:
            num_samples = self.samples_per_gesture
            
        gesture_dir = self.output_dir / gesture_name
        gesture_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Collecting {num_samples} samples for gesture: {gesture_name}")
        print("Press 'q' to quit, 's' to skip current sample")
        
        collected = 0
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
            
            # Add UI elements
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {collected}/{num_samples}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Gesture Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar to capture
                if results.multi_hand_landmarks:
                    # Save frame and landmarks
                    sample_data = self._extract_landmarks(results.multi_hand_landmarks[0])
                    sample_data['gesture'] = gesture_name
                    sample_data['timestamp'] = datetime.now().isoformat()
                    
                    # Save image
                    img_path = gesture_dir / f"sample_{collected:04d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    
                    # Save landmarks
                    landmarks_path = gesture_dir / f"sample_{collected:04d}.json"
                    with open(landmarks_path, 'w') as f:
                        json.dump(sample_data, f, indent=2)
                    
                    collected += 1
                    print(f"Captured sample {collected}/{num_samples}")
                else:
                    print("No hand detected! Please show your hand.")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Completed collecting {collected} samples for {gesture_name}")
    
    def _extract_landmarks(self, hand_landmarks) -> Dict:
        """Extract landmark coordinates from MediaPipe results."""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        return {'landmarks': landmarks}
    
    def collect_all_gestures(self):
        """Collect data for all defined gestures."""
        print("Starting data collection for all gestures...")
        for gesture in self.gestures:
            print(f"\nPreparing to collect {gesture}...")
            input("Press Enter when ready to start collecting...")
            self.collect_gesture_data(gesture)
        
        print("\nData collection completed!")
        self._generate_dataset_info()
    
    def _generate_dataset_info(self):
        """Generate dataset information and statistics."""
        dataset_info = {
            'gestures': self.gestures,
            'total_samples': len(self.gestures) * self.samples_per_gesture,
            'samples_per_gesture': self.samples_per_gesture,
            'collection_date': datetime.now().isoformat(),
            'gesture_counts': {}
        }
        
        for gesture in self.gestures:
            gesture_dir = self.output_dir / gesture
            if gesture_dir.exists():
                sample_count = len(list(gesture_dir.glob("*.jpg")))
                dataset_info['gesture_counts'][gesture] = sample_count
        
        # Save dataset info
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset info saved to {info_path}")
        print(f"Total samples collected: {dataset_info['total_samples']}")


def main():
    """Main function for data collection."""
    collector = GestureDataCollector()
    
    print("Gesture Data Collection Tool")
    print("============================")
    print("This tool will help you collect gesture data for training.")
    print("Make sure you have good lighting and a clear background.")
    print()
    
    choice = input("Collect all gestures (a) or single gesture (s)? ").lower()
    
    if choice == 'a':
        collector.collect_all_gestures()
    elif choice == 's':
        gesture = input(f"Enter gesture name ({', '.join(collector.gestures)}): ")
        if gesture in collector.gestures:
            collector.collect_gesture_data(gesture)
        else:
            print("Invalid gesture name!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main() 