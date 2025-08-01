#!/usr/bin/env python3
"""
Real-time Gesture Data Collection Script

This script uses your webcam to collect real gesture data for training.
It provides a user-friendly interface for collecting hand gesture samples.

Usage:
    python scripts/collect_gesture_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import time
from pathlib import Path
import argparse
from datetime import datetime

class GestureDataCollector:
    def __init__(self, data_path="../data/raw"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture classes
        self.gesture_classes = ['fist', 'point', 'thumbs_up', 'peace', 'open_hand']
        self.current_gesture = 0
        self.samples_per_gesture = 50
        
        # Collection state
        self.collecting = False
        self.sample_count = 0
        self.gesture_samples = {}
        
        # Initialize sample counts
        for gesture in self.gesture_classes:
            gesture_path = self.data_path / gesture
            gesture_path.mkdir(exist_ok=True)
            
            # Count existing samples
            existing_samples = len(list(gesture_path.glob('*.json')))
            self.gesture_samples[gesture] = existing_samples
    
    def draw_gesture_info(self, frame):
        """Draw gesture information on frame."""
        height, width = frame.shape[:2]
        
        # Current gesture info
        current_gesture_name = self.gesture_classes[self.current_gesture]
        cv2.putText(frame, f"Gesture: {current_gesture_name.upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Sample count
        samples_collected = self.gesture_samples[current_gesture_name]
        cv2.putText(frame, f"Samples: {samples_collected}/{self.samples_per_gesture}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'SPACE' to capture sample", 
                   (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'N' for next gesture, 'P' for previous", 
                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'Q' to quit", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        progress = samples_collected / self.samples_per_gesture
        bar_width = 300
        bar_height = 20
        bar_x = width // 2 - bar_width // 2
        bar_y = height - 120
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Progress bar
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Progress text
        progress_text = f"{samples_collected}/{self.samples_per_gesture}"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y + (bar_height + text_size[1]) // 2
        cv2.putText(frame, progress_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def save_gesture_sample(self, landmarks, gesture_name):
        """Save a gesture sample to JSON file."""
        gesture_path = self.data_path / gesture_name
        gesture_path.mkdir(exist_ok=True)
        
        # Create sample data
        sample_data = {
            'gesture': gesture_name,
            'landmarks': landmarks,
            'timestamp': datetime.now().isoformat(),
            'sample_id': f'sample_{self.gesture_samples[gesture_name]:04d}'
        }
        
        # Save to file
        filename = f"sample_{self.gesture_samples[gesture_name]:04d}.json"
        filepath = gesture_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        self.gesture_samples[gesture_name] += 1
        print(f"✅ Saved {gesture_name} sample {self.gesture_samples[gesture_name]}")
    
    def process_frame(self, frame):
        """Process a single frame and detect hand landmarks."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                for landmark in hand_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z
                    })
        
        return frame, landmarks
    
    def run_collection(self):
        """Run the main data collection loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("🎥 Starting gesture data collection...")
        print("📋 Instructions:")
        print("  • Make the gesture shown on screen")
        print("  • Press SPACE to capture a sample")
        print("  • Press N for next gesture, P for previous")
        print("  • Press Q to quit")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Process frame
            frame, landmarks = self.process_frame(frame)
            
            # Draw UI
            self.draw_gesture_info(frame)
            
            # Show frame
            cv2.imshow('Gesture Data Collection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                if landmarks:
                    current_gesture_name = self.gesture_classes[self.current_gesture]
                    if self.gesture_samples[current_gesture_name] < self.samples_per_gesture:
                        self.save_gesture_sample(landmarks, current_gesture_name)
                        
                        # Show success message
                        cv2.putText(frame, "SAMPLE SAVED!", 
                                   (frame.shape[1]//2 - 100, frame.shape[0]//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        cv2.imshow('Gesture Data Collection', frame)
                        cv2.waitKey(500)  # Show for 500ms
                    else:
                        print(f"⚠️  Maximum samples reached for {current_gesture_name}")
                else:
                    print("⚠️  No hand detected. Please show your hand clearly.")
            elif key == ord('n'):  # Next gesture
                self.current_gesture = (self.current_gesture + 1) % len(self.gesture_classes)
                print(f"📝 Switched to gesture: {self.gesture_classes[self.current_gesture]}")
            elif key == ord('p'):  # Previous gesture
                self.current_gesture = (self.current_gesture - 1) % len(self.gesture_classes)
                print(f"📝 Switched to gesture: {self.gesture_classes[self.current_gesture]}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final summary
        self.print_collection_summary()
    
    def print_collection_summary(self):
        """Print a summary of collected data."""
        print("\n📊 Data Collection Summary:")
        print("=" * 50)
        
        total_samples = 0
        for gesture, count in self.gesture_samples.items():
            print(f"  {gesture:12}: {count:3d}/{self.samples_per_gesture} samples")
            total_samples += count
        
        print(f"\n  Total samples collected: {total_samples}")
        print(f"  Target samples: {len(self.gesture_classes) * self.samples_per_gesture}")
        
        completion_rate = (total_samples / (len(self.gesture_classes) * self.samples_per_gesture)) * 100
        print(f"  Completion rate: {completion_rate:.1f}%")
        
        if completion_rate < 100:
            print(f"\n💡 To complete the dataset, collect {len(self.gesture_classes) * self.samples_per_gesture - total_samples} more samples")
        else:
            print("\n🎉 Dataset collection completed!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Collect gesture data using webcam')
    parser.add_argument('--data-path', default='../data/raw', 
                       help='Path to save collected data')
    parser.add_argument('--samples-per-gesture', type=int, default=50,
                       help='Number of samples to collect per gesture')
    
    args = parser.parse_args()
    
    collector = GestureDataCollector(args.data_path)
    collector.samples_per_gesture = args.samples_per_gesture
    
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
        collector.print_collection_summary()

if __name__ == "__main__":
    main() 