#!/usr/bin/env python3
"""
Test script for enhanced hand visualization.

This script demonstrates the improved hand landmark visualization
that was implemented in the main.py script.

Usage:
    python test_enhanced_visualization.py
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from gesture_detection.models import GestureDetector
except ImportError:
    print("❌ Could not import GestureDetector")
    print("   Make sure you have the required dependencies installed")
    sys.exit(1)

class EnhancedVisualizationDemo:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Visualization options
        self.show_landmarks = True
        self.show_landmark_numbers = True
        self.show_landmark_connections = True
        
        # Initialize gesture detector
        try:
            self.gesture_detector = GestureDetector(
                model_type="feature",
                config={"confidence_threshold": 0.5}
            )
            print("✅ Gesture detector initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize gesture detector: {e}")
            self.gesture_detector = None
    
    def draw_enhanced_landmarks(self, frame: np.ndarray, landmarks: list, gesture: str, confidence: float) -> np.ndarray:
        """Draw enhanced hand landmarks with gesture information."""
        if not landmarks or len(landmarks) < 21:
            return frame
        
        # Draw hand landmarks
        for i, landmark in enumerate(landmarks):
            if i < len(landmarks):
                # Convert normalized coordinates to pixel coordinates
                h, w, _ = frame.shape
                x = int(landmark['x'] * w)
                y = int(landmark['y'] * h)
                
                # Different colors for different landmark types
                if i == 0:  # Wrist
                    color = (255, 255, 255)  # White
                    radius = 8
                elif i in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 255, 0)  # Green
                    radius = 6
                elif i in [2, 5, 9, 13, 17]:  # Finger bases
                    color = (255, 0, 0)  # Red
                    radius = 5
                else:  # Other landmarks
                    color = (0, 255, 255)  # Cyan
                    radius = 4
                
                # Draw landmark circle
                cv2.circle(frame, (x, y), radius, color, -1)
                
                # Draw landmark number for debugging
                if self.show_landmark_numbers:
                    cv2.putText(frame, str(i), (x + 5, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw hand connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]
        
        if self.show_landmark_connections:
            for connection in connections:
                if connection[0] < len(landmarks) and connection[1] < len(landmarks):
                    h, w, _ = frame.shape
                    x1 = int(landmarks[connection[0]]['x'] * w)
                    y1 = int(landmarks[connection[0]]['y'] * h)
                    x2 = int(landmarks[connection[1]]['x'] * w)
                    y2 = int(landmarks[connection[1]]['y'] * h)
                    
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Draw gesture information overlay
        if gesture not in ["no_hand", "unknown"] and confidence > 0.5:
            # Map gesture numbers to names
            gesture_mapping = {
                "gesture_0": "clap",
                "gesture_1": "fist", 
                "gesture_2": "open_hand",
                "gesture_3": "peace",
                "gesture_4": "point",
                "gesture_5": "thumbs_up"
            }
            mapped_gesture = gesture_mapping.get(gesture, gesture)
            
            # Create info box
            info_text = f"{mapped_gesture.upper()}"
            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # Calculate text positions
            h, w, _ = frame.shape
            text_x = 20
            text_y = h - 80
            
            # Draw background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(info_text, font, font_scale, thickness)
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, info_text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw confidence
            cv2.putText(frame, confidence_text, (text_x, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def run_demo(self):
        """Run the enhanced visualization demo."""
        print("🎥 Enhanced Hand Visualization Demo")
        print("=" * 40)
        print("Controls:")
        print("  'q' - Quit")
        print("  'l' - Toggle landmarks")
        print("  'n' - Toggle landmark numbers")
        print("  'c' - Toggle landmark connections")
        print()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("✅ Webcam initialized")
        print("📱 Show your hand to see the enhanced visualization!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            landmarks = []
            gesture = "no_hand"
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmark coordinates
                    for landmark in hand_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    # Detect gesture if detector is available
                    if self.gesture_detector:
                        try:
                            gesture, confidence, _ = self.gesture_detector.detect_gesture(frame)
                        except Exception as e:
                            gesture = "unknown"
                            confidence = 0.0
            
            # Draw enhanced landmarks
            if self.show_landmarks and landmarks:
                frame = self.draw_enhanced_landmarks(frame, landmarks, gesture, confidence)
            
            # Add control info
            h, w, _ = frame.shape
            cv2.putText(frame, "Press 'l' to toggle landmarks", (10, h - 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'n' to toggle numbers", (10, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'c' to toggle connections", (10, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show status
            status_text = f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}"
            cv2.putText(frame, status_text, (w - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Enhanced Hand Visualization", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
            elif key == ord('n'):
                self.show_landmark_numbers = not self.show_landmark_numbers
                print(f"Landmark numbers: {'ON' if self.show_landmark_numbers else 'OFF'}")
            elif key == ord('c'):
                self.show_landmark_connections = not self.show_landmark_connections
                print(f"Landmark connections: {'ON' if self.show_landmark_connections else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Demo completed!")

def main():
    """Main function."""
    demo = EnhancedVisualizationDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 