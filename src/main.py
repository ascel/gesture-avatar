"""
Main demo interface for gesture-controlled avatar project.
Integrates gesture detection, avatar animation, and OBS streaming.
"""

import cv2
import numpy as np
import time
import json
import threading
from pathlib import Path
from typing import Dict, Optional
import argparse

# Import project modules
from gesture_detection.models import GestureDetector
from avatar_animation.simple_avatar import SimpleAvatarManager as AvatarManager
from streaming.obs_integration import StreamManager
from utils.data_preprocessing import GestureDataPreprocessor


class GestureAvatarDemo:
    """Main demo class for gesture-controlled avatar system."""
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the demo system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.is_running = False
        
        # Initialize components
        self.gesture_detector = None
        self.avatar_manager = None
        self.stream_manager = None
        
        # Performance tracking
        self.start_time = None
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Demo state
        self.show_webcam = True
        self.show_avatar = True
        self.show_landmarks = True
        self.show_landmark_numbers = True
        self.show_landmark_connections = True
        self.avatar_type = "2d"  # "2d" or "3d"
        
        # Initialize components
        self.initialize_components()
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration")
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "gesture_detection": {
                "model_type": "feature",
                "confidence_threshold": 0.7,
                "max_hands": 2,
                "target_fps": 15
            },
            "avatar_animation": {
                "avatar_type": "2d",
                "animation_speed": 1.0,
                "smooth_transitions": True
            },
            "streaming": {
                "output_resolution": [720, 480],
                "fps": 15,
                "delay_tolerance": 20
            },
            "hardware": {
                "gpu_acceleration": True,
                "max_memory_usage": 0.8
            }
        }
    
    def initialize_components(self):
        """Initialize all system components."""
        print("Initializing Gesture Avatar System...")
        
        # Initialize gesture detector
        print("1. Initializing gesture detector...")
        try:
            model_path = self.get_best_model_path()
            self.gesture_detector = GestureDetector(
                model_type=self.config["gesture_detection"]["model_type"],
                model_path=model_path,
                config=self.config["gesture_detection"]
            )
            print("✓ Gesture detector initialized")
        except Exception as e:
            print(f"✗ Error initializing gesture detector: {e}")
            print("Using MediaPipe-only detection")
            self.gesture_detector = GestureDetector(
                model_type="feature",
                config=self.config["gesture_detection"]
            )
        
        # Initialize avatar manager
        print("2. Initializing avatar manager...")
        try:
            self.avatar_manager = AvatarManager(
                avatar_type=self.config["avatar_animation"]["avatar_type"],
                config=self.config["avatar_animation"]
            )
            print("✓ Avatar manager initialized")
        except Exception as e:
            print(f"✗ Error initializing avatar manager: {e}")
            print("Using fallback avatar manager")
            self.avatar_manager = AvatarManager(
                avatar_type="2d",
                config={"confidence_threshold": 0.5}
            )
        
        # Initialize stream manager
        print("3. Initializing stream manager...")
        try:
            self.stream_manager = StreamManager(config=self.config["streaming"])
            print("✓ Stream manager initialized")
        except Exception as e:
            print(f"✗ Error initializing stream manager: {e}")
            print("OBS streaming will not be available")
        
        print("✓ All components initialized successfully!")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the best available trained model path."""
        models_dir = Path("data/models")
        
        # Check for training summary
        summary_path = models_dir / "training_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                if summary.get('overall_performance', {}).get('best_model'):
                    best_model = summary['overall_performance']['best_model']
                    for model_info in summary.get('models_trained', []):
                        if model_info['type'] == best_model:
                            return model_info['model_path']
            except Exception as e:
                print(f"Error reading training summary: {e}")
        
        # Fallback: check for any available model
        model_files = list(models_dir.glob("*.h5"))
        if model_files:
            return str(model_files[0])
        
        return None
    
    def start_demo(self):
        """Start the demo system."""
        if self.is_running:
            print("Demo is already running!")
            return
        
        print("\nStarting Gesture Avatar Demo")
        print("============================")
        print("Controls:")
        print("  'q' - Quit")
        print("  'w' - Toggle webcam view")
        print("  'a' - Toggle avatar view")
        print("  'l' - Toggle landmarks")
        print("  'n' - Toggle landmark numbers")
        print("  'c' - Toggle landmark connections")
        print("  '1' - Switch to 2D avatar")
        print("  '2' - Switch to 3D avatar")
        print("  's' - Start/stop OBS streaming")
        print("  'r' - Reset avatar")
        print()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start OBS streaming if available
        streaming_active = False
        if self.stream_manager:
            try:
                streaming_active = self.stream_manager.start_streaming()
                if streaming_active:
                    print("✓ OBS streaming started")
                else:
                    print("⚠ OBS streaming not available")
            except Exception as e:
                print(f"⚠ Error starting OBS streaming: {e}")
        
        print("\nDemo started! Press 'q' to quit.")
        
        try:
            while self.is_running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect gesture
                gesture, confidence, additional_info = self.gesture_detector.detect_gesture(frame)
                
                # Debug: Print gesture detection (only if it's not "no_hand" or "unknown")
                if gesture not in ["no_hand", "unknown"] and confidence > 0.4:
                    # Map gesture numbers to names
                    gesture_mapping = {
                        "gesture_0": "fist",
                        "gesture_1": "open_hand",
                        "gesture_2": "peace",
                        "gesture_3": "point",
                        "gesture_4": "thumbs_up"
                    }
                    mapped_gesture = gesture_mapping.get(gesture, gesture)
                    print(f"Detected: {mapped_gesture} (confidence: {confidence:.2f})")
                
                # Update avatar
                avatar_frame = self.avatar_manager.update(gesture, confidence)
                
                # Draw landmarks if enabled
                if self.show_landmarks and additional_info.get("landmarks"):
                    frame = self.draw_enhanced_landmarks(frame, additional_info["landmarks"], gesture, confidence)
                
                # Create display frame
                display_frame = self.create_display_frame(frame, avatar_frame)
                
                # Send to OBS if streaming
                if streaming_active and self.stream_manager:
                    self.stream_manager.send_frame(avatar_frame)
                
                # Show frame
                cv2.imshow("Gesture Avatar Demo", display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):
                    self.show_webcam = not self.show_webcam
                    print(f"Webcam view: {'ON' if self.show_webcam else 'OFF'}")
                elif key == ord('a'):
                    self.show_avatar = not self.show_avatar
                    print(f"Avatar view: {'ON' if self.show_avatar else 'OFF'}")
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('n'):
                    self.show_landmark_numbers = not self.show_landmark_numbers
                    print(f"Landmark numbers: {'ON' if self.show_landmark_numbers else 'OFF'}")
                elif key == ord('c'):
                    self.show_landmark_connections = not self.show_landmark_connections
                    print(f"Landmark connections: {'ON' if self.show_landmark_connections else 'OFF'}")
                elif key == ord('1'):
                    self.avatar_manager.switch_avatar_type("2d")
                    self.avatar_type = "2d"
                    print("Switched to 2D avatar")
                elif key == ord('2'):
                    self.avatar_manager.switch_avatar_type("3d")
                    self.avatar_type = "3d"
                    print("Switched to 3D avatar")
                elif key == ord('s'):
                    if streaming_active and self.stream_manager:
                        self.stream_manager.stop_streaming()
                        streaming_active = False
                        print("OBS streaming stopped")
                    elif self.stream_manager:
                        streaming_active = self.stream_manager.start_streaming()
                        if streaming_active:
                            print("OBS streaming started")
                elif key == ord('r'):
                    self.avatar_manager.reset()
                    print("Avatar reset")
                
                # Update performance counters
                self.frame_count += 1
                self._update_fps()
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # Cleanup
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            
            if streaming_active and self.stream_manager:
                self.stream_manager.stop_streaming()
            
            # Print final statistics
            self.print_final_stats()
    
    def create_display_frame(self, webcam_frame: np.ndarray, avatar_frame: np.ndarray) -> np.ndarray:
        """Create the display frame combining webcam and avatar."""
        # Resize frames to match
        target_height = 480
        target_width = 640
        
        # Resize webcam frame
        webcam_resized = cv2.resize(webcam_frame, (target_width, target_height))
        
        # Resize avatar frame
        avatar_resized = cv2.resize(avatar_frame, (target_width, target_height))
        
        # Create combined frame
        if self.show_webcam and self.show_avatar:
            # Side by side
            combined = np.zeros((target_height, target_width * 2, 3), dtype=np.uint8)
            combined[:, :target_width] = webcam_resized
            combined[:, target_width:] = avatar_resized
            
            # Add separator
            cv2.line(combined, (target_width, 0), (target_width, target_height), (255, 255, 255), 2)
            
            # Add labels
            cv2.putText(combined, "Webcam", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, "Avatar", (target_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        elif self.show_webcam:
            combined = webcam_resized
        elif self.show_avatar:
            combined = avatar_resized
        else:
            # Show blank frame with info
            combined = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            cv2.putText(combined, "Press 'w' or 'a' to show views", (50, target_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add performance info
        combined = self.add_performance_overlay(combined)
        
        return combined
    
    def add_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add performance information overlay."""
        # Performance info
        info_text = [
            f"FPS: {self.fps_counter:.1f}",
            f"Avatar: {self.avatar_type.upper()}",
            f"Streaming: {'ON' if self.stream_manager and self.stream_manager.is_active else 'OFF'}"
        ]
        
        # Get gesture info from avatar manager
        avatar_info = self.avatar_manager.get_animation_info()
        info_text.extend([
            f"Gesture: {avatar_info['current_gesture']}",
            f"Confidence: {avatar_info['gesture_confidence']:.2f}",
            f"Animation: {avatar_info['current_animation']}"
        ])
        
        # Draw info
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
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
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        if current_time - self.last_fps_time > 0:
            self.fps_counter = 1.0 / (current_time - self.last_fps_time)
        self.last_fps_time = current_time
    
    def print_final_stats(self):
        """Print final demo statistics."""
        if self.start_time:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time if total_time > 0 else 0
            
            print("\nDemo Statistics:")
            print("================")
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Total frames: {self.frame_count}")
            print(f"Average FPS: {avg_fps:.1f}")
            
            if self.stream_manager:
                perf_report = self.stream_manager.get_performance_report()
                print(f"Streaming FPS: {perf_report['performance']['current_fps']:.1f}")
                print(f"Streaming latency: {perf_report['performance']['average_latency']*1000:.1f}ms")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Gesture Avatar Demo")
    parser.add_argument("--config", default="config/config.json", 
                       help="Path to configuration file")
    parser.add_argument("--collect-data", action="store_true",
                       help="Start data collection mode")
    parser.add_argument("--preprocess", action="store_true",
                       help="Run data preprocessing")
    parser.add_argument("--train", action="store_true",
                       help="Train gesture recognition models")
    parser.add_argument("--test-obs", action="store_true",
                       help="Test OBS integration")
    
    args = parser.parse_args()
    
    if args.collect_data:
        print("Starting data collection...")
        from utils.data_collection import main as collect_main
        collect_main()
        return
    
    elif args.preprocess:
        print("Starting data preprocessing...")
        from utils.data_preprocessing import main as preprocess_main
        preprocess_main()
        return
    
    elif args.train:
        print("Starting model training...")
        from gesture_detection.train_models import main as train_main
        train_main()
        return
    
    elif args.test_obs:
        print("Testing OBS integration...")
        from streaming.obs_integration import test_obs_integration
        test_obs_integration()
        return
    
    # Start demo
    demo = GestureAvatarDemo(args.config)
    demo.start_demo()


if __name__ == "__main__":
    main() 