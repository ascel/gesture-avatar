"""
Simple avatar animation system that definitely works.
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional


class SimpleAvatar:
    """Simple animated avatar that responds to gestures."""
    
    def __init__(self):
        self.current_gesture = "idle"
        self.animation_time = 0
        self.animation_speed = 0.3
        self.last_gesture_change = time.time()
        
    def update(self, gesture: str, confidence: float) -> np.ndarray:
        """Update avatar based on gesture."""
        current_time = time.time()
        
        # Map gesture numbers to names
        gesture_mapping = {
            "gesture_0": "fist",
            "gesture_1": "open_hand", 
            "gesture_2": "peace",
            "gesture_3": "point",
            "gesture_4": "thumbs_up",
            "gesture_5": "wave"
        }
        
        # Convert gesture to proper name
        mapped_gesture = gesture_mapping.get(gesture, gesture)
        
        # Update gesture if confidence is high enough (lowered threshold for better responsiveness)
        if (mapped_gesture != self.current_gesture and 
            confidence > 0.4 and 
            current_time - self.last_gesture_change > 0.2):
            self.current_gesture = mapped_gesture
            self.last_gesture_change = current_time
            print(f"Avatar gesture changed to: {mapped_gesture} (confidence: {confidence:.2f})")
        
        # Update animation time
        self.animation_time += self.animation_speed
        
        # Create avatar frame
        frame = self.create_avatar_frame()
        
        return frame
    
    def create_avatar_frame(self) -> np.ndarray:
        """Create the current avatar frame."""
        # Create a 400x400 frame
        frame = np.ones((400, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Get animation parameters based on gesture
        params = self.get_animation_params()
        
        # Draw the avatar
        self.draw_avatar(frame, params)
        
        # Add text overlay
        self.add_text_overlay(frame)
        
        return frame
    
    def get_animation_params(self) -> Dict:
        """Get animation parameters for current gesture."""
        base_params = {
            "body_color": (100, 150, 255),  # Blue
            "head_color": (100, 150, 255),
            "eye_color": (255, 255, 255),
            "pupil_color": (0, 0, 0),
            "scale": 1.0,
            "rotation": 0,
            "offset_x": 0,
            "offset_y": 0
        }
        
        # Apply gesture-specific animations
        if self.current_gesture == "idle":
            # Gentle breathing
            scale = 1.0 + 0.05 * np.sin(self.animation_time * 0.5)
            base_params["scale"] = scale
            
        elif self.current_gesture == "wave":
            # Waving animation
            rotation = 15 * np.sin(self.animation_time * 1.0)
            offset_x = 10 * np.sin(self.animation_time * 1.2)
            base_params["rotation"] = rotation
            base_params["offset_x"] = offset_x
            base_params["body_color"] = (200, 100, 150)  # Pink
            
        elif self.current_gesture == "thumbs_up":
            # Bounce up and down
            offset_y = -15 * np.sin(self.animation_time * 1.5)
            scale = 1.0 + 0.08 * np.sin(self.animation_time * 1.0)
            base_params["offset_y"] = offset_y
            base_params["scale"] = scale
            base_params["body_color"] = (150, 200, 100)  # Green
            
        elif self.current_gesture == "point":
            # Side movement
            offset_x = 15 * np.sin(self.animation_time * 1.3)
            rotation = 8 * np.sin(self.animation_time * 0.7)
            base_params["offset_x"] = offset_x
            base_params["rotation"] = rotation
            base_params["body_color"] = (200, 150, 100)  # Orange
            
        elif self.current_gesture == "peace":
            # Gentle sway
            rotation = 10 * np.sin(self.animation_time * 0.9)
            scale = 1.0 + 0.06 * np.sin(self.animation_time * 1.1)
            base_params["rotation"] = rotation
            base_params["scale"] = scale
            base_params["body_color"] = (200, 100, 200)  # Purple
            
        elif self.current_gesture == "open_hand":
            # Scale up and down
            scale = 1.0 + 0.1 * np.sin(self.animation_time * 1.2)
            offset_y = -8 * np.sin(self.animation_time * 1.4)
            base_params["scale"] = scale
            base_params["offset_y"] = offset_y
            base_params["body_color"] = (200, 100, 100)  # Red
            
        elif self.current_gesture == "fist":
            # Shake
            rotation = 12 * np.sin(self.animation_time * 1.6)
            scale = 1.0 + 0.05 * np.sin(self.animation_time * 1.3)
            base_params["rotation"] = rotation
            base_params["scale"] = scale
            base_params["body_color"] = (100, 100, 200)  # Dark Blue
            
        return base_params
    
    def draw_avatar(self, frame: np.ndarray, params: Dict):
        """Draw the avatar on the frame."""
        center_x, center_y = 200, 200  # Center of frame
        
        # Apply transformations
        scale = params["scale"]
        rotation = params["rotation"]
        offset_x = params["offset_x"]
        offset_y = params["offset_y"]
        
        # Calculate transformed center
        tx = int(center_x + offset_x)
        ty = int(center_y + offset_y)
        
        # Draw body (rectangle)
        body_width = int(80 * scale)
        body_height = int(100 * scale)
        body_x = int(tx - body_width // 2)
        body_y = int(ty - body_height // 2)
        
        cv2.rectangle(frame, 
                     (body_x, body_y), 
                     (body_x + body_width, body_y + body_height), 
                     params["body_color"], -1)
        
        # Draw head (circle)
        head_radius = int(30 * scale)
        head_x, head_y = tx, body_y - head_radius - 10
        
        cv2.circle(frame, (head_x, head_y), head_radius, params["head_color"], -1)
        
        # Draw eyes
        eye_radius = int(5 * scale)
        cv2.circle(frame, (head_x - 15, head_y), eye_radius, params["eye_color"], -1)
        cv2.circle(frame, (head_x + 15, head_y), eye_radius, params["eye_color"], -1)
        
        # Draw pupils
        pupil_radius = int(2 * scale)
        cv2.circle(frame, (head_x - 15, head_y), pupil_radius, params["pupil_color"], -1)
        cv2.circle(frame, (head_x + 15, head_y), pupil_radius, params["pupil_color"], -1)
        
        # Draw arms based on gesture
        self.draw_arms(frame, params, tx, body_y, body_width, body_height)
        
        # Draw legs
        leg_width = int(20 * scale)
        leg_height = int(40 * scale)
        cv2.rectangle(frame, 
                     (tx - 30, body_y + body_height), 
                     (tx - 30 + leg_width, body_y + body_height + leg_height), 
                     params["body_color"], -1)
        cv2.rectangle(frame, 
                     (tx + 10, body_y + body_height), 
                     (tx + 10 + leg_width, body_y + body_height + leg_height), 
                     params["body_color"], -1)
        
        # Apply rotation if needed
        if abs(rotation) > 0.1:
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
            frame[:] = cv2.warpAffine(frame, rotation_matrix, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
    
    def draw_arms(self, frame: np.ndarray, params: Dict, center_x: int, body_y: int, 
                 body_width: int, body_height: int):
        """Draw arms based on gesture."""
        scale = params["scale"]
        arm_width = int(15 * scale)
        arm_height = int(40 * scale)
        
        if self.current_gesture == "fist":
            # Arms down with fists
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y + 10), 
                         (center_x - body_width//2, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y + 10), 
                         (center_x + body_width//2 + arm_width, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            
            # Fists
            fist_radius = int(10 * scale)
            cv2.circle(frame, 
                      (center_x - body_width//2 - arm_width - 10, body_y + 10 + arm_height), 
                      fist_radius, params["body_color"], -1)
            cv2.circle(frame, 
                      (center_x + body_width//2 + arm_width + 10, body_y + 10 + arm_height), 
                      fist_radius, params["body_color"], -1)
            
        elif self.current_gesture == "point":
            # Left arm pointing
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y + 10), 
                         (center_x - body_width//2, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            # Pointing finger
            cv2.line(frame, 
                    (center_x - body_width//2 - arm_width, body_y + 10 + arm_height),
                    (center_x - body_width//2 - arm_width - 20, body_y + 10 + arm_height - 10),
                    (255, 255, 255), 3)
            
            # Right arm down
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y + 10), 
                         (center_x + body_width//2 + arm_width, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            
        elif self.current_gesture == "peace":
            # Both arms up in peace sign
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y - 20), 
                         (center_x - body_width//2, body_y - 20 + arm_height), 
                         params["body_color"], -1)
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y - 20), 
                         (center_x + body_width//2 + arm_width, body_y - 20 + arm_height), 
                         params["body_color"], -1)
            
            # Peace signs
            cv2.line(frame, 
                    (center_x - body_width//2 - arm_width, body_y - 20),
                    (center_x - body_width//2 - arm_width - 15, body_y - 20 - 15),
                    (255, 255, 255), 3)
            cv2.line(frame, 
                    (center_x - body_width//2 - arm_width, body_y - 20),
                    (center_x - body_width//2 - arm_width - 15, body_y - 20 + 15),
                    (255, 255, 255), 3)
            
            cv2.line(frame, 
                    (center_x + body_width//2 + arm_width, body_y - 20),
                    (center_x + body_width//2 + arm_width + 15, body_y - 20 - 15),
                    (255, 255, 255), 3)
            cv2.line(frame, 
                    (center_x + body_width//2 + arm_width, body_y - 20),
                    (center_x + body_width//2 + arm_width + 15, body_y - 20 + 15),
                    (255, 255, 255), 3)
            
        elif self.current_gesture == "open_hand":
            # Both arms up, open hands
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y - 20), 
                         (center_x - body_width//2, body_y - 20 + arm_height), 
                         params["body_color"], -1)
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y - 20), 
                         (center_x + body_width//2 + arm_width, body_y - 20 + arm_height), 
                         params["body_color"], -1)
            
            # Open hands (circles)
            hand_radius = int(15 * scale)
            cv2.circle(frame, 
                      (center_x - body_width//2 - arm_width, body_y - 20), 
                      hand_radius, params["body_color"], -1)
            cv2.circle(frame, 
                      (center_x + body_width//2 + arm_width, body_y - 20), 
                      hand_radius, params["body_color"], -1)
            
        elif self.current_gesture == "thumbs_up":
            # Left arm with thumbs up
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y + 10), 
                         (center_x - body_width//2, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            # Thumbs up
            cv2.circle(frame, 
                      (center_x - body_width//2 - arm_width - 10, body_y + 10 + arm_height), 
                      int(10 * scale), params["body_color"], -1)
            cv2.line(frame, 
                    (center_x - body_width//2 - arm_width, body_y + 10 + arm_height),
                    (center_x - body_width//2 - arm_width, body_y + 10 + arm_height - 15),
                    (255, 255, 255), 3)
            
            # Right arm down
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y + 10), 
                         (center_x + body_width//2 + arm_width, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            
        elif self.current_gesture == "wave":
            # Right arm waving
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y - 10), 
                         (center_x + body_width//2 + arm_width, body_y - 10 + arm_height), 
                         params["body_color"], -1)
            # Waving hand
            cv2.circle(frame, 
                      (center_x + body_width//2 + arm_width, body_y - 10), 
                      int(10 * scale), params["body_color"], -1)
            
            # Left arm down
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y + 10), 
                         (center_x - body_width//2, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            
        else:  # idle
            # Arms down
            cv2.rectangle(frame, 
                         (center_x - body_width//2 - arm_width, body_y + 10), 
                         (center_x - body_width//2, body_y + 10 + arm_height), 
                         params["body_color"], -1)
            cv2.rectangle(frame, 
                         (center_x + body_width//2, body_y + 10), 
                         (center_x + body_width//2 + arm_width, body_y + 10 + arm_height), 
                         params["body_color"], -1)
    
    def add_text_overlay(self, frame: np.ndarray):
        """Add text overlay to the frame."""
        # Add gesture info
        cv2.putText(frame, f"Gesture: {self.current_gesture}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add animation info
        cv2.putText(frame, "Robot Avatar", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
        
        # Add debug info
        cv2.putText(frame, f"Animation Time: {self.animation_time:.1f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add instructions
        cv2.putText(frame, "Try: fist, point, peace, open_hand, thumbs_up, wave", 
                   (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


class SimpleAvatarManager:
    """Simple avatar manager that definitely works."""
    
    def __init__(self, avatar_type: str = "2d", config: Optional[Dict] = None):
        self.avatar = SimpleAvatar()
        self.current_gesture = "no_hand"
        self.gesture_confidence = 0.0
        self.avatar_type = avatar_type
        self.config = config or {}
    
    def update(self, gesture: str, confidence: float) -> np.ndarray:
        """Update avatar based on gesture detection."""
        self.current_gesture = gesture
        self.gesture_confidence = confidence
        
        return self.avatar.update(gesture, confidence)
    
    def switch_avatar_type(self, avatar_type: str):
        """Switch avatar type (placeholder)."""
        print(f"Switched to {avatar_type} avatar (using simple avatar)")
    
    def get_animation_info(self) -> Dict:
        """Get current animation information."""
        return {
            'avatar_type': 'simple',
            'current_gesture': self.current_gesture,
            'gesture_confidence': self.gesture_confidence,
            'current_animation': self.avatar.current_gesture,
            'fps': 30.0
        }
    
    def reset(self):
        """Reset avatar to initial state."""
        self.avatar.current_gesture = "idle"
        self.avatar.animation_time = 0
        self.current_gesture = "no_hand"
        self.gesture_confidence = 0.0 