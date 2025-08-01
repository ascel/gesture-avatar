"""
Avatar animation system for gesture-controlled avatar project.
Maps detected gestures to avatar animations with smooth transitions.
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import math


class AvatarAnimation:
    """Base class for avatar animations."""
    
    def __init__(self, avatar_type: str = "2d"):
        """
        Initialize avatar animation.
        
        Args:
            avatar_type: "2d" or "3d"
        """
        self.avatar_type = avatar_type
        self.current_animation = "idle"
        self.animation_frame = 0
        self.animation_speed = 0.5  # Slower for better visibility
        self.transition_duration = 0.3  # seconds
        self.last_animation_change = time.time()
        self.animation_time = 0
        
    def update_animation(self, gesture: str, confidence: float) -> str:
        """
        Update animation based on detected gesture.
        
        Args:
            gesture: Detected gesture name
            confidence: Detection confidence
            
        Returns:
            Current animation name
        """
        raise NotImplementedError
    
    def get_current_frame(self) -> np.ndarray:
        """Get current animation frame."""
        raise NotImplementedError
    
    def reset_animation(self):
        """Reset to idle animation."""
        self.current_animation = "idle"
        self.animation_frame = 0
        self.animation_time = 0


class SpriteBasedAvatar(AvatarAnimation):
    """2D sprite-based avatar using spritesheets."""
    
    def __init__(self, spritesheet_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize sprite-based avatar.
        
        Args:
            spritesheet_path: Path to spritesheet image
            config: Animation configuration
        """
        super().__init__("2d")
        self.config = config or {}
        self.spritesheet = None
        self.sprites = {}
        self.sprite_size = (200, 200)  # Target size for avatars
        
        if spritesheet_path:
            self.load_spritesheet(spritesheet_path)
        else:
            self.load_sample_avatars()
    
    def load_spritesheet(self, spritesheet_path: str):
        """Load spritesheet and extract sprites."""
        try:
            spritesheet = Image.open(spritesheet_path)
            self.spritesheet = np.array(spritesheet)
            
            # Extract sprites for each animation
            animations = ["idle", "wave", "thumbs_up", "point", "clap", "fist"]
            rows = len(animations)
            cols = 8  # 8 frames per animation
            
            for i, animation in enumerate(animations):
                animation_sprites = []
                for j in range(cols):
                    x = j * self.sprite_size[0]
                    y = i * self.sprite_size[1]
                    sprite = self.spritesheet[y:y+self.sprite_size[1], 
                                            x:x+self.sprite_size[0]]
                    animation_sprites.append(sprite)
                self.sprites[animation] = animation_sprites
            
            print(f"Loaded spritesheet with {len(animations)} animations")
            
        except Exception as e:
            print(f"Error loading spritesheet: {e}")
            self.load_sample_avatars()
    
    def load_sample_avatars(self):
        """Load sample avatars from the data/avatars/2d directory."""
        try:
            # Try to load from sample avatars directory
            avatars_dir = Path("data/avatars/2d")
            if not avatars_dir.exists():
                print("Sample avatars directory not found, creating animated sprites")
                self.create_animated_sprites()
                return
            
            # Load the first available avatar (robot, alien, or ghost)
            avatar_options = ["robot", "alien", "ghost"]
            selected_avatar = None
            
            for avatar_name in avatar_options:
                avatar_dir = avatars_dir / avatar_name
                if avatar_dir.exists():
                    selected_avatar = avatar_name
                    break
            
            if selected_avatar is None:
                print("No sample avatars found, creating animated sprites")
                self.create_animated_sprites()
                return
            
            print(f"Loading sample avatar: {selected_avatar}")
            avatar_dir = avatars_dir / selected_avatar
            
            # Load individual gesture sprites and create animations
            gestures = ["idle", "fist", "point", "peace", "open_hand", "thumbs_up", "wave"]
            
            for gesture in gestures:
                sprite_path = avatar_dir / f"{gesture}.png"
                if sprite_path.exists():
                    # Load sprite image
                    sprite_img = Image.open(sprite_path)
                    sprite_array = np.array(sprite_img)
                    
                    # Convert RGBA to RGB if needed
                    if len(sprite_array.shape) == 3 and sprite_array.shape[2] == 4:
                        # Create white background
                        background = np.ones((sprite_array.shape[0], sprite_array.shape[1], 3), dtype=np.uint8) * 255
                        # Composite RGBA over white background
                        alpha = sprite_array[:, :, 3:4] / 255.0
                        sprite_array = (sprite_array[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
                    
                    # Always resize sprite to target size to ensure consistency
                    sprite_array = cv2.resize(sprite_array, self.sprite_size)
                    
                    # Create animated frames from the static sprite
                    animated_frames = self.create_animated_frames(sprite_array, gesture)
                    self.sprites[gesture] = animated_frames
                else:
                    # Create a placeholder sprite for missing gestures
                    placeholder = np.ones((self.sprite_size[0], self.sprite_size[1], 3), dtype=np.uint8) * 128
                    animated_frames = self.create_animated_frames(placeholder, gesture)
                    self.sprites[gesture] = animated_frames
            
            print(f"Loaded {len(self.sprites)} gesture sprites for {selected_avatar}")
            
        except Exception as e:
            print(f"Error loading sample avatars: {e}")
            print("Falling back to animated sprites")
            self.create_animated_sprites()
    
    def create_animated_frames(self, base_sprite: np.ndarray, gesture: str) -> List[np.ndarray]:
        """Create animated frames from a base sprite."""
        frames = []
        num_frames = 8  # 8 frames per animation
        
        for frame_idx in range(num_frames):
            # Create a copy of the base sprite
            frame = base_sprite.copy()
            
            # Apply animation effects based on gesture
            if gesture == "idle":
                # Gentle breathing animation
                scale = 1.0 + 0.03 * np.sin(frame_idx * 0.8)
                frame = self.apply_scale(frame, scale)
                
            elif gesture == "wave":
                # Waving animation - rotate and move
                angle = 15 * np.sin(frame_idx * 1.0)
                offset_x = 8 * np.sin(frame_idx * 1.2)
                frame = self.apply_rotation(frame, angle)
                frame = self.apply_translation(frame, offset_x, 0)
                
            elif gesture == "thumbs_up":
                # Thumbs up animation - bounce up and down
                offset_y = -10 * np.sin(frame_idx * 1.5)
                scale = 1.0 + 0.05 * np.sin(frame_idx * 1.0)
                frame = self.apply_scale(frame, scale)
                frame = self.apply_translation(frame, 0, offset_y)
                
            elif gesture == "point":
                # Pointing animation - move side to side
                offset_x = 12 * np.sin(frame_idx * 1.3)
                angle = 8 * np.sin(frame_idx * 0.7)
                frame = self.apply_translation(frame, offset_x, 0)
                frame = self.apply_rotation(frame, angle)
                
            elif gesture == "peace":
                # Peace sign animation - gentle sway and scale
                angle = 8 * np.sin(frame_idx * 0.9)
                scale = 1.0 + 0.04 * np.sin(frame_idx * 1.1)
                frame = self.apply_rotation(frame, angle)
                frame = self.apply_scale(frame, scale)
                
            elif gesture == "open_hand":
                # Open hand animation - scale up and down with bounce
                scale = 1.0 + 0.06 * np.sin(frame_idx * 1.2)
                offset_y = -5 * np.sin(frame_idx * 1.4)
                frame = self.apply_scale(frame, scale)
                frame = self.apply_translation(frame, 0, offset_y)
                
            elif gesture == "fist":
                # Fist animation - shake and scale
                angle = 12 * np.sin(frame_idx * 1.6)
                scale = 1.0 + 0.03 * np.sin(frame_idx * 1.3)
                frame = self.apply_rotation(frame, angle)
                frame = self.apply_scale(frame, scale)
            
            frames.append(frame)
        
        return frames
    
    def apply_scale(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Apply scaling transformation to frame."""
        if abs(scale - 1.0) < 0.01:
            return frame
        
        h, w = frame.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad or crop to maintain original size
        result = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        
        if new_h > h:
            # Crop height
            start_h = (new_h - h) // 2
            resized = resized[start_h:start_h + h, :]
        elif new_h < h:
            # Pad height
            pad_h = (h - new_h) // 2
            # Ensure we don't exceed bounds
            end_h = min(pad_h + new_h, h)
            resized_h = min(new_h, h - pad_h)
            result[pad_h:end_h, :] = resized[:resized_h, :]
            return result
        
        if new_w > w:
            # Crop width
            start_w = (new_w - w) // 2
            resized = resized[:, start_w:start_w + w]
        elif new_w < w:
            # Pad width
            pad_w = (w - new_w) // 2
            # Ensure we don't exceed bounds
            end_w = min(pad_w + new_w, w)
            resized_w = min(new_w, w - pad_w)
            result[:, pad_w:end_w] = resized[:, :resized_w]
            return result
        
        return resized
    
    def apply_rotation(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Apply rotation transformation to frame."""
        if abs(angle) < 0.1:
            return frame
        
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(frame, rotation_matrix, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
        
        return rotated
    
    def apply_translation(self, frame: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
        """Apply translation transformation to frame."""
        if abs(offset_x) < 1 and abs(offset_y) < 1:
            return frame
        
        h, w = frame.shape[:2]
        
        # Create translation matrix
        translation_matrix = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float32)
        
        # Apply translation
        translated = cv2.warpAffine(frame, translation_matrix, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))
        
        return translated
    
    def create_animated_sprites(self):
        """Create animated colored sprites for testing."""
        gestures = ["idle", "wave", "thumbs_up", "point", "peace", "open_hand", "fist"]
        colors = {
            "idle": (100, 150, 200),      # Blue
            "wave": (200, 100, 150),      # Pink
            "thumbs_up": (150, 200, 100), # Green
            "point": (200, 150, 100),     # Orange
            "peace": (200, 100, 200),     # Purple
            "open_hand": (200, 100, 100), # Red
            "fist": (100, 100, 200)       # Dark Blue
        }
        
        for gesture in gestures:
            animated_frames = self.create_animated_frames_from_color(colors[gesture], gesture)
            self.sprites[gesture] = animated_frames
        
        print("Created animated sprites for testing")
    
    def create_animated_frames_from_color(self, color: Tuple[int, int, int], gesture: str) -> List[np.ndarray]:
        """Create animated frames from a base color."""
        frames = []
        num_frames = 8
        
        for frame_idx in range(num_frames):
            # Create colored sprite
            sprite = np.ones((*self.sprite_size, 3), dtype=np.uint8)
            sprite[:, :] = color
            
            # Add animation effects
            if gesture == "idle":
                # Breathing effect
                scale = 1.0 + 0.05 * np.sin(frame_idx * 0.8)
                sprite = self.apply_scale(sprite, scale)
                
            elif gesture == "wave":
                # Waving effect
                angle = 20 * np.sin(frame_idx * 1.0)
                sprite = self.apply_rotation(sprite, angle)
                
            elif gesture == "thumbs_up":
                # Bounce effect
                offset_y = -15 * np.sin(frame_idx * 1.5)
                sprite = self.apply_translation(sprite, 0, offset_y)
                
            elif gesture == "point":
                # Side movement
                offset_x = 15 * np.sin(frame_idx * 1.3)
                sprite = self.apply_translation(sprite, offset_x, 0)
                
            elif gesture == "peace":
                # Sway effect
                angle = 10 * np.sin(frame_idx * 0.9)
                sprite = self.apply_rotation(sprite, angle)
                
            elif gesture == "open_hand":
                # Scale effect
                scale = 1.0 + 0.08 * np.sin(frame_idx * 1.2)
                sprite = self.apply_scale(sprite, scale)
                
            elif gesture == "fist":
                # Shake effect
                angle = 15 * np.sin(frame_idx * 1.6)
                sprite = self.apply_rotation(sprite, angle)
            
            frames.append(sprite)
        
        return frames
    
    def update_animation(self, gesture: str, confidence: float) -> str:
        """Update animation based on gesture."""
        current_time = time.time()
        
        # Map gesture to animation
        gesture_mapping = {
            "wave": "wave",
            "thumbs_up": "thumbs_up", 
            "point": "point",
            "peace": "peace",
            "open_hand": "open_hand",
            "fist": "fist",
            "no_hand": "idle",
            "unknown": "idle"
        }
        
        target_animation = gesture_mapping.get(gesture, "idle")
        
        # Check if we should change animation
        if (target_animation != self.current_animation and 
            confidence > self.config.get('confidence_threshold', 0.2) and
            current_time - self.last_animation_change > self.transition_duration):
            
            self.current_animation = target_animation
            self.animation_frame = 0
            self.last_animation_change = current_time
        
        # Update animation frame
        self.animation_time += self.animation_speed
        if self.current_animation in self.sprites:
            self.animation_frame = int(self.animation_time) % len(self.sprites[self.current_animation])
        
        return self.current_animation
    
    def get_current_frame(self) -> np.ndarray:
        """Get current animation frame."""
        if self.current_animation in self.sprites:
            frame_idx = int(self.animation_frame)
            if frame_idx < len(self.sprites[self.current_animation]):
                return self.sprites[self.current_animation][frame_idx]
        
        # Return default frame if animation not found
        return np.ones((*self.sprite_size, 3), dtype=np.uint8) * 128


class VRMAvatar(AvatarAnimation):
    """3D VRM avatar using VRM models."""
    
    def __init__(self, vrm_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize VRM avatar.
        
        Args:
            vrm_path: Path to VRM model file
            config: Animation configuration
        """
        super().__init__("3d")
        self.config = config or {}
        self.vrm_model = None
        self.animation_states = {}
        
        if vrm_path:
            self.load_vrm_model(vrm_path)
        else:
            self.create_placeholder_model()
    
    def load_vrm_model(self, vrm_path: str):
        """Load VRM model."""
        # Placeholder for VRM loading
        print(f"VRM loading not implemented yet: {vrm_path}")
        self.create_placeholder_model()
    
    def create_placeholder_model(self):
        """Create placeholder 3D model."""
        # Define animation states for different gestures
        self.animation_states = {
            "idle": {"color": (100, 150, 200), "scale": 1.0, "rotation": 0},
            "wave": {"color": (200, 100, 150), "scale": 1.1, "rotation": 15},
            "thumbs_up": {"color": (150, 200, 100), "scale": 1.05, "rotation": 0},
            "point": {"color": (200, 150, 100), "scale": 1.0, "rotation": 10},
            "peace": {"color": (200, 100, 200), "scale": 1.08, "rotation": 5},
            "open_hand": {"color": (200, 100, 100), "scale": 1.12, "rotation": 0},
            "fist": {"color": (100, 100, 200), "scale": 1.03, "rotation": 8}
        }
    
    def update_animation(self, gesture: str, confidence: float) -> str:
        """Update 3D animation based on gesture."""
        current_time = time.time()
        
        # Map gesture to animation
        gesture_mapping = {
            "wave": "wave",
            "thumbs_up": "thumbs_up",
            "point": "point", 
            "peace": "peace",
            "open_hand": "open_hand",
            "fist": "fist",
            "no_hand": "idle",
            "unknown": "idle"
        }
        
        target_animation = gesture_mapping.get(gesture, "idle")
        
        # Check if we should change animation
        if (target_animation != self.current_animation and 
            confidence > self.config.get('confidence_threshold', 0.6) and
            current_time - self.last_animation_change > self.transition_duration):
            
            self.current_animation = target_animation
            self.last_animation_change = current_time
        
        return self.current_animation
    
    def get_current_frame(self) -> np.ndarray:
        """Get current 3D animation frame as 2D image."""
        if self.current_animation in self.animation_states:
            state = self.animation_states[self.current_animation]
            
            # Create a simple 3D-like representation
            size = (256, 256)
            frame = np.ones((*size, 3), dtype=np.uint8)
            
            # Apply animation state
            color = state["color"]
            scale = state["scale"]
            rotation = state["rotation"]
            
            # Create a simple 3D-like shape
            center_x, center_y = size[1] // 2, size[0] // 2
            radius = int(50 * scale)
            
            # Draw main body
            cv2.circle(frame, (center_x, center_y), radius, color, -1)
            
            # Draw head
            head_y = center_y - int(40 * scale)
            cv2.circle(frame, (center_x, head_y), int(25 * scale), color, -1)
            
            # Apply rotation effect (simplified)
            if rotation != 0:
                # Create rotation matrix
                angle_rad = math.radians(rotation)
                cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
                
                # Apply simple rotation to the frame
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
                frame = cv2.warpAffine(frame, rotation_matrix, (w, h))
            
            return frame
        
        # Return default frame
        return np.ones((256, 256, 3), dtype=np.uint8) * 128


class AvatarManager:
    """Manages avatar animations and transitions."""
    
    def __init__(self, avatar_type: str = "2d", config: Optional[Dict] = None):
        """
        Initialize avatar manager.
        
        Args:
            avatar_type: "2d" or "3d"
            config: Configuration dictionary
        """
        self.avatar_type = avatar_type
        self.config = config or {}
        
        # Initialize avatar
        if avatar_type == "2d":
            self.avatar = SpriteBasedAvatar(config=config)
        elif avatar_type == "3d":
            self.avatar = VRMAvatar(config=config)
        else:
            raise ValueError(f"Unknown avatar type: {avatar_type}")
        
        # Animation state
        self.current_gesture = "no_hand"
        self.gesture_confidence = 0.0
        self.animation_history = []
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
    
    def update(self, gesture: str, confidence: float) -> np.ndarray:
        """
        Update avatar based on gesture detection.
        
        Args:
            gesture: Detected gesture
            confidence: Detection confidence
            
        Returns:
            Current avatar frame
        """
        # Update gesture state
        self.current_gesture = gesture
        self.gesture_confidence = confidence
        
        # Update animation
        current_animation = self.avatar.update_animation(gesture, confidence)
        
        # Get current frame
        frame = self.avatar.get_current_frame()
        
        # Add UI overlay
        frame = self.add_ui_overlay(frame, gesture, confidence, current_animation)
        
        # Update performance tracking
        self._update_fps()
        
        return frame
    
    def add_ui_overlay(self, frame: np.ndarray, gesture: str, confidence: float, 
                      animation: str) -> np.ndarray:
        """Add UI overlay to avatar frame."""
        # Convert to PIL for text rendering
        pil_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_frame)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add text overlay
        text_color = (255, 255, 255)
        bg_color = (0, 0, 0, 128)
        
        # Gesture info
        gesture_text = f"Gesture: {gesture}"
        confidence_text = f"Confidence: {confidence:.2f}"
        animation_text = f"Animation: {animation}"
        fps_text = f"FPS: {self.fps_counter:.1f}"
        
        # Draw background rectangles
        texts = [gesture_text, confidence_text, animation_text, fps_text]
        y_offset = 10
        
        for text in texts:
            bbox = draw.textbbox((10, y_offset), text, font=font)
            # Draw background
            draw.rectangle(bbox, fill=bg_color)
            # Draw text
            draw.text((10, y_offset), text, fill=text_color, font=font)
            y_offset += 25
        
        return np.array(pil_frame)
    
    def _update_fps(self):
        """Update FPS counter."""
        current_time = time.time()
        if current_time - self.last_fps_time > 0:
            self.fps_counter = 1.0 / (current_time - self.last_fps_time)
        self.last_fps_time = current_time
    
    def switch_avatar_type(self, avatar_type: str):
        """Switch between 2D and 3D avatars."""
        if avatar_type == self.avatar_type:
            return
        
        self.avatar_type = avatar_type
        
        if avatar_type == "2d":
            self.avatar = SpriteBasedAvatar(config=self.config)
        elif avatar_type == "3d":
            self.avatar = VRMAvatar(config=self.config)
        else:
            raise ValueError(f"Unknown avatar type: {avatar_type}")
        
        print(f"Switched to {avatar_type} avatar")
    
    def get_animation_info(self) -> Dict:
        """Get current animation information."""
        return {
            'avatar_type': self.avatar_type,
            'current_gesture': self.current_gesture,
            'gesture_confidence': self.gesture_confidence,
            'current_animation': self.avatar.current_animation,
            'fps': self.fps_counter
        }
    
    def reset(self):
        """Reset avatar to initial state."""
        self.avatar.reset_animation()
        self.current_gesture = "no_hand"
        self.gesture_confidence = 0.0
        self.animation_history = [] 