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
        self.animation_speed = 1.0
        self.transition_duration = 0.5  # seconds
        self.last_animation_change = time.time()
        
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


class SpriteBasedAvatar(AvatarAnimation):
    """2D sprite-based avatar using spritesheets."""
    
    def __init__(self, spritesheet_path: str = None, config: Dict = None):
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
        self.sprite_size = (64, 64)  # Default sprite size
        self.frames_per_animation = 8  # Default frames per animation
        
        if spritesheet_path:
            self.load_spritesheet(spritesheet_path)
        else:
            self.create_default_sprites()
    
    def load_spritesheet(self, spritesheet_path: str):
        """Load spritesheet and extract sprites."""
        try:
            spritesheet = Image.open(spritesheet_path)
            self.spritesheet = np.array(spritesheet)
            
            # Extract sprites for each animation
            animations = ["idle", "wave", "thumbs_up", "point", "clap", "fist"]
            rows = len(animations)
            cols = self.frames_per_animation
            
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
            self.create_default_sprites()
    
    def create_default_sprites(self):
        """Create default colored sprites for testing."""
        animations = ["idle", "wave", "thumbs_up", "point", "clap", "fist"]
        colors = {
            "idle": (100, 150, 200),      # Blue
            "wave": (200, 100, 150),      # Pink
            "thumbs_up": (150, 200, 100), # Green
            "point": (200, 150, 100),     # Orange
            "clap": (200, 100, 100),      # Red
            "fist": (100, 100, 200)       # Purple
        }
        
        for animation in animations:
            animation_sprites = []
            for frame in range(self.frames_per_animation):
                # Create colored sprite
                sprite = np.ones((*self.sprite_size, 3), dtype=np.uint8)
                color = colors[animation]
                sprite[:, :] = color
                
                # Add some animation variation
                if frame > 0:
                    # Add a simple animation effect
                    offset = int(frame * 2)
                    sprite[offset:offset+10, :] = [255, 255, 255]  # White stripe
                
                animation_sprites.append(sprite)
            
            self.sprites[animation] = animation_sprites
        
        print("Created default sprites for testing")
    
    def update_animation(self, gesture: str, confidence: float) -> str:
        """Update animation based on gesture."""
        current_time = time.time()
        
        # Map gesture to animation
        gesture_mapping = {
            "wave": "wave",
            "thumbs_up": "thumbs_up", 
            "point": "point",
            "clap": "clap",
            "fist": "fist",
            "no_hand": "idle",
            "unknown": "idle"
        }
        
        target_animation = gesture_mapping.get(gesture, "idle")
        
        # Check if we should change animation
        if (target_animation != self.current_animation and 
            confidence > self.config.get('confidence_threshold', 0.7) and
            current_time - self.last_animation_change > self.transition_duration):
            
            self.current_animation = target_animation
            self.animation_frame = 0
            self.last_animation_change = current_time
        
        # Update animation frame
        if self.current_animation in self.sprites:
            self.animation_frame = (self.animation_frame + self.animation_speed) % len(self.sprites[self.current_animation])
        
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
    
    def __init__(self, vrm_path: str = None, config: Dict = None):
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
        """Load VRM model (placeholder for now)."""
        # In a real implementation, this would load a VRM model
        # For now, we'll create a placeholder
        self.create_placeholder_model()
    
    def create_placeholder_model(self):
        """Create placeholder 3D avatar for testing."""
        # Create simple 3D-like representation using 2D graphics
        self.animation_states = {
            "idle": {"rotation": 0, "scale": 1.0, "color": (100, 150, 200)},
            "wave": {"rotation": 15, "scale": 1.1, "color": (200, 100, 150)},
            "thumbs_up": {"rotation": -10, "scale": 1.05, "color": (150, 200, 100)},
            "point": {"rotation": 5, "scale": 1.0, "color": (200, 150, 100)},
            "clap": {"rotation": 0, "scale": 1.2, "color": (200, 100, 100)},
            "fist": {"rotation": 0, "scale": 0.9, "color": (100, 100, 200)}
        }
    
    def update_animation(self, gesture: str, confidence: float) -> str:
        """Update 3D animation based on gesture."""
        current_time = time.time()
        
        # Map gesture to animation
        gesture_mapping = {
            "wave": "wave",
            "thumbs_up": "thumbs_up",
            "point": "point", 
            "clap": "clap",
            "fist": "fist",
            "no_hand": "idle",
            "unknown": "idle"
        }
        
        target_animation = gesture_mapping.get(gesture, "idle")
        
        # Check if we should change animation
        if (target_animation != self.current_animation and 
            confidence > self.config.get('confidence_threshold', 0.7) and
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
    
    def __init__(self, avatar_type: str = "2d", config: Dict = None):
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