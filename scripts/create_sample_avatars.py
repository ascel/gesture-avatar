#!/usr/bin/env python3
"""
Create sample avatar animations for the gesture avatar system.
This script generates 2D sprites and 3D-style avatars with animations.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def create_2d_avatar_sprites():
    """Create 2D avatar sprites for different gestures."""
    print("Creating 2D avatar sprites...")
    
    # Create avatars directory
    avatars_dir = Path("data/avatars/2d")
    avatars_dir.mkdir(parents=True, exist_ok=True)
    
    # Avatar configurations
    avatars = {
        "robot": {
            "color": (100, 150, 255),  # Blue
            "size": (200, 200),
            "gestures": ["idle", "fist", "point", "peace", "open_hand", "thumbs_up", "wave"]
        },
        "alien": {
            "color": (255, 100, 150),  # Pink
            "size": (200, 200),
            "gestures": ["idle", "fist", "point", "peace", "open_hand", "thumbs_up", "wave"]
        },
        "ghost": {
            "color": (200, 200, 255),  # Light blue
            "size": (200, 200),
            "gestures": ["idle", "fist", "point", "peace", "open_hand", "thumbs_up", "wave"]
        }
    }
    
    for avatar_name, config in avatars.items():
        print(f"Creating {avatar_name} avatar...")
        avatar_dir = avatars_dir / avatar_name
        avatar_dir.mkdir(exist_ok=True)
        
        # Create sprites for each gesture
        for gesture in config["gestures"]:
            sprite = create_2d_sprite(avatar_name, gesture, config)
            sprite_path = avatar_dir / f"{gesture}.png"
            sprite.save(sprite_path)
            print(f"  Created {gesture}.png")
    
    # Create spritesheet
    create_spritesheet(avatars_dir)
    
    print("2D avatars created successfully!")

def create_2d_sprite(avatar_type, gesture, config):
    """Create a 2D sprite for a specific avatar and gesture."""
    width, height = config["size"]
    color = config["color"]
    
    # Create base image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw avatar based on type
    if avatar_type == "robot":
        draw_robot_avatar(draw, gesture, width, height, color)
    elif avatar_type == "alien":
        draw_alien_avatar(draw, gesture, width, height, color)
    elif avatar_type == "ghost":
        draw_ghost_avatar(draw, gesture, width, height, color)
    
    return img

def draw_robot_avatar(draw, gesture, width, height, color):
    """Draw a robot avatar."""
    center_x, center_y = width // 2, height // 2
    
    # Body (rectangle)
    body_width, body_height = 80, 100
    body_x = center_x - body_width // 2
    body_y = center_y - body_height // 2
    draw.rectangle([body_x, body_y, body_x + body_width, body_y + body_height], 
                  fill=color, outline=(50, 50, 50), width=3)
    
    # Head (circle)
    head_radius = 30
    head_x, head_y = center_x, body_y - head_radius - 10
    draw.ellipse([head_x - head_radius, head_y - head_radius, 
                  head_x + head_radius, head_y + head_radius], 
                 fill=color, outline=(50, 50, 50), width=3)
    
    # Eyes
    eye_radius = 5
    draw.ellipse([head_x - 15 - eye_radius, head_y - eye_radius,
                  head_x - 15 + eye_radius, head_y + eye_radius], fill=(255, 255, 255))
    draw.ellipse([head_x + 15 - eye_radius, head_y - eye_radius,
                  head_x + 15 + eye_radius, head_y + eye_radius], fill=(255, 255, 255))
    
    # Arms based on gesture
    draw_robot_arms(draw, gesture, center_x, body_y, body_width, body_height, color)
    
    # Legs
    leg_width, leg_height = 20, 40
    draw.rectangle([center_x - 30, body_y + body_height, 
                   center_x - 30 + leg_width, body_y + body_height + leg_height], 
                  fill=color, outline=(50, 50, 50), width=2)
    draw.rectangle([center_x + 10, body_y + body_height, 
                   center_x + 10 + leg_width, body_y + body_height + leg_height], 
                  fill=color, outline=(50, 50, 50), width=2)

def draw_robot_arms(draw, gesture, center_x, body_y, body_width, body_height, color):
    """Draw robot arms based on gesture."""
    arm_width, arm_height = 15, 40
    
    if gesture == "fist":
        # Arms down, fists
        draw.rectangle([center_x - body_width//2 - arm_width, body_y + 10,
                       center_x - body_width//2, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        draw.rectangle([center_x + body_width//2, body_y + 10,
                       center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        
        # Fists
        draw.ellipse([center_x - body_width//2 - arm_width - 10, body_y + 10 + arm_height,
                     center_x - body_width//2 - arm_width + 10, body_y + 10 + arm_height + 20], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.ellipse([center_x + body_width//2 + arm_width - 10, body_y + 10 + arm_height,
                     center_x + body_width//2 + arm_width + 10, body_y + 10 + arm_height + 20], 
                    fill=color, outline=(50, 50, 50), width=2)
    
    elif gesture == "point":
        # Left arm pointing
        draw.rectangle([center_x - body_width//2 - arm_width, body_y + 10,
                       center_x - body_width//2, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        # Pointing finger
        draw.line([center_x - body_width//2 - arm_width, body_y + 10 + arm_height,
                  center_x - body_width//2 - arm_width - 20, body_y + 10 + arm_height - 10], 
                 fill=(255, 255, 255), width=3)
        
        # Right arm down
        draw.rectangle([center_x + body_width//2, body_y + 10,
                       center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
    
    elif gesture == "peace":
        # Both arms up in peace sign
        draw.rectangle([center_x - body_width//2 - arm_width, body_y - 20,
                       center_x - body_width//2, body_y - 20 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        draw.rectangle([center_x + body_width//2, body_y - 20,
                       center_x + body_width//2 + arm_width, body_y - 20 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        
        # Peace signs
        draw.line([center_x - body_width//2 - arm_width, body_y - 20,
                  center_x - body_width//2 - arm_width - 15, body_y - 20 - 15], 
                 fill=(255, 255, 255), width=3)
        draw.line([center_x - body_width//2 - arm_width, body_y - 20,
                  center_x - body_width//2 - arm_width - 15, body_y - 20 + 15], 
                 fill=(255, 255, 255), width=3)
        
        draw.line([center_x + body_width//2 + arm_width, body_y - 20,
                  center_x + body_width//2 + arm_width + 15, body_y - 20 - 15], 
                 fill=(255, 255, 255), width=3)
        draw.line([center_x + body_width//2 + arm_width, body_y - 20,
                  center_x + body_width//2 + arm_width + 15, body_y - 20 + 15], 
                 fill=(255, 255, 255), width=3)
    
    elif gesture == "open_hand":
        # Both arms up, open hands
        draw.rectangle([center_x - body_width//2 - arm_width, body_y - 20,
                       center_x - body_width//2, body_y - 20 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        draw.rectangle([center_x + body_width//2, body_y - 20,
                       center_x + body_width//2 + arm_width, body_y - 20 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        
        # Open hands (circles)
        draw.ellipse([center_x - body_width//2 - arm_width - 15, body_y - 20 - 15,
                     center_x - body_width//2 - arm_width + 15, body_y - 20 + 15], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.ellipse([center_x + body_width//2 + arm_width - 15, body_y - 20 - 15,
                     center_x + body_width//2 + arm_width + 15, body_y - 20 + 15], 
                    fill=color, outline=(50, 50, 50), width=2)
    
    elif gesture == "thumbs_up":
        # Left arm with thumbs up
        draw.rectangle([center_x - body_width//2 - arm_width, body_y + 10,
                       center_x - body_width//2, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        # Thumbs up
        draw.ellipse([center_x - body_width//2 - arm_width - 10, body_y + 10 + arm_height,
                     center_x - body_width//2 - arm_width + 10, body_y + 10 + arm_height + 20], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.line([center_x - body_width//2 - arm_width, body_y + 10 + arm_height,
                  center_x - body_width//2 - arm_width, body_y + 10 + arm_height - 15], 
                 fill=(255, 255, 255), width=3)
        
        # Right arm down
        draw.rectangle([center_x + body_width//2, body_y + 10,
                       center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
    
    elif gesture == "wave":
        # Right arm waving
        draw.rectangle([center_x + body_width//2, body_y - 10,
                       center_x + body_width//2 + arm_width, body_y - 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        # Waving hand
        draw.ellipse([center_x + body_width//2 + arm_width - 10, body_y - 10,
                     center_x + body_width//2 + arm_width + 10, body_y - 10 + 20], 
                    fill=color, outline=(50, 50, 50), width=2)
        
        # Left arm down
        draw.rectangle([center_x - body_width//2 - arm_width, body_y + 10,
                       center_x - body_width//2, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
    
    else:  # idle
        # Arms down
        draw.rectangle([center_x - body_width//2 - arm_width, body_y + 10,
                       center_x - body_width//2, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)
        draw.rectangle([center_x + body_width//2, body_y + 10,
                       center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                      fill=color, outline=(50, 50, 50), width=2)

def draw_alien_avatar(draw, gesture, width, height, color):
    """Draw an alien avatar."""
    center_x, center_y = width // 2, height // 2
    
    # Body (oval)
    body_width, body_height = 70, 90
    body_x = center_x - body_width // 2
    body_y = center_y - body_height // 2
    draw.ellipse([body_x, body_y, body_x + body_width, body_y + body_height], 
                fill=color, outline=(50, 50, 50), width=3)
    
    # Head (large oval)
    head_width, head_height = 60, 80
    head_x, head_y = center_x - head_width // 2, body_y - head_height - 5
    draw.ellipse([head_x, head_y, head_x + head_width, head_y + head_height], 
                fill=color, outline=(50, 50, 50), width=3)
    
    # Large eyes
    eye_width, eye_height = 15, 20
    draw.ellipse([head_x + 10 - eye_width//2, head_y + 20 - eye_height//2,
                  head_x + 10 + eye_width//2, head_y + 20 + eye_height//2], 
                fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse([head_x + 50 - eye_width//2, head_y + 20 - eye_height//2,
                  head_x + 50 + eye_width//2, head_y + 20 + eye_height//2], 
                fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Antennae
    draw.line([head_x + 15, head_y, head_x + 15, head_y - 20], fill=color, width=3)
    draw.ellipse([head_x + 15 - 5, head_y - 25, head_x + 15 + 5, head_y - 15], 
                fill=(255, 255, 0))
    draw.line([head_x + 45, head_y, head_x + 45, head_y - 20], fill=color, width=3)
    draw.ellipse([head_x + 45 - 5, head_y - 25, head_x + 45 + 5, head_y - 15], 
                fill=(255, 255, 0))
    
    # Arms based on gesture (simplified for alien)
    draw_alien_arms(draw, gesture, center_x, body_y, body_width, body_height, color)
    
    # Legs (simple)
    leg_width, leg_height = 15, 30
    draw.ellipse([center_x - 25, body_y + body_height, 
                  center_x - 25 + leg_width, body_y + body_height + leg_height], 
                fill=color, outline=(50, 50, 50), width=2)
    draw.ellipse([center_x + 10, body_y + body_height, 
                  center_x + 10 + leg_width, body_y + body_height + leg_height], 
                fill=color, outline=(50, 50, 50), width=2)

def draw_alien_arms(draw, gesture, center_x, body_y, body_width, body_height, color):
    """Draw alien arms based on gesture."""
    arm_width, arm_height = 12, 35
    
    if gesture == "fist":
        # Arms down with fists
        draw.ellipse([center_x - body_width//2 - arm_width, body_y + 10,
                     center_x - body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.ellipse([center_x + body_width//2 - arm_width, body_y + 10,
                     center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)
        
        # Fists
        draw.ellipse([center_x - body_width//2 - arm_width - 8, body_y + 10 + arm_height,
                     center_x - body_width//2 - arm_width + 8, body_y + 10 + arm_height + 16], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.ellipse([center_x + body_width//2 + arm_width - 8, body_y + 10 + arm_height,
                     center_x + body_width//2 + arm_width + 8, body_y + 10 + arm_height + 16], 
                    fill=color, outline=(50, 50, 50), width=2)
    
    elif gesture == "point":
        # Left arm pointing
        draw.ellipse([center_x - body_width//2 - arm_width, body_y + 10,
                     center_x - body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.line([center_x - body_width//2 - arm_width, body_y + 10 + arm_height,
                  center_x - body_width//2 - arm_width - 15, body_y + 10 + arm_height - 8], 
                 fill=(255, 255, 255), width=3)
        
        # Right arm down
        draw.ellipse([center_x + body_width//2 - arm_width, body_y + 10,
                     center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)
    
    else:  # Default arms down
        draw.ellipse([center_x - body_width//2 - arm_width, body_y + 10,
                     center_x - body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)
        draw.ellipse([center_x + body_width//2 - arm_width, body_y + 10,
                     center_x + body_width//2 + arm_width, body_y + 10 + arm_height], 
                    fill=color, outline=(50, 50, 50), width=2)

def draw_ghost_avatar(draw, gesture, width, height, color):
    """Draw a ghost avatar."""
    center_x, center_y = width // 2, height // 2
    
    # Ghost body (rounded rectangle with wavy bottom)
    body_width, body_height = 80, 100
    body_x = center_x - body_width // 2
    body_y = center_y - body_height // 2
    
    # Draw main body
    draw.ellipse([body_x, body_y, body_x + body_width, body_y + body_height - 20], 
                fill=color, outline=(50, 50, 50), width=3)
    
    # Wavy bottom
    for i in range(0, body_width, 10):
        draw.ellipse([body_x + i, body_y + body_height - 30, 
                     body_x + i + 10, body_y + body_height - 10], 
                    fill=color, outline=(50, 50, 50), width=2)
    
    # Eyes
    eye_radius = 8
    draw.ellipse([center_x - 20 - eye_radius, center_y - 20 - eye_radius,
                  center_x - 20 + eye_radius, center_y - 20 + eye_radius], 
                fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse([center_x + 20 - eye_radius, center_y - 20 - eye_radius,
                  center_x + 20 + eye_radius, center_y - 20 + eye_radius], 
                fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    
    # Pupils
    pupil_radius = 3
    draw.ellipse([center_x - 20 - pupil_radius, center_y - 20 - pupil_radius,
                  center_x - 20 + pupil_radius, center_y - 20 + pupil_radius], 
                fill=(0, 0, 0))
    draw.ellipse([center_x + 20 - pupil_radius, center_y - 20 - pupil_radius,
                  center_x + 20 + pupil_radius, center_y - 20 + pupil_radius], 
                fill=(0, 0, 0))
    
    # Mouth
    if gesture == "fist":
        # Frown
        draw.arc([center_x - 15, center_y, center_x + 15, center_y + 20], 
                0, 180, fill=(0, 0, 0), width=2)
    elif gesture in ["peace", "open_hand", "thumbs_up"]:
        # Smile
        draw.arc([center_x - 15, center_y - 10, center_x + 15, center_y + 10], 
                180, 360, fill=(0, 0, 0), width=2)
    else:
        # Neutral
        draw.line([center_x - 10, center_y + 10, center_x + 10, center_y + 10], 
                 fill=(0, 0, 0), width=2)

def create_spritesheet(avatars_dir):
    """Create a spritesheet for easier loading."""
    print("Creating spritesheet...")
    
    # Get all avatar directories
    avatar_dirs = [d for d in avatars_dir.iterdir() if d.is_dir()]
    
    for avatar_dir in avatar_dirs:
        avatar_name = avatar_dir.name
        sprite_files = list(avatar_dir.glob("*.png"))
        
        if not sprite_files:
            continue
        
        # Load first sprite to get dimensions
        first_sprite = Image.open(sprite_files[0])
        sprite_width, sprite_height = first_sprite.size
        
        # Create spritesheet
        num_sprites = len(sprite_files)
        spritesheet_width = sprite_width * num_sprites
        spritesheet_height = sprite_height
        
        spritesheet = Image.new('RGBA', (spritesheet_width, spritesheet_height), (0, 0, 0, 0))
        
        # Place sprites horizontally
        for i, sprite_file in enumerate(sorted(sprite_files)):
            sprite = Image.open(sprite_file)
            spritesheet.paste(sprite, (i * sprite_width, 0))
        
        # Save spritesheet
        spritesheet_path = avatar_dir / "spritesheet.png"
        spritesheet.save(spritesheet_path)
        
        # Create metadata
        metadata = {
            "sprite_width": sprite_width,
            "sprite_height": sprite_height,
            "num_sprites": num_sprites,
            "gestures": [f.stem for f in sorted(sprite_files)]
        }
        
        metadata_path = avatar_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Created spritesheet for {avatar_name}")

def create_3d_style_avatars():
    """Create 3D-style avatar configurations."""
    print("Creating 3D-style avatar configurations...")
    
    avatars_dir = Path("data/avatars/3d")
    avatars_dir.mkdir(parents=True, exist_ok=True)
    
    # Create VRM-style avatar configurations
    avatars = {
        "anime_girl": {
            "type": "vrm",
            "model_path": "sample_models/anime_girl.vrm",
            "animations": {
                "idle": {"blend_shape": "idle", "duration": 2.0},
                "fist": {"blend_shape": "angry", "duration": 0.5},
                "point": {"blend_shape": "point", "duration": 0.3},
                "peace": {"blend_shape": "peace", "duration": 0.5},
                "open_hand": {"blend_shape": "happy", "duration": 0.5},
                "thumbs_up": {"blend_shape": "thumbs_up", "duration": 0.5},
                "wave": {"blend_shape": "wave", "duration": 1.0}
            },
            "colors": {
                "hair": "#8B4513",
                "eyes": "#4169E1",
                "skin": "#FFE4C4"
            }
        },
        "robot_3d": {
            "type": "vrm",
            "model_path": "sample_models/robot_3d.vrm",
            "animations": {
                "idle": {"blend_shape": "idle", "duration": 1.0},
                "fist": {"blend_shape": "combat", "duration": 0.3},
                "point": {"blend_shape": "point", "duration": 0.2},
                "peace": {"blend_shape": "peace", "duration": 0.5},
                "open_hand": {"blend_shape": "open", "duration": 0.3},
                "thumbs_up": {"blend_shape": "approve", "duration": 0.5},
                "wave": {"blend_shape": "greet", "duration": 0.8}
            },
            "colors": {
                "body": "#C0C0C0",
                "eyes": "#00FF00",
                "accents": "#FF4500"
            }
        }
    }
    
    for avatar_name, config in avatars.items():
        avatar_dir = avatars_dir / avatar_name
        avatar_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_path = avatar_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  Created 3D avatar config for {avatar_name}")

def create_avatar_config():
    """Create main avatar configuration file."""
    config = {
        "available_avatars": {
            "2d": {
                "robot": {
                    "name": "Robot",
                    "description": "A friendly robot avatar",
                    "path": "data/avatars/2d/robot",
                    "type": "spritesheet"
                },
                "alien": {
                    "name": "Alien",
                    "description": "A cute alien avatar",
                    "path": "data/avatars/2d/alien",
                    "type": "spritesheet"
                },
                "ghost": {
                    "name": "Ghost",
                    "description": "A friendly ghost avatar",
                    "path": "data/avatars/2d/ghost",
                    "type": "spritesheet"
                }
            },
            "3d": {
                "anime_girl": {
                    "name": "Anime Girl",
                    "description": "A 3D anime-style avatar",
                    "path": "data/avatars/3d/anime_girl",
                    "type": "vrm"
                },
                "robot_3d": {
                    "name": "3D Robot",
                    "description": "A 3D robot avatar",
                    "path": "data/avatars/3d/robot_3d",
                    "type": "vrm"
                }
            }
        },
        "default_avatar": "robot",
        "default_type": "2d",
        "animation_settings": {
            "transition_speed": 0.3,
            "idle_animation": True,
            "smooth_transitions": True
        }
    }
    
    config_path = Path("data/avatars/avatar_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Avatar configuration saved to: {config_path}")

if __name__ == "__main__":
    print("=== Sample Avatar Animation Creator ===")
    
    # Create 2D avatars
    create_2d_avatar_sprites()
    
    # Create 3D avatar configurations
    create_3d_style_avatars()
    
    # Create main configuration
    create_avatar_config()
    
    print("\n=== Avatar Creation Complete ===")
    print("Created avatars:")
    print("  2D: robot, alien, ghost")
    print("  3D: anime_girl, robot_3d")
    print("\nYou can now use these avatars in the gesture avatar app!") 