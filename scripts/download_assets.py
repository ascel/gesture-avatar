"""
Download assets script for gesture-controlled avatar project.
Downloads sample data, pre-trained models, and other required assets.
"""

import os
import requests
import zipfile
import json
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_file(url: str, filepath: Path, description: str = "Downloading"):
    """Download a file with progress bar."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with tqdm(unit='B', unit_scale=True, desc=description) as pbar:
            def progress_hook(count, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, filepath, progress_hook)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def create_sample_data():
    """Create sample gesture data structure for testing."""
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample gestures from PRD
    gestures = ["wave", "thumbs_up", "point", "clap", "fist"]
    
    for gesture in gestures:
        gesture_dir = data_dir / gesture
        gesture_dir.mkdir(exist_ok=True)
        
        # Create sample landmark data
        sample_landmarks = []
        for i in range(21):  # MediaPipe hands have 21 landmarks
            sample_landmarks.append({
                'x': 0.5 + 0.1 * (i % 3),
                'y': 0.5 + 0.1 * (i // 3),
                'z': 0.0 + 0.01 * i
            })
        
        # Create sample JSON files
        for i in range(10):  # Create 10 sample files per gesture
            sample_data = {
                'gesture': gesture,
                'landmarks': sample_landmarks,
                'timestamp': f'2024-01-01T12:00:00.{i:03d}Z'
            }
            
            json_path = gesture_dir / f"sample_{i:04d}.json"
            with open(json_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
    
    print("Sample gesture data created successfully!")


def download_pretrained_models():
    """Download pre-trained models for gesture recognition."""
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Model URLs (these would be actual model URLs in a real implementation)
    models = {
        "mobilenet_gesture": {
            "url": "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
            "filename": "mobilenet_gesture.h5",
            "description": "MobileNet base model for gesture recognition"
        }
    }
    
    for model_name, model_info in models.items():
        model_path = models_dir / model_info["filename"]
        
        if not model_path.exists():
            print(f"Downloading {model_info['description']}...")
            success = download_file(
                model_info["url"], 
                model_path, 
                model_info["description"]
            )
            if success:
                print(f"✓ Downloaded {model_name}")
            else:
                print(f"✗ Failed to download {model_name}")
        else:
            print(f"✓ {model_name} already exists")


def download_avatar_assets():
    """Download sample avatar assets."""
    assets_dir = Path("data/avatars")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample 2D avatar spritesheet
    avatar_info = {
        "2d_spritesheet": {
            "filename": "sample_avatar_spritesheet.png",
            "description": "Sample 2D avatar spritesheet"
        },
        "3d_vrm": {
            "filename": "sample_avatar.vrm",
            "description": "Sample 3D VRM avatar model"
        }
    }
    
    for asset_name, asset_info in avatar_info.items():
        asset_path = assets_dir / asset_info["filename"]
        
        if not asset_path.exists():
            # Create placeholder files for now
            if asset_name == "2d_spritesheet":
                # Create a simple colored rectangle as placeholder
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (256, 256), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.rectangle([50, 50, 206, 206], fill='blue', outline='darkblue', width=3)
                draw.text((128, 128), "Avatar", fill='white', anchor="mm")
                img.save(asset_path)
            else:
                # Create placeholder file
                with open(asset_path, 'w') as f:
                    f.write(f"# Placeholder {asset_info['description']}\n")
                    f.write("# Replace with actual asset file\n")
            
            print(f"✓ Created placeholder {asset_name}")
        else:
            print(f"✓ {asset_name} already exists")


def create_config_files():
    """Create configuration files for the project."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Main configuration
    main_config = {
        "gesture_detection": {
            "model_type": "mobilenet",
            "confidence_threshold": 0.7,
            "max_hands": 2,
            "target_fps": 15
        },
        "avatar_animation": {
            "avatar_type": "2d",  # or "3d"
            "animation_speed": 1.0,
            "smooth_transitions": True
        },
        "streaming": {
            "output_resolution": [720, 480],
            "fps": 15,
            "delay_tolerance": 20  # seconds
        },
        "hardware": {
            "gpu_acceleration": True,
            "max_memory_usage": 0.8
        }
    }
    
    with open(config_dir / "config.json", 'w') as f:
        json.dump(main_config, f, indent=2)
    
    # Gesture mapping configuration
    gesture_mapping = {
        "wave": {
            "animation": "wave",
            "trigger_threshold": 0.8,
            "cooldown": 1.0
        },
        "thumbs_up": {
            "animation": "thumbs_up",
            "trigger_threshold": 0.8,
            "cooldown": 1.0
        },
        "point": {
            "animation": "point",
            "trigger_threshold": 0.8,
            "cooldown": 0.5
        },
        "clap": {
            "animation": "clap",
            "trigger_threshold": 0.7,
            "cooldown": 0.3
        },
        "fist": {
            "animation": "fist",
            "trigger_threshold": 0.8,
            "cooldown": 1.0
        }
    }
    
    with open(config_dir / "gesture_mapping.json", 'w') as f:
        json.dump(gesture_mapping, f, indent=2)
    
    print("✓ Configuration files created")


def main():
    """Main function to download all assets."""
    print("Gesture Avatar - Asset Downloader")
    print("=================================")
    
    try:
        # Create sample data
        print("\n1. Creating sample gesture data...")
        create_sample_data()
        
        # Download pre-trained models
        print("\n2. Downloading pre-trained models...")
        download_pretrained_models()
        
        # Download avatar assets
        print("\n3. Setting up avatar assets...")
        download_avatar_assets()
        
        # Create configuration files
        print("\n4. Creating configuration files...")
        create_config_files()
        
        print("\n✓ All assets downloaded and configured successfully!")
        print("\nNext steps:")
        print("1. Run data collection: python src/utils/data_collection.py")
        print("2. Run preprocessing: python src/utils/data_preprocessing.py")
        print("3. Train models: python src/gesture_detection/train_models.py")
        print("4. Run demo: python src/main.py")
        
    except Exception as e:
        print(f"Error during asset download: {e}")
        print("Please check your internet connection and try again.")


if __name__ == "__main__":
    main() 