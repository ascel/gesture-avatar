#!/usr/bin/env python3
"""
Create a sample pre-trained gesture recognition model.
This script generates synthetic training data and trains a basic model.
"""

import os
import sys
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gesture_detection.models import FeatureBasedGestureModel

def generate_synthetic_gesture_data(num_samples=1000):
    """Generate synthetic gesture data for training."""
    gestures = ['fist', 'point', 'peace', 'open_hand', 'thumbs_up', 'wave']
    data = []
    labels = []
    
    for gesture in gestures:
        for _ in range(num_samples // len(gestures)):
            # Generate synthetic landmarks based on gesture
            landmarks = generate_gesture_landmarks(gesture)
            data.append(landmarks)
            labels.append(gesture)
    
    return data, labels

def generate_gesture_landmarks(gesture):
    """Generate synthetic landmarks for a specific gesture."""
    # Base hand position (wrist at center)
    base_x, base_y = 0.5, 0.5
    
    # Initialize landmarks (21 points for MediaPipe hands)
    landmarks = []
    
    # Wrist (landmark 0)
    landmarks.append({'x': base_x, 'y': base_y, 'z': 0.0})
    
    # Thumb landmarks (1-4)
    if gesture == 'thumbs_up':
        landmarks.extend([
            {'x': base_x - 0.05, 'y': base_y - 0.1, 'z': 0.0},  # Thumb CMC
            {'x': base_x - 0.08, 'y': base_y - 0.15, 'z': 0.0},  # Thumb MCP
            {'x': base_x - 0.1, 'y': base_y - 0.2, 'z': 0.0},   # Thumb IP
            {'x': base_x - 0.12, 'y': base_y - 0.25, 'z': 0.0}  # Thumb tip
        ])
    else:
        landmarks.extend([
            {'x': base_x - 0.05, 'y': base_y + 0.05, 'z': 0.0},
            {'x': base_x - 0.08, 'y': base_y + 0.08, 'z': 0.0},
            {'x': base_x - 0.1, 'y': base_y + 0.1, 'z': 0.0},
            {'x': base_x - 0.12, 'y': base_y + 0.12, 'z': 0.0}
        ])
    
    # Index finger landmarks (5-8)
    if gesture in ['point', 'peace', 'open_hand']:
        landmarks.extend([
            {'x': base_x + 0.02, 'y': base_y - 0.05, 'z': 0.0},  # Index MCP
            {'x': base_x + 0.03, 'y': base_y - 0.1, 'z': 0.0},   # Index PIP
            {'x': base_x + 0.04, 'y': base_y - 0.15, 'z': 0.0},  # Index DIP
            {'x': base_x + 0.05, 'y': base_y - 0.2, 'z': 0.0}   # Index tip
        ])
    else:
        landmarks.extend([
            {'x': base_x + 0.02, 'y': base_y + 0.05, 'z': 0.0},
            {'x': base_x + 0.03, 'y': base_y + 0.08, 'z': 0.0},
            {'x': base_x + 0.04, 'y': base_y + 0.1, 'z': 0.0},
            {'x': base_x + 0.05, 'y': base_y + 0.12, 'z': 0.0}
        ])
    
    # Middle finger landmarks (9-12)
    if gesture in ['peace', 'open_hand']:
        landmarks.extend([
            {'x': base_x + 0.04, 'y': base_y - 0.05, 'z': 0.0},  # Middle MCP
            {'x': base_x + 0.05, 'y': base_y - 0.1, 'z': 0.0},   # Middle PIP
            {'x': base_x + 0.06, 'y': base_y - 0.15, 'z': 0.0},  # Middle DIP
            {'x': base_x + 0.07, 'y': base_y - 0.2, 'z': 0.0}   # Middle tip
        ])
    else:
        landmarks.extend([
            {'x': base_x + 0.04, 'y': base_y + 0.05, 'z': 0.0},
            {'x': base_x + 0.05, 'y': base_y + 0.08, 'z': 0.0},
            {'x': base_x + 0.06, 'y': base_y + 0.1, 'z': 0.0},
            {'x': base_x + 0.07, 'y': base_y + 0.12, 'z': 0.0}
        ])
    
    # Ring finger landmarks (13-16)
    if gesture == 'open_hand':
        landmarks.extend([
            {'x': base_x + 0.06, 'y': base_y - 0.05, 'z': 0.0},  # Ring MCP
            {'x': base_x + 0.07, 'y': base_y - 0.1, 'z': 0.0},   # Ring PIP
            {'x': base_x + 0.08, 'y': base_y - 0.15, 'z': 0.0},  # Ring DIP
            {'x': base_x + 0.09, 'y': base_y - 0.2, 'z': 0.0}   # Ring tip
        ])
    else:
        landmarks.extend([
            {'x': base_x + 0.06, 'y': base_y + 0.05, 'z': 0.0},
            {'x': base_x + 0.07, 'y': base_y + 0.08, 'z': 0.0},
            {'x': base_x + 0.08, 'y': base_y + 0.1, 'z': 0.0},
            {'x': base_x + 0.09, 'y': base_y + 0.12, 'z': 0.0}
        ])
    
    # Pinky finger landmarks (17-20)
    if gesture == 'open_hand':
        landmarks.extend([
            {'x': base_x + 0.08, 'y': base_y - 0.05, 'z': 0.0},  # Pinky MCP
            {'x': base_x + 0.09, 'y': base_y - 0.1, 'z': 0.0},   # Pinky PIP
            {'x': base_x + 0.1, 'y': base_y - 0.15, 'z': 0.0},   # Pinky DIP
            {'x': base_x + 0.11, 'y': base_y - 0.2, 'z': 0.0}   # Pinky tip
        ])
    else:
        landmarks.extend([
            {'x': base_x + 0.08, 'y': base_y + 0.05, 'z': 0.0},
            {'x': base_x + 0.09, 'y': base_y + 0.08, 'z': 0.0},
            {'x': base_x + 0.1, 'y': base_y + 0.1, 'z': 0.0},
            {'x': base_x + 0.11, 'y': base_y + 0.12, 'z': 0.0}
        ])
    
    # Add some noise for realism
    for landmark in landmarks:
        landmark['x'] += np.random.normal(0, 0.01)
        landmark['y'] += np.random.normal(0, 0.01)
        landmark['z'] += np.random.normal(0, 0.005)
    
    return landmarks

def create_sample_model():
    """Create and save a sample gesture recognition model."""
    print("Creating sample gesture recognition model...")
    
    # Create models directory
    models_dir = Path("data/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    print("Generating synthetic training data...")
    data, labels = generate_synthetic_gesture_data(num_samples=2000)
    
    # Convert to feature-based format
    model = FeatureBasedGestureModel()
    
    # Extract features
    print("Extracting features...")
    features = []
    for landmarks in data:
        feature_dict = model.extract_features(landmarks)
        features.append(list(feature_dict.values()))
    
    X = np.array(features)
    y = np.array(labels)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print("Training model...")
    history = model.train(
        X_train_scaled, y_train_encoded,
        X_val_scaled, y_val_encoded,
        epochs=50, batch_size=32
    )
    
    # Save model and preprocessing artifacts
    model_path = models_dir / "sample_gesture_model.h5"
    print(f"Saving model to {model_path}...")
    
    model.save_model(str(model_path))
    
    # Save preprocessing artifacts
    joblib.dump(scaler, models_dir / "feature_scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    
    # Save class mapping
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(models_dir / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Evaluate model
    print("Evaluating model...")
    val_predictions = model.model.predict(X_val_scaled)
    val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == y_val_encoded)
    
    print(f"Validation accuracy: {val_accuracy:.3f}")
    print(f"Model saved to: {model_path}")
    print(f"Supported gestures: {list(label_encoder.classes_)}")
    
    return str(model_path)

def create_config_file():
    """Create a configuration file for the sample model."""
    config = {
        "gesture_detection": {
            "model_type": "feature",
            "model_path": "data/models/sample_gesture_model.h5",
            "confidence_threshold": 0.7,
            "smoothing_window": 5
        },
        "avatar_animation": {
            "avatar_type": "2d",
            "animation_speed": 1.0,
            "transition_smoothness": 0.3
        },
        "streaming": {
            "fps": 30,
            "resolution": [640, 480],
            "virtual_camera": False
        }
    }
    
    config_path = Path("config/config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")

if __name__ == "__main__":
    print("=== Sample Gesture Recognition Model Creator ===")
    
    # Create sample model
    model_path = create_sample_model()
    
    # Create configuration
    create_config_file()
    
    print("\n=== Setup Complete ===")
    print("You can now run the app with:")
    print("python src/main.py")
    print("\nThe app will use the sample model for gesture recognition!") 