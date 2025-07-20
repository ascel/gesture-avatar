#!/usr/bin/env python3
"""
Dataset Expansion Script

This script expands the gesture recognition dataset by:
1. Downloading and integrating online datasets
2. Implementing data augmentation techniques
3. Creating synthetic data
4. Collecting additional real data

Usage:
    python scripts/expand_dataset.py
"""

import json
import numpy as np
import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
import cv2
import mediapipe as mp
from sklearn.utils import resample
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DatasetExpander:
    def __init__(self, data_path="../data"):
        self.data_path = Path(data_path)
        self.raw_path = self.data_path / "raw"
        self.expanded_path = self.data_path / "expanded"
        self.expanded_path.mkdir(exist_ok=True)
        
        # MediaPipe setup for hand landmark detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Target dataset sizes
        self.target_samples_per_gesture = 500
        self.current_gestures = ['clap', 'fist', 'point', 'thumbs_up', 'wave']
        
    def download_online_datasets(self):
        """Download and integrate online gesture datasets."""
        print("🔍 Searching for online gesture datasets...")
        
        # List of potential online datasets
        datasets = {
            "hand_gesture_dataset": {
                "url": "https://github.com/ardamavi/Sign-Language-Digits-Dataset",
                "description": "Sign language digits dataset with hand gestures"
            },
            "mediapipe_gesture_dataset": {
                "url": "https://www.kaggle.com/datasets/gti-upm/leapgestrecog",
                "description": "Leap Motion gesture recognition dataset"
            },
            "hand_pose_dataset": {
                "url": "https://github.com/CMU-Perceptual-Computing-Lab/openpose",
                "description": "OpenPose hand pose dataset"
            }
        }
        
        print("📥 Available online datasets:")
        for name, info in datasets.items():
            print(f"  • {name}: {info['description']}")
            print(f"    URL: {info['url']}")
        
        print("\n⚠️  Note: Manual download required for some datasets due to licensing")
        print("   Please download and place in data/external/ directory")
        
        return datasets
    
    def create_data_augmentation_pipeline(self):
        """Create data augmentation pipeline for existing samples."""
        print("\n🔄 Creating data augmentation pipeline...")
        
        augmented_data = []
        
        for gesture in self.current_gestures:
            gesture_path = self.raw_path / gesture
            print(f"  Augmenting {gesture} samples...")
            
            # Load existing samples
            existing_samples = []
            for json_file in gesture_path.glob('*.json'):
                with open(json_file, 'r') as f:
                    sample = json.load(f)
                    existing_samples.append(sample)
            
            # Calculate how many augmented samples we need
            current_count = len(existing_samples)
            target_count = self.target_samples_per_gesture
            needed_count = target_count - current_count
            
            if needed_count <= 0:
                print(f"    ✓ {gesture} already has enough samples ({current_count})")
                continue
            
            print(f"    Generating {needed_count} augmented samples...")
            
            # Generate augmented samples
            for i in tqdm(range(needed_count), desc=f"Augmenting {gesture}"):
                # Randomly select a base sample
                base_sample = random.choice(existing_samples)
                
                # Apply augmentation
                augmented_sample = self._augment_sample(base_sample, gesture, i)
                augmented_data.append(augmented_sample)
        
        return augmented_data
    
    def _augment_sample(self, base_sample, gesture, index):
        """Apply augmentation techniques to a single sample."""
        landmarks = np.array([[lm['x'], lm['y'], lm['z']] for lm in base_sample['landmarks']])
        
        # Apply random transformations
        augmented_landmarks = landmarks.copy()
        
        # 1. Random rotation (small angles)
        angle = np.random.uniform(-15, 15) * np.pi / 180  # ±15 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        augmented_landmarks = augmented_landmarks @ rotation_matrix.T
        
        # 2. Random scaling (0.9 to 1.1)
        scale = np.random.uniform(0.9, 1.1)
        augmented_landmarks *= scale
        
        # 3. Random translation (small offsets)
        translation = np.random.uniform(-0.1, 0.1, 3)
        augmented_landmarks += translation
        
        # 4. Random noise (small amount)
        noise = np.random.normal(0, 0.01, augmented_landmarks.shape)
        augmented_landmarks += noise
        
        # 5. Random finger position variations
        finger_indices = [4, 8, 12, 16, 20]  # Fingertips
        for idx in finger_indices:
            if idx < len(augmented_landmarks):
                # Add small random variation to finger positions
                finger_variation = np.random.uniform(-0.05, 0.05, 3)
                augmented_landmarks[idx] += finger_variation
        
        # Convert back to original format
        augmented_landmarks_list = []
        for i, (x, y, z) in enumerate(augmented_landmarks):
            augmented_landmarks_list.append({
                'x': float(x),
                'y': float(y),
                'z': float(z)
            })
        
        return {
            'gesture': gesture,
            'landmarks': augmented_landmarks_list,
            'timestamp': base_sample.get('timestamp', ''),
            'augmented': True,
            'base_sample': base_sample.get('sample_id', 'unknown')
        }
    
    def create_synthetic_data(self):
        """Create synthetic gesture data based on hand anatomy."""
        print("\n🤖 Creating synthetic gesture data...")
        
        synthetic_data = []
        
        # Define gesture templates based on hand anatomy
        gesture_templates = {
            'fist': {
                'finger_extensions': [0.1, 0.1, 0.1, 0.1, 0.1],  # All fingers closed
                'thumb_angle': 0.3,
                'palm_orientation': 0.0
            },
            'open_hand': {
                'finger_extensions': [0.8, 0.9, 0.9, 0.8, 0.7],  # All fingers extended
                'thumb_angle': 0.5,
                'palm_orientation': 0.0
            },
            'point': {
                'finger_extensions': [0.2, 0.9, 0.1, 0.1, 0.1],  # Only index extended
                'thumb_angle': 0.4,
                'palm_orientation': 0.0
            },
            'peace': {
                'finger_extensions': [0.2, 0.9, 0.9, 0.1, 0.1],  # Index and middle extended
                'thumb_angle': 0.4,
                'palm_orientation': 0.0
            },
            'thumbs_up': {
                'finger_extensions': [0.8, 0.1, 0.1, 0.1, 0.1],  # Only thumb extended
                'thumb_angle': 1.2,
                'palm_orientation': 0.0
            }
        }
        
        for gesture, template in gesture_templates.items():
            print(f"  Generating synthetic {gesture} samples...")
            
            for i in tqdm(range(100), desc=f"Synthetic {gesture}"):
                synthetic_sample = self._generate_synthetic_gesture(gesture, template, i)
                synthetic_data.append(synthetic_sample)
        
        return synthetic_data
    
    def _generate_synthetic_gesture(self, gesture, template, index):
        """Generate a synthetic gesture based on template."""
        # Base hand landmarks (21 points from MediaPipe)
        base_landmarks = self._get_base_hand_landmarks()
        
        # Apply gesture-specific modifications
        modified_landmarks = self._apply_gesture_template(base_landmarks, template)
        
        # Add some randomness
        modified_landmarks += np.random.normal(0, 0.02, modified_landmarks.shape)
        
        # Convert to list format
        landmarks_list = []
        for i, (x, y, z) in enumerate(modified_landmarks):
            landmarks_list.append({
                'x': float(x),
                'y': float(y),
                'z': float(z)
            })
        
        return {
            'gesture': gesture,
            'landmarks': landmarks_list,
            'timestamp': f'synthetic_{index}',
            'synthetic': True,
            'template': template
        }
    
    def _get_base_hand_landmarks(self):
        """Get base hand landmark positions."""
        # Simplified base hand landmarks (21 points)
        landmarks = np.array([
            # Wrist
            [0.0, 0.0, 0.0],
            # Thumb
            [0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [0.2, 0.0, 0.0], [0.25, 0.0, 0.0],
            # Index finger
            [0.0, 0.1, 0.0], [0.0, 0.15, 0.0], [0.0, 0.2, 0.0], [0.0, 0.25, 0.0],
            # Middle finger
            [0.0, 0.2, 0.0], [0.0, 0.25, 0.0], [0.0, 0.3, 0.0], [0.0, 0.35, 0.0],
            # Ring finger
            [0.0, 0.3, 0.0], [0.0, 0.35, 0.0], [0.0, 0.4, 0.0], [0.0, 0.45, 0.0],
            # Pinky
            [0.0, 0.4, 0.0], [0.0, 0.45, 0.0], [0.0, 0.5, 0.0], [0.0, 0.55, 0.0]
        ])
        
        return landmarks
    
    def _apply_gesture_template(self, landmarks, template):
        """Apply gesture template modifications to landmarks."""
        modified = landmarks.copy()
        
        # Apply finger extensions
        finger_extensions = template['finger_extensions']
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        
        for i, (tip_idx, extension) in enumerate(zip(finger_tips, finger_extensions)):
            if tip_idx < len(modified):
                # Modify finger tip position based on extension
                base_pos = modified[tip_idx - 1]  # Base of finger
                direction = modified[tip_idx] - base_pos
                modified[tip_idx] = base_pos + direction * extension
        
        return modified
    
    def collect_additional_real_data(self):
        """Collect additional real data using webcam."""
        print("\n📹 Setting up real-time data collection...")
        print("   This will open your webcam for manual data collection")
        print("   Press 'q' to quit, 's' to save current frame")
        
        # Implementation would go here
        # For now, we'll create a placeholder
        print("   ⚠️  Real-time collection requires user interaction")
        print("   Please implement webcam integration separately")
        
        return []
    
    def integrate_external_datasets(self):
        """Integrate external datasets if available."""
        external_path = self.data_path / "external"
        
        if not external_path.exists():
            print(f"\n📁 Creating external data directory: {external_path}")
            external_path.mkdir(exist_ok=True)
            print("   Please place downloaded datasets in this directory")
            return []
        
        print(f"\n🔗 Checking for external datasets in {external_path}")
        
        # Look for common dataset formats
        dataset_files = list(external_path.glob("*.csv")) + list(external_path.glob("*.json"))
        
        if not dataset_files:
            print("   No external datasets found")
            return []
        
        integrated_data = []
        
        for dataset_file in dataset_files:
            print(f"   Processing {dataset_file.name}...")
            # Implementation for dataset integration would go here
            # This depends on the specific format of external datasets
        
        return integrated_data
    
    def save_expanded_dataset(self, augmented_data, synthetic_data, external_data):
        """Save the expanded dataset."""
        print("\n💾 Saving expanded dataset...")
        
        # Combine all data
        all_data = augmented_data + synthetic_data + external_data
        
        # Group by gesture
        gesture_groups = {}
        for sample in all_data:
            gesture = sample['gesture']
            if gesture not in gesture_groups:
                gesture_groups[gesture] = []
            gesture_groups[gesture].append(sample)
        
        # Save to expanded directory
        for gesture, samples in gesture_groups.items():
            gesture_path = self.expanded_path / gesture
            gesture_path.mkdir(exist_ok=True)
            
            print(f"  Saving {len(samples)} samples for {gesture}...")
            
            for i, sample in enumerate(samples):
                filename = f"expanded_{gesture}_{i:04d}.json"
                filepath = gesture_path / filename
                
                with open(filepath, 'w') as f:
                    json.dump(sample, f, indent=2)
        
        print(f"✅ Expanded dataset saved to {self.expanded_path}")
        
        # Print summary
        total_samples = sum(len(samples) for samples in gesture_groups.values())
        print(f"\n📊 Dataset Expansion Summary:")
        print(f"  • Total samples: {total_samples}")
        print(f"  • Gesture classes: {len(gesture_groups)}")
        for gesture, samples in gesture_groups.items():
            print(f"    - {gesture}: {len(samples)} samples")
    
    def run_expansion(self):
        """Run the complete dataset expansion pipeline."""
        print("🚀 Starting dataset expansion...")
        
        # 1. Check online datasets
        online_datasets = self.download_online_datasets()
        
        # 2. Create augmented data
        augmented_data = self.create_data_augmentation_pipeline()
        
        # 3. Create synthetic data
        synthetic_data = self.create_synthetic_data()
        
        # 4. Collect additional real data (placeholder)
        real_data = self.collect_additional_real_data()
        
        # 5. Integrate external datasets
        external_data = self.integrate_external_datasets()
        
        # 6. Save expanded dataset
        self.save_expanded_dataset(augmented_data, synthetic_data, external_data)
        
        print("\n🎉 Dataset expansion completed!")
        print("\n📋 Next steps:")
        print("  1. Download external datasets from the provided URLs")
        print("  2. Implement webcam data collection for real data")
        print("  3. Retrain your model with the expanded dataset")
        print("  4. Consider using transfer learning from pre-trained models")

def main():
    """Main function."""
    expander = DatasetExpander()
    expander.run_expansion()

if __name__ == "__main__":
    main() 