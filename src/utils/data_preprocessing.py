"""
Data preprocessing utility for gesture-controlled avatar project.
Prepares collected gesture data for model training with augmentation and normalization.
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')


class GestureDataPreprocessor:
    """Preprocesses gesture data for model training."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """
        Initialize the data preprocessor.
        
        Args:
            raw_data_dir: Directory containing raw gesture data
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers and encoders
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data augmentation parameters
        self.augmentation_params = {
            'rotation_range': 15,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': False,  # Don't flip hands
            'fill_mode': 'nearest'
        }
    
    def load_landmarks_data(self) -> pd.DataFrame:
        """Load all landmark data from JSON files."""
        data = []
        
        for gesture_dir in self.raw_data_dir.iterdir():
            if not gesture_dir.is_dir():
                continue
                
            gesture_name = gesture_dir.name
            
            for json_file in gesture_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    sample_data = json.load(f)
                
                landmarks = sample_data['landmarks']
                
                # Flatten landmarks into feature vector
                feature_vector = []
                for landmark in landmarks:
                    feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])
                
                data.append({
                    'gesture': gesture_name,
                    'sample_id': json_file.stem,
                    'features': feature_vector,
                    'timestamp': sample_data.get('timestamp', '')
                })
        
        return pd.DataFrame(data)
    
    def extract_hand_features(self, landmarks: List[Dict]) -> Dict:
        """
        Extract meaningful features from hand landmarks.
        
        Args:
            landmarks: List of landmark dictionaries with x, y, z coordinates
            
        Returns:
            Dictionary of extracted features
        """
        if len(landmarks) != 21:  # MediaPipe hands have 21 landmarks
            return None
        
        # Convert to numpy array
        landmarks_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Extract features
        features = {}
        
        # Hand span (distance between wrist and middle finger tip)
        features['hand_span'] = np.linalg.norm(landmarks_array[0] - landmarks_array[12])
        
        # Palm area (approximate)
        palm_width = np.linalg.norm(landmarks_array[0] - landmarks_array[5])
        palm_height = np.linalg.norm(landmarks_array[5] - landmarks_array[17])
        features['palm_area'] = palm_width * palm_height
        
        # Finger extensions
        features['thumb_extension'] = np.linalg.norm(landmarks_array[2] - landmarks_array[4])
        features['index_extension'] = np.linalg.norm(landmarks_array[5] - landmarks_array[8])
        features['middle_extension'] = np.linalg.norm(landmarks_array[9] - landmarks_array[12])
        features['ring_extension'] = np.linalg.norm(landmarks_array[13] - landmarks_array[16])
        features['pinky_extension'] = np.linalg.norm(landmarks_array[17] - landmarks_array[20])
        
        # Hand center
        features['hand_center_x'] = np.mean(landmarks_array[:, 0])
        features['hand_center_y'] = np.mean(landmarks_array[:, 1])
        features['hand_center_z'] = np.mean(landmarks_array[:, 2])
        
        # Finger angles (relative to palm)
        features['thumb_angle'] = np.arctan2(landmarks_array[4, 1] - landmarks_array[2, 1],
                                           landmarks_array[4, 0] - landmarks_array[2, 0])
        features['index_angle'] = np.arctan2(landmarks_array[8, 1] - landmarks_array[5, 1],
                                           landmarks_array[8, 0] - landmarks_array[5, 0])
        
        # Hand orientation
        palm_normal = np.cross(landmarks_array[5] - landmarks_array[0],
                              landmarks_array[17] - landmarks_array[0])
        features['palm_orientation'] = np.arctan2(palm_normal[1], palm_normal[0])
        
        return features
    
    def augment_landmarks(self, landmarks: List[Dict], num_augmentations: int = 3) -> List[List[Dict]]:
        """
        Augment landmark data with noise and variations.
        
        Args:
            landmarks: Original landmarks
            num_augmentations: Number of augmented samples to generate
            
        Returns:
            List of augmented landmark sets
        """
        augmented_samples = []
        
        for _ in range(num_augmentations):
            augmented_landmarks = []
            
            for landmark in landmarks:
                # Add noise to coordinates
                noise_x = np.random.normal(0, 0.01)  # Small noise
                noise_y = np.random.normal(0, 0.01)
                noise_z = np.random.normal(0, 0.005)
                
                augmented_landmark = {
                    'x': landmark['x'] + noise_x,
                    'y': landmark['y'] + noise_y,
                    'z': landmark['z'] + noise_z
                }
                augmented_landmarks.append(augmented_landmark)
            
            augmented_samples.append(augmented_landmarks)
        
        return augmented_samples
    
    def prepare_training_data(self, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Prepare training, validation, and test datasets.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder)
        """
        print("Loading and preprocessing gesture data...")
        
        # Load raw data
        raw_df = self.load_landmarks_data()
        
        if raw_df.empty:
            raise ValueError("No data found in raw data directory!")
        
        print(f"Loaded {len(raw_df)} samples")
        
        # Extract features
        feature_data = []
        labels = []
        
        for _, row in raw_df.iterrows():
            # Parse landmarks from feature vector
            features = row['features']
            landmarks = []
            for i in range(0, len(features), 3):
                landmarks.append({
                    'x': features[i],
                    'y': features[i+1],
                    'z': features[i+2]
                })
            
            # Extract hand features for original sample
            hand_features = self.extract_hand_features(landmarks)
            if hand_features is None:
                continue
            feature_data.append(list(hand_features.values()))
            labels.append(row['gesture'])

            # Augment and extract features for each augmented sample
            augmented_landmarks_list = self.augment_landmarks(landmarks, num_augmentations=3)
            for aug_landmarks in augmented_landmarks_list:
                aug_features = self.extract_hand_features(aug_landmarks)
                if aug_features is not None:
                    feature_data.append(list(aug_features.values()))
                    labels.append(row['gesture'])
        
        # Convert to numpy arrays
        X = np.array(feature_data)
        y = np.array(labels)
        
        print(f"Extracted features for {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        print(f"Unique gestures: {np.unique(y)}")
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Save preprocessing artifacts
        self._save_preprocessing_artifacts()
        
        # Save processed datasets
        self._save_processed_datasets(X_train_scaled, X_val_scaled, X_test_scaled,
                                    y_train_encoded, y_val_encoded, y_test_encoded)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Validation set: {X_val_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_encoded, y_val_encoded, y_test_encoded,
                self.feature_scaler, self.label_encoder)
    
    def _save_preprocessing_artifacts(self):
        """Save preprocessing artifacts (scalers, encoders)."""
        import joblib
        
        # Save scaler
        scaler_path = self.processed_data_dir / "feature_scaler.pkl"
        joblib.dump(self.feature_scaler, scaler_path)
        
        # Save label encoder
        encoder_path = self.processed_data_dir / "label_encoder.pkl"
        joblib.dump(self.label_encoder, encoder_path)
        
        # Save feature names
        feature_names = [
            'hand_span', 'palm_area', 'thumb_extension', 'index_extension',
            'middle_extension', 'ring_extension', 'pinky_extension',
            'hand_center_x', 'hand_center_y', 'hand_center_z',
            'thumb_angle', 'index_angle', 'palm_orientation'
        ]
        
        feature_info = {
            'feature_names': feature_names,
            'num_features': len(feature_names),
            'gesture_classes': self.label_encoder.classes_.tolist()
        }
        
        with open(self.processed_data_dir / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"Preprocessing artifacts saved to {self.processed_data_dir}")
    
    def _save_processed_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed datasets as numpy arrays."""
        np.save(self.processed_data_dir / "X_train.npy", X_train)
        np.save(self.processed_data_dir / "X_val.npy", X_val)
        np.save(self.processed_data_dir / "X_test.npy", X_test)
        np.save(self.processed_data_dir / "y_train.npy", y_train)
        np.save(self.processed_data_dir / "y_val.npy", y_val)
        np.save(self.processed_data_dir / "y_test.npy", y_test)
        
        print(f"Processed datasets saved to {self.processed_data_dir}")
    
    def create_image_dataset(self, target_size: Tuple[int, int] = (224, 224)) -> Tuple:
        """
        Create image-based dataset from gesture images.
        
        Args:
            target_size: Target image size (width, height)
            
        Returns:
            Tuple of (train_generator, val_generator, test_generator, class_indices)
        """
        print("Creating image dataset...")
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            **self.augmentation_params,
            rescale=1./255
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.raw_data_dir,
            target_size=target_size,
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.raw_data_dir,
            target_size=target_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Save class indices
        class_indices = train_generator.class_indices
        with open(self.processed_data_dir / "class_indices.json", 'w') as f:
            json.dump(class_indices, f, indent=2)
        
        print(f"Image dataset created with {len(class_indices)} classes")
        print(f"Class indices: {class_indices}")
        
        return train_generator, val_generator, class_indices


def main():
    """Main function for data preprocessing."""
    preprocessor = GestureDataPreprocessor()
    
    print("Gesture Data Preprocessing")
    print("=========================")
    
    try:
        # Prepare feature-based training data
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         scaler, encoder) = preprocessor.prepare_training_data()
        
        print("\nFeature-based dataset prepared successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(encoder.classes_)}")
        
        # Try to create image dataset if images exist
        try:
            train_gen, val_gen, class_indices = preprocessor.create_image_dataset()
            print("\nImage dataset created successfully!")
        except Exception as e:
            print(f"\nImage dataset creation failed: {e}")
            print("This is expected if no images are available yet.")
        
        print("\nPreprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Make sure you have collected gesture data first using the data collection tool.")


if __name__ == "__main__":
    main() 