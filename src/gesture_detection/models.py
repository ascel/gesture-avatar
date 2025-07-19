"""
Gesture recognition models for gesture-controlled avatar project.
Implements MediaPipe + custom CNN architecture for real-time gesture detection.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import cv2
import json
from pathlib import Path
import joblib


class GestureRecognitionModel:
    """Base class for gesture recognition models."""
    
    def __init__(self, model_path: str = None, config: Dict = None):
        """
        Initialize the gesture recognition model.
        
        Args:
            model_path: Path to pre-trained model
            config: Model configuration
        """
        self.model = None
        self.config = config or {}
        self.model_path = model_path
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.get('max_hands', 2),
            min_detection_confidence=self.config.get('confidence_threshold', 0.7),
            min_tracking_confidence=0.5
        )
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def preprocess_landmarks(self, landmarks: List[Dict]) -> np.ndarray:
        """
        Preprocess landmarks for model input.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Preprocessed feature array
        """
        raise NotImplementedError
    
    def predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        """
        Predict gesture from landmarks.
        
        Args:
            landmarks: List of landmark dictionaries
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        raise NotImplementedError
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        raise NotImplementedError
    
    def save_model(self, model_path: str):
        """Save trained model."""
        raise NotImplementedError


class FeatureBasedGestureModel(GestureRecognitionModel):
    """Feature-based gesture recognition model using hand landmarks."""
    
    def __init__(self, model_path: str = None, config: Dict = None):
        super().__init__(model_path, config)
        self.feature_scaler = None
        self.label_encoder = None
        self.feature_names = [
            'hand_span', 'palm_area', 'thumb_extension', 'index_extension',
            'middle_extension', 'ring_extension', 'pinky_extension',
            'hand_center_x', 'hand_center_y', 'hand_center_z',
            'thumb_angle', 'index_angle', 'palm_orientation'
        ]
    
    def extract_features(self, landmarks: List[Dict]) -> Dict:
        """Extract hand features from landmarks."""
        if len(landmarks) != 21:
            return None
        
        landmarks_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        features = {}
        
        # Hand span
        features['hand_span'] = np.linalg.norm(landmarks_array[0] - landmarks_array[12])
        
        # Palm area
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
        
        # Finger angles
        features['thumb_angle'] = np.arctan2(landmarks_array[4, 1] - landmarks_array[2, 1],
                                           landmarks_array[4, 0] - landmarks_array[2, 0])
        features['index_angle'] = np.arctan2(landmarks_array[8, 1] - landmarks_array[5, 1],
                                           landmarks_array[8, 0] - landmarks_array[5, 0])
        
        # Hand orientation
        palm_normal = np.cross(landmarks_array[5] - landmarks_array[0],
                              landmarks_array[17] - landmarks_array[0])
        features['palm_orientation'] = np.arctan2(palm_normal[1], palm_normal[0])
        
        return features
    
    def preprocess_landmarks(self, landmarks: List[Dict]) -> np.ndarray:
        """Preprocess landmarks into feature array."""
        features = self.extract_features(landmarks)
        if features is None:
            return None
        
        feature_array = np.array([features[name] for name in self.feature_names])
        
        if self.feature_scaler is not None:
            feature_array = self.feature_scaler.transform(feature_array.reshape(1, -1))
        
        return feature_array
    
    def predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        """Predict gesture from landmarks."""
        if self.model is None:
            # Fallback to basic gesture detection
            return self._basic_gesture_detection(landmarks)
        
        feature_array = self.preprocess_landmarks(landmarks)
        if feature_array is None:
            return "unknown", 0.0
        
        # Get prediction
        prediction = self.model.predict(feature_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Decode class label
        if self.label_encoder is not None:
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
        else:
            gesture_name = f"gesture_{predicted_class}"
        
        return gesture_name, confidence
    
    def _basic_gesture_detection(self, landmarks: List[Dict]) -> Tuple[str, float]:
        """Basic gesture detection without trained model."""
        if not landmarks or len(landmarks) < 21:  # MediaPipe hands have 21 landmarks
            return "no_hand", 0.0
        
        # Get key landmarks
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Get palm base
        wrist = landmarks[0]
        
        # Calculate distances
        def distance(p1, p2):
            return ((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)**0.5
        
        # Check if fingers are extended
        thumb_extended = thumb_tip['y'] < wrist['y']
        index_extended = index_tip['y'] < wrist['y']
        middle_extended = middle_tip['y'] < wrist['y']
        ring_extended = ring_tip['y'] < wrist['y']
        pinky_extended = pinky_tip['y'] < wrist['y']
        
        # Basic gesture classification
        extended_fingers = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        if extended_fingers == 0:
            return "fist", 0.8
        elif extended_fingers == 1 and index_extended:
            return "point", 0.8
        elif extended_fingers == 2 and index_extended and middle_extended:
            return "peace", 0.8
        elif extended_fingers == 5:
            return "open_hand", 0.8
        elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
            return "thumbs_up", 0.8
        else:
            return "unknown", 0.5
    
    def build_model(self, num_classes: int, input_dim: int = 13) -> tf.keras.Model:
        """Build the neural network model."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32):
        """Train the model."""
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        # Build model
        self.model = self.build_model(num_classes, input_dim)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_gesture_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def load_model(self, model_path: str):
        """Load pre-trained model and preprocessing artifacts."""
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load preprocessing artifacts
        model_dir = Path(model_path).parent
        
        scaler_path = model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
        
        encoder_path = model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)
    
    def save_model(self, model_path: str):
        """Save trained model and preprocessing artifacts."""
        # Save model
        self.model.save(model_path)
        
        # Save preprocessing artifacts
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, model_dir / "feature_scaler.pkl")
        
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")


class ImageBasedGestureModel(GestureRecognitionModel):
    """Image-based gesture recognition model using MobileNet."""
    
    def __init__(self, model_path: str = None, config: Dict = None):
        super().__init__(model_path, config)
        self.input_shape = (224, 224, 3)
        self.class_indices = None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize image
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict gesture from image."""
        if self.model is None:
            # Fallback to basic gesture detection
            return "unknown", 0.5
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Get prediction
        prediction = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Decode class label
        if self.class_indices is not None:
            gesture_name = list(self.class_indices.keys())[predicted_class]
        else:
            gesture_name = f"gesture_{predicted_class}"
        
        return gesture_name, confidence
    
    def build_model(self, num_classes: int) -> tf.keras.Model:
        """Build MobileNet-based model."""
        # Load pre-trained MobileNet
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Add custom layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, val_generator, epochs: int = 50):
        """Train the model."""
        num_classes = len(train_generator.class_indices)
        self.class_indices = train_generator.class_indices
        
        # Build model
        self.model = self.build_model(num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_image_gesture_model.h5', save_best_only=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices if available
        model_dir = Path(model_path).parent
        indices_path = model_dir / "class_indices.json"
        if indices_path.exists():
            with open(indices_path, 'r') as f:
                self.class_indices = json.load(f)
    
    def save_model(self, model_path: str):
        """Save trained model."""
        self.model.save(model_path)
        
        # Save class indices
        if self.class_indices is not None:
            model_dir = Path(model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            with open(model_dir / "class_indices.json", 'w') as f:
                json.dump(self.class_indices, f, indent=2)


class GestureDetector:
    """Main gesture detector class that combines MediaPipe and custom models."""
    
    def __init__(self, model_type: str = "feature", model_path: str = None, config: Dict = None):
        """
        Initialize the gesture detector.
        
        Args:
            model_type: "feature" or "image"
            model_path: Path to pre-trained model
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        
        # Initialize MediaPipe
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize model
        if model_type == "feature":
            self.model = FeatureBasedGestureModel(model_path, config)
        elif model_type == "image":
            self.model = ImageBasedGestureModel(model_path, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = 0
    
    def detect_gesture(self, frame: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Detect gesture in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (gesture_name, confidence, additional_info)
        """
        # Process with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return "no_hand", 0.0, {"landmarks": None}
        
        # Get first hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Convert landmarks to dictionary format
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        # Predict gesture
        if self.model_type == "feature":
            gesture, confidence = self.model.predict(landmarks)
        else:  # image
            gesture, confidence = self.model.predict(frame)
        
        # Additional info
        additional_info = {
            "landmarks": landmarks,
            "num_hands": len(results.multi_hand_landmarks),
            "fps": self._calculate_fps()
        }
        
        return gesture, confidence, additional_info
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        import time
        current_time = time.time()
        
        if current_time - self.last_fps_time > 0:
            fps = 1.0 / (current_time - self.last_fps_time)
            self.fps_counter = fps
        
        self.last_fps_time = current_time
        return self.fps_counter
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """Draw hand landmarks on frame."""
        if landmarks is None:
            return frame
        
        # Simple landmark drawing without MediaPipe
        h, w = frame.shape[:2]
        
        for i, landmark in enumerate(landmarks):
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            
            # Draw different colors for different landmark types
            if i == 0:  # Wrist
                color = (0, 255, 0)  # Green
                radius = 8
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                color = (255, 0, 0)  # Red
                radius = 6
            else:  # Other landmarks
                color = (0, 0, 255)  # Blue
                radius = 4
            
            cv2.circle(frame, (x, y), radius, color, -1)
        
        return frame 