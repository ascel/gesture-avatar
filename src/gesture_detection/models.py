"""
Gesture recognition models for gesture-controlled avatar project.
Implements MediaPipe + custom CNN architecture for real-time gesture detection.
"""

import numpy as np
import tensorflow as tf
try:
    # TensorFlow 2.15+ uses standalone Keras
    import keras
    from keras import layers, models, optimizers
    from keras.applications import MobileNetV2
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
    from keras.applications import EfficientNetB0
except ImportError:
    # Fallback for older TensorFlow versions
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
    from tensorflow.keras.applications import EfficientNetB0
import mediapipe as mp
from typing import List, Dict, Tuple, Optional
import cv2
import json
from pathlib import Path
import joblib


def configure_gpu():
    """Configure GPU settings for optimal performance."""
    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for faster training
            try:
                keras.mixed_precision.set_global_policy('mixed_float16')
            except:
                # Fallback for older versions
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            print("✅ GPU acceleration configured with memory growth and mixed precision")
            return True
        else:
            print("⚠️  No GPU found, using CPU")
            return False
    except Exception as e:
        print(f"⚠️  GPU configuration failed: {e}")
        print("   Continuing with CPU")
        return False


def top_3_accuracy(y_true, y_pred):
    """Custom metric for top-3 accuracy."""
    try:
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    except:
        # Fallback for older versions
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


class GestureTrainingCallback(Callback):
    """Custom callback for gesture model training with advanced monitoring."""
    
    def __init__(self, patience=15, min_delta=0.001):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_accuracy = 0
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_accuracy = logs.get('val_accuracy', 0)
        
        if current_val_accuracy > self.best_val_accuracy + self.min_delta:
            self.best_val_accuracy = current_val_accuracy
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print(f'\nEarly stopping triggered at epoch {epoch}')
            
        # Print detailed metrics
        print(f"Epoch {epoch + 1}: "
              f"loss={logs.get('loss', 0):.4f}, "
              f"accuracy={logs.get('accuracy', 0):.4f}, "
              f"val_loss={logs.get('val_loss', 0):.4f}, "
              f"val_accuracy={logs.get('val_accuracy', 0):.4f}")


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
        
        # Add batch dimension if needed
        if len(feature_array.shape) == 1:
            feature_array = np.expand_dims(feature_array, axis=0)
        
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
    
    def build_model(self, num_classes: int, input_dim: int = 13):
        """Build an optimized ResNet-inspired neural network model."""
        
        def residual_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
            """Residual block with skip connections."""
            shortcut = x
            
            # First convolution
            x = layers.Dense(filters, use_bias=use_bias, name=f"{name}_conv1")(x)
            x = layers.BatchNormalization(name=f"{name}_bn1")(x)
            x = layers.ReLU(name=f"{name}_relu1")(x)
            x = layers.Dropout(0.2)(x)
            
            # Second convolution
            x = layers.Dense(filters, use_bias=use_bias, name=f"{name}_conv2")(x)
            x = layers.BatchNormalization(name=f"{name}_bn2")(x)
            
            # Skip connection (if dimensions match)
            if shortcut.shape[-1] == filters:
                x = layers.Add(name=f"{name}_add")([shortcut, x])
            else:
                # Projection shortcut if dimensions don't match
                shortcut = layers.Dense(filters, use_bias=use_bias, name=f"{name}_shortcut")(shortcut)
                x = layers.Add(name=f"{name}_add")([shortcut, x])
            
            x = layers.ReLU(name=f"{name}_relu_out")(x)
            return x
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,), name='input_features')
        
        # Initial feature extraction
        x = layers.Dense(256, activation='relu', name='initial_dense')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.Dropout(0.3)(x)
        
        # Residual blocks
        x = residual_block(x, 256, name='res_block_1')
        x = residual_block(x, 256, name='res_block_2')
        
        x = residual_block(x, 128, name='res_block_3')
        x = residual_block(x, 128, name='res_block_4')
        
        x = residual_block(x, 64, name='res_block_5')
        x = residual_block(x, 64, name='res_block_6')
        
        # Global feature aggregation
        x = layers.Dense(128, activation='relu', name='global_features')(x)
        x = layers.BatchNormalization(name='global_bn')(x)
        x = layers.Dropout(0.4)(x)
        
        # Attention mechanism
        attention_weights = layers.Dense(128, activation='softmax', name='attention')(x)
        x = layers.Multiply(name='attention_applied')([x, attention_weights])
        
        # Final classification layers
        x = layers.Dense(64, activation='relu', name='final_dense')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_Gesture_Model')
        
        # Compile with advanced optimizer settings
        optimizer = optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 32):
        """Train the optimized ResNet-inspired model with GPU acceleration."""
        # Configure GPU if available
        gpu_available = configure_gpu()
        
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        
        print(f"🚀 Training ResNet-inspired gesture model")
        print(f"   • Input features: {input_dim}")
        print(f"   • Classes: {num_classes}")
        print(f"   • Training samples: {len(X_train)}")
        print(f"   • Validation samples: {len(X_val)}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Max epochs: {epochs}")
        print(f"   • GPU acceleration: {'Enabled' if gpu_available else 'Disabled'}")
        print()
        
        # Build model
        self.model = self.build_model(num_classes, input_dim)
        
        # Print model summary
        print("📊 Model Architecture:")
        self.model.summary()
        print()
        
        # Advanced callbacks
        callbacks = [
            GestureTrainingCallback(patience=20, min_delta=0.001),
            ModelCheckpoint(
                'data/models/best_gesture_model.h5', 
                save_best_only=True, 
                monitor='val_accuracy',
                mode='max'
            ),
            ReduceLROnPlateau(
                factor=0.7, 
                patience=10, 
                min_lr=1e-7,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Train with class weights for imbalanced datasets
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print("⚖️  Class weights for balanced training:")
        for class_idx, weight in class_weight_dict.items():
            print(f"   • Class {class_idx}: {weight:.3f}")
        print()
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )
        
        print(f"\n✅ Training completed!")
        print(f"   • Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        print(f"   • Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return history
    
    def load_model(self, model_path: str):
        """Load pre-trained model and preprocessing artifacts."""
        # Load model
        try:
            self.model = keras.models.load_model(model_path)
        except:
            # Fallback for older versions
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
        try:
            self.model.save(model_path)
        except:
            # Fallback for older versions
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
    
    def build_model(self, num_classes: int):
        """Build an optimized ResNet-inspired image-based model."""
        
        def residual_block_2d(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
            """2D Residual block with skip connections."""
            shortcut = x
            
            # First convolution
            x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', 
                            use_bias=use_bias, name=f"{name}_conv1")(x)
            x = layers.BatchNormalization(name=f"{name}_bn1")(x)
            x = layers.ReLU(name=f"{name}_relu1")(x)
            x = layers.Dropout(0.2)(x)
            
            # Second convolution
            x = layers.Conv2D(filters, kernel_size, padding='same', 
                            use_bias=use_bias, name=f"{name}_conv2")(x)
            x = layers.BatchNormalization(name=f"{name}_bn2")(x)
            
            # Skip connection (if dimensions match)
            if shortcut.shape[-1] == filters:
                x = layers.Add(name=f"{name}_add")([shortcut, x])
            else:
                # Projection shortcut if dimensions don't match
                shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', 
                                       use_bias=use_bias, name=f"{name}_shortcut")(shortcut)
                x = layers.Add(name=f"{name}_add")([shortcut, x])
            
            x = layers.ReLU(name=f"{name}_relu_out")(x)
            return x
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='input_image')
        
        # Initial feature extraction
        x = layers.Conv2D(64, 7, strides=2, padding='same', name='initial_conv')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        x = layers.ReLU(name='initial_relu')(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same', name='initial_pool')(x)
        
        # Residual blocks with different filter sizes
        x = residual_block_2d(x, 64, name='res_block_1')
        x = residual_block_2d(x, 64, name='res_block_2')
        
        x = residual_block_2d(x, 128, stride=2, name='res_block_3')
        x = residual_block_2d(x, 128, name='res_block_4')
        
        x = residual_block_2d(x, 256, stride=2, name='res_block_5')
        x = residual_block_2d(x, 256, name='res_block_6')
        
        # Global feature aggregation
        x = layers.GlobalAveragePooling2D(name='global_pool')(x)
        
        # Feature refinement with attention
        x = layers.Dense(512, activation='relu', name='feature_refinement')(x)
        x = layers.BatchNormalization(name='refinement_bn')(x)
        x = layers.Dropout(0.4)(x)
        
        # Multi-head attention mechanism
        attention_heads = []
        for i in range(4):  # 4 attention heads
            attention_head = layers.Dense(128, activation='softmax', 
                                        name=f'attention_head_{i}')(x)
            attention_heads.append(attention_head)
        
        # Combine attention heads
        attention_combined = layers.Concatenate(name='attention_combined')(attention_heads)
        x = layers.Multiply(name='attention_applied')([x, attention_combined])
        
        # Final classification layers
        x = layers.Dense(256, activation='relu', name='final_dense_1')(x)
        x = layers.BatchNormalization(name='final_bn_1')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu', name='final_dense_2')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='ResNet_Image_Gesture_Model')
        
        # Compile with advanced optimizer settings
        optimizer = optimizers.AdamW(
            learning_rate=0.0001,  # Lower learning rate for image models
            weight_decay=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_generator, val_generator, epochs: int = 100):
        """Train the optimized ResNet-inspired image model with GPU acceleration."""
        # Configure GPU if available
        gpu_available = configure_gpu()
        
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
        try:
            self.model = keras.models.load_model(model_path)
        except:
            # Fallback for older versions
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


class EfficientNetGestureModel(GestureRecognitionModel):
    """Image-based gesture recognition model using EfficientNetB0."""
    def __init__(self, model_path: str = None, config: Dict = None):
        super().__init__(model_path, config)
        self.input_shape = (224, 224, 3)
        self.class_indices = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        return image

    def build_model(self, num_classes: int):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False  # Optionally fine-tune later
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs, name='EfficientNet_Gesture_Model')
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0005),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_generator, val_generator, epochs=50):
        num_classes = train_generator.num_classes
        self.model = self.build_model(num_classes)
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('data/models/best_efficientnet_gesture_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss', mode='min')
        ]
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        if self.model is None:
            return "unknown", 0.0
        img = self.preprocess_image(image)
        img = np.expand_dims(img, axis=0)
        preds = self.model.predict(img, verbose=0)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])
        if self.class_indices:
            inv_map = {v: k for k, v in self.class_indices.items()}
            gesture_name = inv_map.get(class_idx, f"gesture_{class_idx}")
        else:
            gesture_name = f"gesture_{class_idx}"
        return gesture_name, confidence

    def load_model(self, model_path: str):
        try:
            self.model = keras.models.load_model(model_path)
        except:
            self.model = tf.keras.models.load_model(model_path)

    def save_model(self, model_path: str):
        self.model.save(model_path)


class EfficientNet1DLandmarkModel(GestureRecognitionModel):
    """Landmark-based gesture recognition model using a 1D EfficientNet-inspired architecture."""
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

    def build_model(self, num_classes: int, input_dim: int = 13):
        from tensorflow.keras import layers, models
        inputs = layers.Input(shape=(input_dim, 1), name='input_features')
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs, name='EfficientNet1D_Landmark_Model')
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 200, batch_size: int = 32):
        gpu_available = configure_gpu()
        num_classes = len(np.unique(y_train))
        input_dim = X_train.shape[1]
        # Reshape for Conv1D: (samples, timesteps, features) -> (samples, features, 1)
        X_train_reshaped = np.expand_dims(X_train, axis=-1)
        X_val_reshaped = np.expand_dims(X_val, axis=-1)
        self.model = self.build_model(num_classes, input_dim)
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ModelCheckpoint('data/models/best_efficientnet1d_gesture_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
            ReduceLROnPlateau(factor=0.7, patience=10, min_lr=1e-7, monitor='val_loss', mode='min')
        ]
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1,
            shuffle=True
        )
        return history

    def preprocess_landmarks(self, landmarks: List[Dict]) -> np.ndarray:
        features = self.extract_features(landmarks)
        if features is None:
            return None
        feature_array = np.array([features[name] for name in self.feature_names])
        if self.feature_scaler is not None:
            feature_array = self.feature_scaler.transform(feature_array.reshape(1, -1))
        return feature_array

    def predict(self, landmarks: List[Dict]) -> Tuple[str, float]:
        if self.model is None:
            return self._basic_gesture_detection(landmarks)
        feature_array = self.preprocess_landmarks(landmarks)
        if feature_array is None:
            return "unknown", 0.0
        # Reshape for Conv1D: (1, features, 1)
        feature_array = np.expand_dims(feature_array, axis=-1)
        prediction = self.model.predict(feature_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        if self.label_encoder is not None:
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
        else:
            gesture_name = f"gesture_{predicted_class}"
        return gesture_name, confidence

    def load_model(self, model_path: str):
        try:
            self.model = keras.models.load_model(model_path)
        except:
            self.model = tf.keras.models.load_model(model_path)
        model_dir = Path(model_path).parent
        scaler_path = model_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
        encoder_path = model_dir / "label_encoder.pkl"
        if encoder_path.exists():
            self.label_encoder = joblib.load(encoder_path)

    def save_model(self, model_path: str):
        self.model.save(model_path)
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.feature_scaler is not None:
            joblib.dump(self.feature_scaler, model_dir / "feature_scaler.pkl")
        if self.label_encoder is not None:
            joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")


class GestureDetector:
    """Main gesture detector class that combines MediaPipe and custom models."""
    
    def __init__(self, model_type: str = "feature", model_path: str = None, config: Dict = None, model_backbone: str = "resnet"):
        """
        Initialize the gesture detector.
        
        Args:
            model_type: "feature" or "image"
            model_path: Path to pre-trained model
            config: Configuration dictionary
            model_backbone: Backbone for image model ("resnet" or "efficientnet")
        """
        self.model_type = model_type
        self.config = config or {}
        self.model_backbone = model_backbone
        
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
            if model_backbone == "efficientnet1d":
                self.model = EfficientNet1DLandmarkModel(model_path, config)
            else:
                self.model = FeatureBasedGestureModel(model_path, config)
        elif model_type == "image":
            if model_backbone == "efficientnet":
                self.model = EfficientNetGestureModel(model_path, config)
            else:
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