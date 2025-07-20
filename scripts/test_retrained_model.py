#!/usr/bin/env python3
"""
Test Retrained Model Script

This script tests the retrained gesture recognition model to verify it works correctly.

Usage:
    python scripts/test_retrained_model.py
"""

import json
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from gesture_detection.models import FeatureBasedGestureModel
    from utils.data_preprocessing import GestureDataPreprocessor
except ImportError:
    # Fallback imports for when running from scripts directory
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.gesture_detection.models import FeatureBasedGestureModel
    from src.utils.data_preprocessing import GestureDataPreprocessor

class ModelTester:
    def __init__(self):
        self.base_path = Path("../data")
        self.models_path = self.base_path / "models"
        self.test_data_path = self.base_path / "processed"
        
        # Model paths
        self.feature_model_path = self.models_path / "feature_gesture_model.h5"
        self.best_model_path = Path("../best_gesture_model.h5")
        
    def check_model_availability(self):
        """Check if trained models are available."""
        print("🔍 Checking model availability...")
        
        models_found = {}
        
        if self.feature_model_path.exists():
            models_found['feature_model'] = str(self.feature_model_path)
            print(f"  ✅ Feature model found: {self.feature_model_path}")
        else:
            print(f"  ❌ Feature model not found: {self.feature_model_path}")
        
        if self.best_model_path.exists():
            models_found['best_model'] = str(self.best_model_path)
            print(f"  ✅ Best model found: {self.best_model_path}")
        else:
            print(f"  ❌ Best model not found: {self.best_model_path}")
        
        return models_found
    
    def load_model(self, model_path):
        """Load a trained model."""
        print(f"\n📥 Loading model from {model_path}...")
        
        try:
            model = FeatureBasedGestureModel()
            model.load_model(str(model_path))
            
            # Load preprocessing artifacts
            scaler_path = self.test_data_path / "feature_scaler.pkl"
            encoder_path = self.test_data_path / "label_encoder.pkl"
            
            if scaler_path.exists() and encoder_path.exists():
                import joblib
                model.feature_scaler = joblib.load(scaler_path)
                model.label_encoder = joblib.load(encoder_path)
                print("  ✅ Preprocessing artifacts loaded")
            else:
                print("  ⚠️  Preprocessing artifacts not found")
            
            return model
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            return None
    
    def test_with_sample_data(self, model):
        """Test model with sample gesture data."""
        print("\n🧪 Testing with sample data...")
        
        # Create sample gesture data
        sample_gestures = self._create_sample_gestures()
        
        results = []
        
        for gesture_name, landmarks in sample_gestures.items():
            print(f"\n  Testing {gesture_name} gesture...")
            
            # Extract features
            features = self._extract_features_from_landmarks(landmarks)
            
            if features is None:
                print(f"    ❌ Failed to extract features")
                continue
            
            # Make prediction
            try:
                prediction = model.predict(features.reshape(1, -1))
                predicted_gesture = prediction[0]
                confidence = np.max(model.model.predict(features.reshape(1, -1)))
                
                results.append({
                    'gesture': gesture_name,
                    'predicted': predicted_gesture,
                    'confidence': confidence,
                    'correct': predicted_gesture == gesture_name
                })
                
                status = "✅" if predicted_gesture == gesture_name else "❌"
                print(f"    {status} Predicted: {predicted_gesture} (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"    ❌ Prediction failed: {e}")
                results.append({
                    'gesture': gesture_name,
                    'predicted': 'error',
                    'confidence': 0.0,
                    'correct': False
                })
        
        return results
    
    def test_with_real_data(self, model):
        """Test model with real data from processed directory."""
        print("\n🧪 Testing with real data...")
        
        if not self.test_data_path.exists():
            print("  ❌ No test data found")
            return []
        
        # Load test data
        try:
            X_test = np.load(self.test_data_path / "X_test.npy")
            y_test = np.load(self.test_data_path / "y_test.npy")
            
            print(f"  📊 Test data loaded: {X_test.shape}")
            
            # Make predictions
            predictions = model.model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(predicted_classes == y_test)
            
            print(f"  📈 Test accuracy: {accuracy:.3f}")
            
            # Per-class accuracy
            unique_classes = np.unique(y_test)
            class_accuracies = {}
            
            for class_id in unique_classes:
                mask = y_test == class_id
                class_accuracy = np.mean(predicted_classes[mask] == y_test[mask])
                class_accuracies[class_id] = class_accuracy
            
            print(f"  📊 Per-class accuracy:")
            for class_id, acc in class_accuracies.items():
                class_name = model.label_encoder.classes_[class_id] if model.label_encoder else f"class_{class_id}"
                print(f"    {class_name:12}: {acc:.3f}")
            
            return [{
                'test_type': 'real_data',
                'accuracy': accuracy,
                'class_accuracies': class_accuracies
            }]
            
        except Exception as e:
            print(f"  ❌ Real data test failed: {e}")
            return []
    
    def _create_sample_gestures(self):
        """Create sample gesture landmarks for testing."""
        sample_gestures = {}
        
        # Fist gesture (all fingers closed)
        fist_landmarks = []
        for i in range(21):
            if i == 0:  # Wrist
                fist_landmarks.append({'x': 0.0, 'y': 0.0, 'z': 0.0})
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                fist_landmarks.append({'x': 0.1, 'y': 0.1, 'z': 0.0})
            else:  # Other landmarks
                fist_landmarks.append({'x': 0.05, 'y': 0.05, 'z': 0.0})
        sample_gestures['fist'] = fist_landmarks
        
        # Point gesture (index finger extended)
        point_landmarks = []
        for i in range(21):
            if i == 0:  # Wrist
                point_landmarks.append({'x': 0.0, 'y': 0.0, 'z': 0.0})
            elif i == 8:  # Index fingertip
                point_landmarks.append({'x': 0.0, 'y': 0.3, 'z': 0.0})
            elif i in [4, 12, 16, 20]:  # Other fingertips
                point_landmarks.append({'x': 0.1, 'y': 0.1, 'z': 0.0})
            else:  # Other landmarks
                point_landmarks.append({'x': 0.05, 'y': 0.05, 'z': 0.0})
        sample_gestures['point'] = point_landmarks
        
        # Thumbs up gesture
        thumbs_up_landmarks = []
        for i in range(21):
            if i == 0:  # Wrist
                thumbs_up_landmarks.append({'x': 0.0, 'y': 0.0, 'z': 0.0})
            elif i == 4:  # Thumb tip
                thumbs_up_landmarks.append({'x': 0.3, 'y': 0.0, 'z': 0.0})
            elif i in [8, 12, 16, 20]:  # Other fingertips
                thumbs_up_landmarks.append({'x': 0.1, 'y': 0.1, 'z': 0.0})
            else:  # Other landmarks
                thumbs_up_landmarks.append({'x': 0.05, 'y': 0.05, 'z': 0.0})
        sample_gestures['thumbs_up'] = thumbs_up_landmarks
        
        return sample_gestures
    
    def _extract_features_from_landmarks(self, landmarks):
        """Extract features from landmarks."""
        if len(landmarks) != 21:
            return None
        
        # Convert to numpy array
        landmarks_array = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        
        # Extract features (same as in preprocessing)
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
        
        # Convert to feature vector
        feature_vector = np.array([
            features['hand_span'],
            features['palm_area'],
            features['thumb_extension'],
            features['index_extension'],
            features['middle_extension'],
            features['ring_extension'],
            features['pinky_extension'],
            features['hand_center_x'],
            features['hand_center_y'],
            features['hand_center_z'],
            features['thumb_angle'],
            features['index_angle'],
            features['palm_orientation']
        ])
        
        return feature_vector
    
    def run_comprehensive_test(self):
        """Run comprehensive model testing."""
        print("🧪 Model Testing Suite")
        print("=" * 50)
        
        # 1. Check model availability
        models_found = self.check_model_availability()
        
        if not models_found:
            print("\n❌ No models found for testing!")
            print("   Please run 'python scripts/retrain_model.py' first")
            return False
        
        # 2. Test each available model
        all_results = {}
        
        for model_name, model_path in models_found.items():
            print(f"\n{'='*20} Testing {model_name} {'='*20}")
            
            # Load model
            model = self.load_model(model_path)
            if model is None:
                continue
            
            # Test with sample data
            sample_results = self.test_with_sample_data(model)
            
            # Test with real data
            real_results = self.test_with_real_data(model)
            
            all_results[model_name] = {
                'sample_results': sample_results,
                'real_results': real_results
            }
        
        # 3. Generate test summary
        self._generate_test_summary(all_results)
        
        return True
    
    def _generate_test_summary(self, all_results):
        """Generate comprehensive test summary."""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        for model_name, results in all_results.items():
            print(f"\n🔍 {model_name.upper()}:")
            
            # Sample data results
            sample_results = results['sample_results']
            if sample_results:
                correct_predictions = sum(1 for r in sample_results if r['correct'])
                total_predictions = len(sample_results)
                sample_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                print(f"  📝 Sample data accuracy: {sample_accuracy:.3f} ({correct_predictions}/{total_predictions})")
                
                for result in sample_results:
                    status = "✅" if result['correct'] else "❌"
                    print(f"    {status} {result['gesture']} → {result['predicted']} (conf: {result['confidence']:.3f})")
            
            # Real data results
            real_results = results['real_results']
            if real_results:
                for result in real_results:
                    if 'accuracy' in result:
                        print(f"  📊 Real data accuracy: {result['accuracy']:.3f}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        
        all_accuracies = []
        for model_name, results in all_results.items():
            # Sample data accuracy
            sample_results = results['sample_results']
            if sample_results:
                correct = sum(1 for r in sample_results if r['correct'])
                total = len(sample_results)
                if total > 0:
                    all_accuracies.append(correct / total)
            
            # Real data accuracy
            real_results = results['real_results']
            if real_results:
                for result in real_results:
                    if 'accuracy' in result:
                        all_accuracies.append(result['accuracy'])
        
        if all_accuracies:
            avg_accuracy = np.mean(all_accuracies)
            print(f"  📈 Average accuracy: {avg_accuracy:.3f}")
            
            if avg_accuracy >= 0.9:
                print("  🎉 Excellent performance!")
            elif avg_accuracy >= 0.8:
                print("  ✅ Good performance")
            elif avg_accuracy >= 0.7:
                print("  ⚠️  Acceptable performance")
            else:
                print("  ❌ Poor performance - consider retraining")
        else:
            print("  ⚠️  No accuracy metrics available")
        
        print(f"\n📋 Next steps:")
        print(f"   1. Test with real-time webcam input")
        print(f"   2. Deploy to your avatar system")
        print(f"   3. Monitor real-world performance")
        print(f"   4. Collect feedback and iterate")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test retrained gesture recognition model')
    parser.add_argument('--model', choices=['feature', 'best', 'all'], 
                       default='all', help='Which model to test')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ModelTester()
    
    # Run tests
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n✅ Model testing completed successfully!")
    else:
        print("\n❌ Model testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 