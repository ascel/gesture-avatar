#!/usr/bin/env python3
"""
Model Retraining Script with GPU Support

This script retrains the gesture recognition model using the expanded dataset
from multiple sources (raw, synthetic, external) with GPU acceleration.

Usage:
    conda run -n py310 python scripts/retrain_model.py [--data-source expanded|raw|combined]
    or
    python scripts/retrain_model.py [--data-source expanded|raw|combined]
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

def configure_gpu():
    """Configure GPU settings for optimal performance."""
    try:
        import tensorflow as tf
        print("🔍 Checking GPU availability...")
        
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✅ Memory growth enabled for {gpu.name}")
            
            # Set mixed precision for faster training
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("   ✅ Mixed precision enabled (float16)")
            except Exception as e:
                print(f"   ⚠️  Mixed precision failed: {e}")
            
            print("✅ GPU acceleration configured successfully!")
            return True
        else:
            print("⚠️  No GPU found, using CPU")
            print("   💡 To enable GPU support:")
            print("   • Install tensorflow-gpu: conda install tensorflow-gpu -c conda-forge")
            print("   • Or install CUDA toolkit and cuDNN")
            return False
    except ImportError:
        print("⚠️  TensorFlow not installed, using CPU")
        print("   💡 Install with: conda install tensorflow -c conda-forge")
        return False
    except Exception as e:
        print(f"⚠️  GPU configuration failed: {e}")
        print("   Continuing with CPU")
        return False

try:
    from gesture_detection.train_models import train_feature_model, train_image_model
    from utils.data_preprocessing import GestureDataPreprocessor
except ImportError:
    # Fallback imports for when running from scripts directory
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.gesture_detection.train_models import train_feature_model, train_image_model
    from src.utils.data_preprocessing import GestureDataPreprocessor

class ModelRetrainer:
    def __init__(self, data_source="expanded"):
        self.data_source = data_source
        self.base_path = Path("data")
        self.models_path = self.base_path / "models"
        self.models_path.mkdir(exist_ok=True)
        
        # Configure GPU acceleration
        self.gpu_available = configure_gpu()
        
        # Data paths
        self.raw_path = self.base_path / "raw"
        self.expanded_path = self.base_path / "expanded"
        self.external_path = self.base_path / "external"
        self.processed_path = self.base_path / "processed"
        
        # Results tracking
        self.training_results = {}
        
    def check_data_availability(self):
        """Check what data sources are available."""
        print("🔍 Checking data availability...")
        
        data_sources = {}
        
        # Check raw data
        if self.raw_path.exists():
            raw_samples = sum(len(list((self.raw_path / gesture).glob('*.json'))) 
                            for gesture in ['peace', 'fist', 'point', 'thumbs_up', 'open_hand'] 
                            if (self.raw_path / gesture).exists())
            data_sources['raw'] = raw_samples
            print(f"  📁 Raw data: {raw_samples} samples")
        
        # Check expanded data
        if self.expanded_path.exists():
            expanded_samples = sum(len(list((self.expanded_path / gesture).glob('*.json'))) 
                                 for gesture in ['clap', 'fist', 'point', 'thumbs_up', 'wave', 'peace', 'open_hand'] 
                                 if (self.expanded_path / gesture).exists())
            data_sources['expanded'] = expanded_samples
            print(f"  🔄 Expanded data: {expanded_samples} samples")
        
        # Check external data
        if self.external_path.exists():
            external_files = list(self.external_path.glob('*.csv')) + list(self.external_path.glob('*.json'))
            data_sources['external'] = len(external_files)
            print(f"  📥 External datasets: {len(external_files)} files")
        
        return data_sources
    
    def prepare_expanded_dataset(self):
        """Prepare the expanded dataset for training."""
        print("\n🔄 Preparing expanded dataset...")
        
        if not self.expanded_path.exists():
            print("❌ Expanded dataset not found!")
            print("   Run 'python scripts/expand_dataset.py' first")
            return False
        
        # Load all samples from expanded dataset
        all_samples = []
        gesture_counts = {}
        
        for gesture_dir in self.expanded_path.iterdir():
            if not gesture_dir.is_dir():
                continue
                
            gesture_name = gesture_dir.name
            gesture_samples = []
            
            for json_file in gesture_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    sample = json.load(f)
                
                # Standardize sample format
                standardized_sample = self._standardize_sample(sample, gesture_name)
                gesture_samples.append(standardized_sample)
            
            gesture_counts[gesture_name] = len(gesture_samples)
            all_samples.extend(gesture_samples)
        
        print(f"📊 Loaded {len(all_samples)} samples from expanded dataset:")
        for gesture, count in gesture_counts.items():
            print(f"   {gesture:12}: {count:4d} samples")
        
        # Save to processed directory
        self._save_processed_data(all_samples)
        
        return True
    
    def prepare_combined_dataset(self):
        """Prepare combined dataset from all sources."""
        print("\n🔄 Preparing combined dataset...")
        
        all_samples = []
        
        # 1. Load raw data
        if self.raw_path.exists():
            print("  📁 Loading raw data...")
            raw_samples = self._load_samples_from_directory(self.raw_path)
            all_samples.extend(raw_samples)
            print(f"     Loaded {len(raw_samples)} raw samples")
        
        # 2. Load expanded data
        if self.expanded_path.exists():
            print("  🔄 Loading expanded data...")
            expanded_samples = self._load_samples_from_directory(self.expanded_path)
            all_samples.extend(expanded_samples)
            print(f"     Loaded {len(expanded_samples)} expanded samples")
        
        # 3. Load external data (if available)
        if self.external_path.exists():
            print("  📥 Loading external data...")
            external_samples = self._load_external_data()
            all_samples.extend(external_samples)
            print(f"     Loaded {len(external_samples)} external samples")
        
        if not all_samples:
            print("❌ No data found in any source!")
            return False
        
        # Analyze dataset
        gesture_counts = {}
        source_counts = {}
        
        for sample in all_samples:
            gesture = sample['gesture']
            source = sample.get('source', 'unknown')
            
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n📊 Combined dataset statistics:")
        print(f"   Total samples: {len(all_samples)}")
        print(f"   Gesture classes: {len(gesture_counts)}")
        print(f"   Data sources: {len(source_counts)}")
        
        print(f"\n📈 Samples per gesture:")
        for gesture, count in sorted(gesture_counts.items()):
            print(f"   {gesture:12}: {count:4d} samples")
        
        print(f"\n📈 Samples per source:")
        for source, count in sorted(source_counts.items()):
            print(f"   {source:12}: {count:4d} samples")
        
        # Save to processed directory
        self._save_processed_data(all_samples)
        
        return True
    
    def _load_samples_from_directory(self, directory):
        """Load samples from a directory structure."""
        samples = []
        
        for gesture_dir in directory.iterdir():
            if not gesture_dir.is_dir():
                continue
                
            gesture_name = gesture_dir.name
            
            for json_file in gesture_dir.glob('*.json'):
                with open(json_file, 'r') as f:
                    sample = json.load(f)
                
                # Standardize sample format
                standardized_sample = self._standardize_sample(sample, gesture_name)
                samples.append(standardized_sample)
        
        return samples
    
    def _load_external_data(self):
        """Load external datasets if available."""
        samples = []
        
        # This would be implemented based on specific external dataset formats
        # For now, return empty list
        return samples
    
    def _standardize_sample(self, sample, gesture_name):
        """Standardize sample format for training."""
        # Ensure we have the required fields
        standardized = {
            'gesture': gesture_name,
            'landmarks': sample.get('landmarks', []),
            'timestamp': sample.get('timestamp', datetime.now().isoformat()),
            'sample_id': sample.get('sample_id', sample.get('id', 'unknown')),
            'source': sample.get('source', 'unknown')
        }
        
        # Add metadata
        metadata = sample.get('metadata', {})
        metadata.update({
            'original_id': sample.get('original_sample', sample.get('synthetic_id', sample.get('original_id', 'unknown'))),
            'confidence': sample.get('confidence', 1.0),
            'augmented': sample.get('augmented', False),
            'synthetic': sample.get('synthetic', False)
        })
        standardized['metadata'] = metadata
        
        return standardized
    
    def _save_processed_data(self, samples):
        """Save processed data in the format expected by the training script."""
        print("\n💾 Saving processed data...")
        
        # Create processed directory structure
        processed_dir = self.processed_path
        processed_dir.mkdir(exist_ok=True)
        
        # Group samples by gesture
        gesture_groups = {}
        for sample in samples:
            gesture = sample['gesture']
            if gesture not in gesture_groups:
                gesture_groups[gesture] = []
            gesture_groups[gesture].append(sample)
        
        # Save each gesture group
        for gesture, gesture_samples in gesture_groups.items():
            gesture_dir = processed_dir / gesture
            gesture_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(gesture_samples):
                filename = f"processed_{gesture}_{i:04d}.json"
                filepath = gesture_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(sample, f, indent=2)
        
        print(f"✅ Saved {len(samples)} processed samples to {processed_dir}")
    
    def run_preprocessing(self):
        """Run data preprocessing for training."""
        print("\n🔄 Running data preprocessing...")
        
        # Initialize preprocessor
        preprocessor = GestureDataPreprocessor(
            raw_data_dir=str(self.processed_path),
            processed_data_dir=str(self.processed_path)
        )
        
        try:
            # Prepare training data
            X_train, X_val, X_test, y_train, y_val, y_test, scaler, encoder = preprocessor.prepare_training_data()
            
            print(f"✅ Preprocessing completed!")
            print(f"   Training set: {X_train.shape}")
            print(f"   Validation set: {X_val.shape}")
            print(f"   Test set: {X_test.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return False
    
    def train_feature_model(self, epochs=200, batch_size=32):
        """Train the optimized feature-based model with GPU acceleration."""
        print("\n🧠 Training Optimized Feature-Based Model")
        print("=" * 60)
        print(f"🚀 GPU acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        print(f"📊 Training parameters:")
        print(f"   • Epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Data source: {self.data_source}")
        print()
        
        try:
            model, results = train_feature_model(
                data_dir=str(self.processed_path),
                model_save_dir=str(self.models_path),
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.training_results['feature_model'] = results
            
            print(f"✅ Feature model training completed!")
            print(f"   Final accuracy: {results['test_results']['accuracy']:.3f}")
            print(f"   Final F1-score: {results['test_results']['macro_avg_f1']:.3f}")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Feature model training failed: {e}")
            return None, None
    
    def train_image_model(self, epochs=100):
        """Train the optimized image-based model with GPU acceleration."""
        print("\n🖼️  Training Optimized Image-Based Model")
        print("=" * 60)
        print(f"🚀 GPU acceleration: {'Enabled' if self.gpu_available else 'Disabled'}")
        print(f"📊 Training parameters:")
        print(f"   • Epochs: {epochs}")
        print(f"   • Data source: {self.data_source}")
        print()
        
        try:
            model, results = train_image_model(
                data_dir=str(self.processed_path),
                model_save_dir=str(self.models_path),
                epochs=epochs
            )
            
            if results:
                self.training_results['image_model'] = results
                print(f"✅ Image model training completed!")
                print(f"   Final accuracy: {results['training_history']['final_accuracy']:.3f}")
                print(f"   Final validation accuracy: {results['training_history']['final_val_accuracy']:.3f}")
            else:
                print("⚠️  Image model training skipped (no image data available)")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Image model training failed: {e}")
            return None, None
    
    def compare_with_previous_model(self):
        """Compare new model performance with previous model."""
        print("\n📊 Comparing with previous model...")
        
        # Check if previous model exists
        previous_model_path = self.models_path / "best_gesture_model.h5"
        if not previous_model_path.exists():
            print("   No previous model found for comparison")
            return
        
        # Load previous model results
        previous_results_path = self.models_path / "feature_model_results.json"
        if previous_results_path.exists():
            with open(previous_results_path, 'r') as f:
                previous_results = json.load(f)
            
            if 'feature_model' in self.training_results:
                current_results = self.training_results['feature_model']
                
                print(f"   Previous model accuracy: {previous_results['test_results']['accuracy']:.3f}")
                print(f"   New model accuracy: {current_results['test_results']['accuracy']:.3f}")
                
                improvement = current_results['test_results']['accuracy'] - previous_results['test_results']['accuracy']
                print(f"   Improvement: {improvement:+.3f}")
                
                if improvement > 0:
                    print("   🎉 New model performs better!")
                elif improvement < 0:
                    print("   ⚠️  Previous model performed better")
                else:
                    print("   ➡️  Models perform similarly")
    
    def save_training_summary(self):
        """Save comprehensive training summary."""
        print("\n📋 Saving training summary...")
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'data_source': self.data_source,
            'models_trained': list(self.training_results.keys()),
            'results': self.training_results
        }
        
        summary_path = self.models_path / "retraining_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Training summary saved to {summary_path}")
    
    def run_retraining(self, epochs=100, batch_size=32):
        """Run the complete retraining pipeline."""
        print("🚀 Starting Model Retraining")
        print("=" * 50)
        
        # 1. Check data availability
        data_sources = self.check_data_availability()
        
        if not any(data_sources.values()):
            print("❌ No data found! Please collect or expand your dataset first.")
            return False
        
        # 2. Prepare dataset
        if self.data_source == "expanded":
            success = self.prepare_expanded_dataset()
        elif self.data_source == "combined":
            success = self.prepare_combined_dataset()
        else:
            print(f"❌ Unknown data source: {self.data_source}")
            return False
        
        if not success:
            return False
        
        # 3. Run preprocessing
        if not self.run_preprocessing():
            return False
        
        # 4. Train models
        feature_model, feature_results = self.train_feature_model(epochs, batch_size)
        image_model, image_results = self.train_image_model(epochs // 2)  # Fewer epochs for image model
        
        # 5. Compare with previous model
        self.compare_with_previous_model()
        
        # 6. Save summary
        self.save_training_summary()
        
        print("\n🎉 Retraining completed!")
        print("\n📋 Next steps:")
        print("   1. Test the new model with real-time gesture recognition")
        print("   2. Deploy the updated model to your avatar system")
        print("   3. Monitor performance and collect feedback")
        print("   4. Consider collecting more data for further improvements")
        
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Retrain gesture recognition model with GPU acceleration')
    parser.add_argument('--data-source', choices=['expanded', 'raw', 'combined'], 
                       default='combined', help='Data source to use for training')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of training epochs (increased for ResNet training)')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Training batch size')
    
    args = parser.parse_args()
    
    # Initialize retrainer (GPU will be configured automatically)
    retrainer = ModelRetrainer(data_source=args.data_source)
    
    print(f"\n🎯 Retraining Configuration:")
    print(f"   • Data source: {args.data_source}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • GPU acceleration: {'Enabled' if retrainer.gpu_available else 'Disabled'}")
    print()
    
    # Run retraining
    success = retrainer.run_retraining(epochs=args.epochs, batch_size=args.batch_size)
    
    if success:
        print("\n✅ Retraining completed successfully!")
    else:
        print("\n❌ Retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 