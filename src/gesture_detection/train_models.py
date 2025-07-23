"""
Training script for gesture recognition models.
Trains feature-based and image-based models for gesture detection.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .models import FeatureBasedGestureModel, ImageBasedGestureModel, EfficientNetGestureModel
from ..utils.data_preprocessing import GestureDataPreprocessor


def train_feature_model(data_dir: str = "data/processed", 
                       model_save_dir: str = "data/models",
                       epochs: int = 200,
                       batch_size: int = 32):
    """Train optimized ResNet-inspired feature-based gesture recognition model."""
    print("🚀 Training Optimized ResNet Feature-Based Gesture Model")
    print("=" * 60)
    
    # Load preprocessed data
    data_path = Path(data_dir)
    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Load preprocessing artifacts
    feature_scaler = None
    label_encoder = None
    
    scaler_path = data_path / "feature_scaler.pkl"
    encoder_path = data_path / "label_encoder.pkl"
    
    if scaler_path.exists():
        import joblib
        feature_scaler = joblib.load(scaler_path)
    
    if encoder_path.exists():
        import joblib
        label_encoder = joblib.load(encoder_path)
    
    # Initialize model
    model = FeatureBasedGestureModel()
    model.feature_scaler = feature_scaler
    model.label_encoder = label_encoder
    
    # Train model with optimized settings
    print("\n🚀 Starting optimized training...")
    print("   • Using ResNet-inspired architecture")
    print("   • Residual connections for better gradient flow")
    print("   • Attention mechanisms for feature focus")
    print("   • Advanced optimization with AdamW")
    print("   • Class balancing for imbalanced datasets")
    print()
    
    history = model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate classification report
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = [f"gesture_{i}" for i in range(len(np.unique(y_test)))]
    
    report = classification_report(y_test, y_pred_classes, 
                                 target_names=target_names, output_dict=True)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    
    # Save model and results
    model_save_path = Path(model_save_dir) / "feature_gesture_model.h5"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(model_save_path))
    
    # Save training results with enhanced metrics
    results = {
        'model_type': 'resnet_feature_based',
        'architecture': 'ResNet-inspired with attention',
        'training_date': datetime.now().isoformat(),
        'model_path': str(model_save_path),
        'training_history': {
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['accuracy']),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_accuracy': float(max(history.history['accuracy']))
        },
        'test_results': {
            'accuracy': float(report['accuracy']),
            'macro_avg_f1': float(report['macro avg']['f1-score']),
            'weighted_avg_f1': float(report['weighted avg']['f1-score'])
        },
        'class_metrics': {name: metrics for name, metrics in report.items() 
                         if name not in ['accuracy', 'macro avg', 'weighted avg']},
        'optimization_features': [
            'Residual connections',
            'Attention mechanisms', 
            'Batch normalization',
            'AdamW optimizer',
            'Class balancing',
            'Advanced callbacks'
        ]
    }
    
    results_path = model_save_path.parent / "feature_model_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, model_save_path.parent / "feature_training_history.png")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, target_names, 
                         model_save_path.parent / "feature_confusion_matrix.png")
    
    print(f"\n✅ Optimized model training completed!")
    print(f"📁 Model saved to: {model_save_path}")
    print(f"📊 Results saved to: {results_path}")
    print(f"🏆 Best validation accuracy: {results['training_history']['best_val_accuracy']:.4f}")
    print(f"🎯 Test accuracy: {results['test_results']['accuracy']:.4f}")
    
    return model, results


def train_image_model(data_dir: str = "data/raw",
                     model_save_dir: str = "data/models",
                     epochs: int = 100):
    """Train optimized ResNet-inspired image-based gesture recognition model."""
    print("🚀 Training Optimized ResNet Image-Based Gesture Model")
    print("=" * 60)
    
    # Initialize preprocessor to create image dataset
    preprocessor = GestureDataPreprocessor(raw_data_dir=data_dir)
    
    try:
        # Create image dataset
        train_generator, val_generator, class_indices = preprocessor.create_image_dataset()
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {list(class_indices.keys())}")
        
        # Initialize model
        model = ImageBasedGestureModel()
        
        # Train model with optimized settings
        print("\n🚀 Starting optimized image model training...")
        print("   • Using ResNet-inspired 2D architecture")
        print("   • Convolutional residual connections")
        print("   • Multi-head attention mechanisms")
        print("   • Advanced optimization with AdamW")
        print("   • Progressive feature refinement")
        print()
        
        history = model.train(train_generator, val_generator, epochs)
        
        # Save model
        model_save_path = Path(model_save_dir) / "image_gesture_model.h5"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model.save_model(str(model_save_path))
        
        # Save training results with enhanced metrics
        results = {
            'model_type': 'resnet_image_based',
            'architecture': 'ResNet-inspired 2D with multi-head attention',
            'training_date': datetime.now().isoformat(),
            'model_path': str(model_save_path),
            'class_indices': class_indices,
            'training_history': {
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['accuracy']),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_accuracy': float(max(history.history['accuracy']))
            },
            'optimization_features': [
                '2D Residual connections',
                'Multi-head attention (4 heads)',
                'Convolutional layers',
                'Batch normalization',
                'AdamW optimizer',
                'Progressive feature refinement'
            ]
        }
        
        results_path = model_save_path.parent / "image_model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot training history
        plot_training_history(history, model_save_path.parent / "image_training_history.png")
        
        print(f"\n✅ Optimized image model training completed!")
        print(f"📁 Model saved to: {model_save_path}")
        print(f"📊 Results saved to: {results_path}")
        print(f"🏆 Best validation accuracy: {results['training_history']['best_val_accuracy']:.4f}")
        print(f"🎯 Final training accuracy: {results['training_history']['final_accuracy']:.4f}")
        
        return model, results
        
    except Exception as e:
        print(f"Error training image model: {e}")
        print("Make sure you have collected image data first.")
        return None, None


def train_efficientnet_model(data_dir: str = "data/raw",
                            model_save_dir: str = "data/models",
                            epochs: int = 50):
    """Train EfficientNet-based image gesture recognition model."""
    print("🚀 Training EfficientNet Image-Based Gesture Model")
    print("=" * 60)
    preprocessor = GestureDataPreprocessor(raw_data_dir=data_dir)
    try:
        train_generator, val_generator, class_indices = preprocessor.create_image_dataset()
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {list(class_indices.keys())}")
        model = EfficientNetGestureModel()
        model.class_indices = class_indices
        history = model.train(train_generator, val_generator, epochs)
        model_save_path = Path(model_save_dir) / "efficientnet_gesture_model.h5"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))
        results = {
            'model_type': 'efficientnet_image_based',
            'architecture': 'EfficientNetB0',
            'training_date': datetime.now().isoformat(),
            'model_path': str(model_save_path),
            'class_indices': class_indices,
            'training_history': {
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['accuracy']),
                'best_val_accuracy': float(max(history.history['val_accuracy'])),
                'best_accuracy': float(max(history.history['accuracy']))
            },
            'optimization_features': [
                'EfficientNetB0',
                'GlobalAveragePooling',
                'Adam optimizer',
                'EarlyStopping',
                'ModelCheckpoint',
                'ReduceLROnPlateau'
            ]
        }
        results_path = model_save_path.parent / "efficientnet_model_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ EfficientNet model training completed!")
        print(f"📁 Model saved to: {model_save_path}")
        print(f"📊 Results saved to: {results_path}")
        print(f"🏆 Best validation accuracy: {results['training_history']['best_val_accuracy']:.4f}")
        print(f"🎯 Final training accuracy: {results['training_history']['final_accuracy']:.4f}")
        return model, results
    except Exception as e:
        print(f"Error training EfficientNet model: {e}")
        print("Make sure you have collected image data first.")
        return None, None


def train_efficientnet1d_model(data_dir: str = "data/processed", 
                              model_save_dir: str = "data/models", 
                              epochs: int = 200, 
                              batch_size: int = 32):
    """Train EfficientNet1D-based feature gesture recognition model."""
    print("🚀 Training EfficientNet1D Landmark Gesture Model")
    print("=" * 60)
    data_path = Path(data_dir)
    X_train = np.load(data_path / "X_train.npy")
    X_val = np.load(data_path / "X_val.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_train = np.load(data_path / "y_train.npy")
    y_val = np.load(data_path / "y_val.npy")
    y_test = np.load(data_path / "y_test.npy")
    feature_scaler = None
    label_encoder = None
    scaler_path = data_path / "feature_scaler.pkl"
    encoder_path = data_path / "label_encoder.pkl"
    if scaler_path.exists():
        import joblib
        feature_scaler = joblib.load(scaler_path)
    if encoder_path.exists():
        import joblib
        label_encoder = joblib.load(encoder_path)
    from .models import EfficientNet1DLandmarkModel
    model = EfficientNet1DLandmarkModel()
    model.feature_scaler = feature_scaler
    model.label_encoder = label_encoder
    history = model.train(X_train, y_train, X_val, y_val, epochs, batch_size)
    print("\nEvaluating model...")
    X_test_reshaped = np.expand_dims(X_test, axis=-1)
    y_pred = model.model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)
    if label_encoder is not None:
        target_names = label_encoder.classes_
    else:
        target_names = [f"gesture_{i}" for i in range(len(np.unique(y_test)))]
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred_classes, target_names=target_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    model_save_path = Path(model_save_dir) / "efficientnet1d_gesture_model.h5"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_save_path))
    results = {
        'model_type': 'efficientnet1d_feature_based',
        'architecture': 'EfficientNet1D-inspired',
        'training_date': datetime.now().isoformat(),
        'model_path': str(model_save_path),
        'training_history': {
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'epochs_trained': len(history.history['accuracy']),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'best_accuracy': float(max(history.history['accuracy']))
        },
        'test_results': {
            'accuracy': float(report['accuracy']),
            'macro_avg_f1': float(report['macro avg']['f1-score']),
            'weighted_avg_f1': float(report['weighted avg']['f1-score'])
        },
        'class_metrics': {name: metrics for name, metrics in report.items() if name not in ['accuracy', 'macro avg', 'weighted avg']},
        'optimization_features': [
            '1D Conv blocks',
            'Batch normalization',
            'GlobalAveragePooling1D',
            'Adam optimizer',
            'EarlyStopping',
            'ModelCheckpoint',
            'ReduceLROnPlateau'
        ]
    }
    results_path = model_save_path.parent / "efficientnet1d_model_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ EfficientNet1D model training completed!")
    print(f"📁 Model saved to: {model_save_path}")
    print(f"📊 Results saved to: {results_path}")
    print(f"🏆 Best validation accuracy: {results['training_history']['best_val_accuracy']:.4f}")
    print(f"🎯 Test accuracy: {results['test_results']['accuracy']:.4f}")
    return model, results


def plot_training_history(history, save_path: Path):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, target_names, save_path: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {save_path}")


def evaluate_model_performance(model_path: str, test_data_path: str = "data/processed"):
    """Evaluate a trained model's performance."""
    print("Model Performance Evaluation")
    print("============================")
    
    # Load test data
    test_path = Path(test_data_path)
    X_test = np.load(test_path / "X_test.npy")
    y_test = np.load(test_path / "y_test.npy")
    
    # Load model
    if "feature" in model_path:
        model = FeatureBasedGestureModel(model_path)
    else:
        model = ImageBasedGestureModel(model_path)
    
    # Make predictions
    y_pred = model.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_test)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Samples: {len(y_test)}")
    
    return accuracy


def main():
    """Main training function for optimized ResNet-inspired models."""
    print("🚀 Optimized ResNet Gesture Recognition Model Training")
    print("=" * 60)
    print("Features:")
    print("  • ResNet-inspired architectures")
    print("  • Residual connections for better gradient flow")
    print("  • Attention mechanisms for feature focus")
    print("  • Advanced optimization with AdamW")
    print("  • Class balancing for imbalanced datasets")
    print("  • Smart callbacks and early stopping")
    print()
    
    # Configuration for optimized models
    config = {
        'feature_model': {
            'epochs': 200,  # More epochs for ResNet training
            'batch_size': 32
        },
        'image_model': {
            'epochs': 100  # More epochs for ResNet training
        }
    }
    
    try:
        # Train feature-based model
        print("\n1. 🎯 Training Optimized Feature-Based Model...")
        feature_model, feature_results = train_feature_model(
            epochs=config['feature_model']['epochs'],
            batch_size=config['feature_model']['batch_size']
        )
        
        if feature_results:
            print(f"✅ Optimized feature model training completed!")
            print(f"  🏆 Best validation accuracy: {feature_results['training_history']['best_val_accuracy']:.4f}")
            print(f"  🎯 Test accuracy: {feature_results['test_results']['accuracy']:.4f}")
            print(f"  📊 F1 score: {feature_results['test_results']['macro_avg_f1']:.4f}")
        
        # Train image-based model (if data available)
        print("\n2. 🖼️  Training Optimized Image-Based Model...")
        image_model, image_results = train_image_model(
            epochs=config['image_model']['epochs']
        )
        
        if image_results:
            print(f"✅ Optimized image model training completed!")
            print(f"  🏆 Best validation accuracy: {image_results['training_history']['best_val_accuracy']:.4f}")
            print(f"  🎯 Final training accuracy: {image_results['training_history']['final_accuracy']:.4f}")
        
        # Train EfficientNet model (if data available)
        print("\n3. 🦾 Training EfficientNet Image-Based Model...")
        efficientnet_model, efficientnet_results = train_efficientnet_model(
            epochs=50
        )
        if efficientnet_results:
            print(f"✅ EfficientNet model training completed!")
            print(f"  🏆 Best validation accuracy: {efficientnet_results['training_history']['best_val_accuracy']:.4f}")
            print(f"  🎯 Final training accuracy: {efficientnet_results['training_history']['final_accuracy']:.4f}")
        
        # Train EfficientNet1D model (if data available)
        print("\n4. 🦾 Training EfficientNet1D Landmark Model...")
        efficientnet1d_model, efficientnet1d_results = train_efficientnet1d_model(
            epochs=config['feature_model']['epochs'],
            batch_size=config['feature_model']['batch_size']
        )
        if efficientnet1d_results:
            print(f"✅ EfficientNet1D model training completed!")
            print(f"  🏆 Best validation accuracy: {efficientnet1d_results['training_history']['best_val_accuracy']:.4f}")
            print(f"  🎯 Test accuracy: {efficientnet1d_results['test_results']['accuracy']:.4f}")
        
        # Generate summary report
        print("\n5. 📊 Generating Summary Report...")
        generate_training_summary(feature_results, image_results)
        
        print("\n🎉 Optimized training completed successfully!")
        print("\n📋 Next steps:")
        print("1. Test optimized models: python test_optimized_model.py")
        print("2. Run enhanced demo: python src/main.py")
        print("3. Check performance plots and confusion matrices in data/models/")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure you have preprocessed data available.")


def generate_training_summary(feature_results, image_results):
    """Generate a summary report of optimized ResNet training results."""
    summary = {
        'training_date': datetime.now().isoformat(),
        'architecture_type': 'ResNet-inspired with attention',
        'models_trained': [],
        'overall_performance': {},
        'optimization_features': [
            'Residual connections',
            'Attention mechanisms',
            'Batch normalization', 
            'AdamW optimizer',
            'Class balancing',
            'Advanced callbacks'
        ]
    }
    
    if feature_results:
        summary['models_trained'].append({
            'type': 'resnet_feature_based',
            'architecture': feature_results.get('architecture', 'ResNet-inspired with attention'),
            'accuracy': feature_results['test_results']['accuracy'],
            'f1_score': feature_results['test_results']['macro_avg_f1'],
            'best_val_accuracy': feature_results['training_history']['best_val_accuracy'],
            'model_path': feature_results['model_path']
        })
    
    if image_results:
        summary['models_trained'].append({
            'type': 'resnet_image_based',
            'architecture': image_results.get('architecture', 'ResNet-inspired 2D with multi-head attention'),
            'accuracy': image_results['training_history']['final_accuracy'],
            'best_val_accuracy': image_results['training_history']['best_val_accuracy'],
            'model_path': image_results['model_path']
        })
    
    # Calculate overall performance
    if summary['models_trained']:
        accuracies = [m['accuracy'] for m in summary['models_trained'] if 'accuracy' in m]
        f1_scores = [m['f1_score'] for m in summary['models_trained'] if 'f1_score' in m]
        
        summary['overall_performance'] = {
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'avg_f1_score': np.mean(f1_scores) if f1_scores else 0,
            'best_model': max(summary['models_trained'], 
                            key=lambda x: x.get('accuracy', 0))['type']
        }
    
    # Save summary
    summary_path = Path("data/models/training_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Optimized training summary saved to: {summary_path}")
    
    # Print summary
    print("\n🏆 Optimized Training Summary:")
    print("=" * 40)
    for model in summary['models_trained']:
        print(f"\n{model['type'].replace('_', ' ').title()}:")
        print(f"  🏗️  Architecture: {model['architecture']}")
        print(f"  🎯 Accuracy: {model['accuracy']:.4f}")
        print(f"  🏆 Best Val Accuracy: {model['best_val_accuracy']:.4f}")
        if 'f1_score' in model:
            print(f"  📊 F1 Score: {model['f1_score']:.4f}")
    
    if summary['overall_performance']:
        print(f"\n📈 Overall Performance:")
        print(f"  📊 Average Accuracy: {summary['overall_performance']['avg_accuracy']:.4f}")
        print(f"  🎯 Average F1 Score: {summary['overall_performance']['avg_f1_score']:.4f}")
        print(f"  🏆 Best Model: {summary['overall_performance']['best_model']}")
    
    print(f"\n🚀 Optimization Features:")
    for feature in summary['optimization_features']:
        print(f"  • {feature}")


if __name__ == "__main__":
    main() 