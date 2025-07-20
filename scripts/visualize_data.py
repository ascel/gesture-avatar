#!/usr/bin/env python3
"""
Gesture Data Visualization Script

This script provides comprehensive visualization of the gesture recognition dataset
used for training and validating the hand gesture model.

Usage:
    python scripts/visualize_data.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_gesture_data():
    """Load all gesture data from JSON files."""
    data_path = Path('../data')
    raw_path = data_path / 'raw'
    
    # List available gesture classes
    gesture_classes = [d.name for d in raw_path.iterdir() if d.is_dir()]
    print(f"Found gesture classes: {gesture_classes}")
    
    data = []
    
    for gesture_class in gesture_classes:
        gesture_path = raw_path / gesture_class
        
        for json_file in gesture_path.glob('*.json'):
            with open(json_file, 'r') as f:
                sample = json.load(f)
                
            # Extract landmarks
            landmarks = sample['landmarks']
            
            # Convert to numpy array
            coords = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
            
            data.append({
                'gesture': gesture_class,
                'sample_id': json_file.stem,
                'landmarks': coords,
                'timestamp': sample.get('timestamp', '')
            })
    
    return data, gesture_classes

def create_dataframe(gesture_data):
    """Create DataFrame for analysis."""
    df_data = []
    for sample in gesture_data:
        gesture = sample['gesture']
        sample_id = sample['sample_id']
        landmarks = sample['landmarks']
        
        for i, (x, y, z) in enumerate(landmarks):
            df_data.append({
                'gesture': gesture,
                'sample_id': sample_id,
                'landmark_id': i,
                'x': x,
                'y': y,
                'z': z
            })
    
    return pd.DataFrame(df_data)

def extract_hand_features(landmarks):
    """Extract hand features from landmarks (same as in the model)."""
    landmarks_array = np.array(landmarks)
    
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

def plot_dataset_overview(df, gesture_classes):
    """Plot dataset overview and distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sample count per gesture
    sample_counts = df.groupby('gesture')['sample_id'].nunique()
    axes[0].bar(sample_counts.index, sample_counts.values, color='skyblue', alpha=0.7)
    axes[0].set_title('Samples per Gesture Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gesture Class')
    axes[0].set_ylabel('Number of Samples')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(sample_counts.values):
        axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
    
    # Landmark count per gesture
    landmark_counts = df.groupby('gesture').size()
    axes[1].bar(landmark_counts.index, landmark_counts.values, color='lightcoral', alpha=0.7)
    axes[1].set_title('Total Landmarks per Gesture Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Gesture Class')
    axes[1].set_ylabel('Number of Landmarks')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(landmark_counts.values):
        axes[1].text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../docs/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sample_counts, landmark_counts

def plot_coordinate_distributions(df, gesture_classes):
    """Plot coordinate distributions for each gesture."""
    # X and Y coordinate distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # X coordinate distribution
    for i, gesture in enumerate(gesture_classes):
        gesture_data_subset = df[df['gesture'] == gesture]
        axes[0, i].hist(gesture_data_subset['x'], bins=20, alpha=0.7, color='skyblue')
        axes[0, i].set_title(f'{gesture.title()} - X Distribution')
        axes[0, i].set_xlabel('X Coordinate')
        axes[0, i].set_ylabel('Frequency')
    
    # Y coordinate distribution
    for i, gesture in enumerate(gesture_classes):
        gesture_data_subset = df[df['gesture'] == gesture]
        axes[1, i].hist(gesture_data_subset['y'], bins=20, alpha=0.7, color='lightcoral')
        axes[1, i].set_title(f'{gesture.title()} - Y Distribution')
        axes[1, i].set_xlabel('Y Coordinate')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../docs/coordinate_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Z coordinate distribution (depth)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, gesture in enumerate(gesture_classes):
        gesture_data_subset = df[df['gesture'] == gesture]
        axes[i].hist(gesture_data_subset['z'], bins=20, alpha=0.7, color='lightgreen')
        axes[i].set_title(f'{gesture.title()} - Z Distribution')
        axes[i].set_xlabel('Z Coordinate (Depth)')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('../docs/depth_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_analysis(feature_df, gesture_classes):
    """Plot feature analysis and distributions."""
    # Feature correlation heatmap
    numeric_features = feature_df.select_dtypes(include=[np.number]).columns
    correlation_matrix = feature_df[numeric_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../docs/feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Key features distribution
    key_features = ['hand_span', 'palm_area', 'thumb_extension', 'index_extension', 
                    'middle_extension', 'ring_extension', 'pinky_extension']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        if i < len(axes):
            for gesture in gesture_classes:
                gesture_data_subset = feature_df[feature_df['gesture'] == gesture]
                axes[i].hist(gesture_data_subset[feature], alpha=0.6, label=gesture, bins=10)
            
            axes[i].set_title(f'{feature.replace("_", " ").title()}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(key_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../docs/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_train_val_split(df, gesture_classes):
    """Plot training/validation split analysis."""
    # Create sample IDs for splitting
    sample_ids = df[['gesture', 'sample_id']].drop_duplicates()
    
    # Split data (80% train, 20% validation)
    train_samples, val_samples = train_test_split(
        sample_ids, test_size=0.2, random_state=42, stratify=sample_ids['gesture']
    )
    
    # Create train/val masks
    train_mask = df.merge(train_samples, on=['gesture', 'sample_id'], how='inner').index
    val_mask = df.merge(val_samples, on=['gesture', 'sample_id'], how='inner').index
    
    df_train = df.loc[train_mask]
    df_val = df.loc[val_mask]
    
    print("=== TRAINING/VALIDATION SPLIT ===")
    print(f"Training samples: {len(df_train['sample_id'].unique())}")
    print(f"Validation samples: {len(df_val['sample_id'].unique())}")
    print(f"Total samples: {len(df['sample_id'].unique())}")
    
    # Distribution by gesture
    train_dist = df_train.groupby('gesture')['sample_id'].nunique()
    val_dist = df_val.groupby('gesture')['sample_id'].nunique()
    
    print("\nTraining samples per gesture:")
    print(train_dist)
    print("\nValidation samples per gesture:")
    print(val_dist)
    
    # Visualize split
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training distribution
    axes[0].bar(train_dist.index, train_dist.values, color='lightblue', alpha=0.7)
    axes[0].set_title('Training Set Distribution', fontweight='bold')
    axes[0].set_xlabel('Gesture')
    axes[0].set_ylabel('Number of Samples')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Validation distribution
    axes[1].bar(val_dist.index, val_dist.values, color='lightcoral', alpha=0.7)
    axes[1].set_title('Validation Set Distribution', fontweight='bold')
    axes[1].set_xlabel('Gesture')
    axes[1].set_ylabel('Number of Samples')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../docs/train_val_split.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_train, df_val

def print_data_summary(gesture_data, df, gesture_classes):
    """Print comprehensive data summary."""
    print("\n=== DATA ANALYSIS SUMMARY ===")
    
    print("📊 Dataset Overview:")
    print(f"  • Total samples: {len(gesture_data)}")
    print(f"  • Gesture classes: {len(gesture_classes)}")
    print(f"  • Samples per class: {len(gesture_data) // len(gesture_classes)}")
    print(f"  • Landmarks per sample: 21")
    print(f"  • Features extracted: 13")
    
    print("\n🎯 Gesture Classes:")
    for gesture in gesture_classes:
        print(f"  • {gesture}")
    
    print("\n📈 Data Quality:")
    print(f"  • Missing values: {df.isnull().sum().sum()}")
    print(f"  • Coordinate range: X({df['x'].min():.2f}, {df['x'].max():.2f}), "
          f"Y({df['y'].min():.2f}, {df['y'].max():.2f}), Z({df['z'].min():.2f}, {df['z'].max():.2f})")
    
    print("\n🔧 Recommendations:")
    print("  • Consider collecting more samples per gesture class for better model training")
    print("  • Implement data augmentation techniques to increase dataset size")
    print("  • Add more diverse hand positions and orientations")
    print("  • Consider collecting data from different users for better generalization")
    print("  • Implement real-time data collection pipeline for continuous improvement")
    
    print("\n✅ Dataset is ready for model training with proper train/validation split!")

def main():
    """Main function to run all visualizations."""
    print("Loading gesture data...")
    gesture_data, gesture_classes = load_gesture_data()
    
    print("Creating DataFrame...")
    df = create_dataframe(gesture_data)
    
    print("Extracting features...")
    feature_data = []
    for sample in gesture_data:
        features = extract_hand_features(sample['landmarks'])
        features['gesture'] = sample['gesture']
        features['sample_id'] = sample['sample_id']
        feature_data.append(features)
    
    feature_df = pd.DataFrame(feature_data)
    
    print("Generating visualizations...")
    
    # Create docs directory if it doesn't exist
    docs_path = Path('../docs')
    docs_path.mkdir(exist_ok=True)
    
    # Generate all plots
    plot_dataset_overview(df, gesture_classes)
    plot_coordinate_distributions(df, gesture_classes)
    plot_feature_analysis(feature_df, gesture_classes)
    plot_train_val_split(df, gesture_classes)
    
    # Print summary
    print_data_summary(gesture_data, df, gesture_classes)
    
    print("\nAll visualizations completed! Check the 'docs' folder for saved images.")

if __name__ == "__main__":
    main() 