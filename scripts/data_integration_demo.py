#!/usr/bin/env python3
"""
Data Integration Demo Script

This script demonstrates how the system processes and labels data from 3 sources:
1. Raw data (existing samples)
2. Synthetic data (generated)
3. External datasets (downloaded)

Usage:
    python scripts/data_integration_demo.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import random
from datetime import datetime

class DataIntegrationDemo:
    def __init__(self):
        self.data_path = Path("../data")
        self.raw_path = self.data_path / "raw"
        self.expanded_path = self.data_path / "expanded"
        self.external_path = self.data_path / "external"
        
        # Gesture classes
        self.gesture_classes = ['clap', 'fist', 'point', 'thumbs_up', 'wave', 'peace', 'open_hand']
        
    def demo_source_1_raw_data_processing(self):
        """Demo: Process existing raw data with augmentation."""
        print("=" * 60)
        print("SOURCE 1: RAW DATA PROCESSING & AUGMENTATION")
        print("=" * 60)
        
        # Load existing raw data
        raw_samples = []
        for gesture in self.gesture_classes:
            gesture_path = self.raw_path / gesture
            if gesture_path.exists():
                for json_file in gesture_path.glob('*.json'):
                    with open(json_file, 'r') as f:
                        sample = json.load(f)
                        raw_samples.append(sample)
        
        print(f"📁 Found {len(raw_samples)} existing raw samples")
        
        # Process and augment each sample
        augmented_samples = []
        for i, sample in enumerate(raw_samples):
            print(f"\n🔄 Processing sample {i+1}: {sample.get('gesture', 'unknown')}")
            
            # Extract landmarks
            landmarks = sample['landmarks']
            print(f"   Original landmarks: {len(landmarks)} points")
            
            # Apply augmentation techniques
            augmented_sample = self._augment_sample(sample)
            augmented_samples.append(augmented_sample)
            
            print(f"   ✅ Created augmented version")
            
            # Show sample structure
            if i == 0:  # Show structure for first sample
                print(f"\n📋 Sample structure:")
                print(f"   - Gesture: {augmented_sample['gesture']}")
                print(f"   - Landmarks: {len(augmented_sample['landmarks'])} points")
                print(f"   - Source: {augmented_sample['source']}")
                print(f"   - Augmented: {augmented_sample['augmented']}")
        
        return augmented_samples
    
    def demo_source_2_synthetic_data_generation(self):
        """Demo: Generate synthetic data with predefined templates."""
        print("\n" + "=" * 60)
        print("SOURCE 2: SYNTHETIC DATA GENERATION")
        print("=" * 60)
        
        # Define gesture templates
        gesture_templates = {
            'fist': {
                'finger_extensions': [0.1, 0.1, 0.1, 0.1, 0.1],  # All fingers closed
                'description': 'All fingers curled into fist'
            },
            'open_hand': {
                'finger_extensions': [0.8, 0.9, 0.9, 0.8, 0.7],  # All fingers extended
                'description': 'All fingers extended and spread'
            },
            'point': {
                'finger_extensions': [0.2, 0.9, 0.1, 0.1, 0.1],  # Only index extended
                'description': 'Index finger pointing, others closed'
            },
            'peace': {
                'finger_extensions': [0.2, 0.9, 0.9, 0.1, 0.1],  # Index and middle extended
                'description': 'Index and middle fingers extended'
            },
            'thumbs_up': {
                'finger_extensions': [0.8, 0.1, 0.1, 0.1, 0.1],  # Only thumb extended
                'description': 'Thumb up, other fingers closed'
            }
        }
        
        synthetic_samples = []
        
        for gesture, template in gesture_templates.items():
            print(f"\n🤖 Generating synthetic {gesture} samples...")
            print(f"   Template: {template['description']}")
            
            # Generate multiple variations
            for i in range(5):  # Generate 5 samples per gesture
                synthetic_sample = self._generate_synthetic_sample(gesture, template, i)
                synthetic_samples.append(synthetic_sample)
                
                print(f"   ✅ Generated sample {i+1}")
                
                # Show first sample structure
                if i == 0:
                    print(f"   📋 Sample structure:")
                    print(f"      - Gesture: {synthetic_sample['gesture']}")
                    print(f"      - Landmarks: {len(synthetic_sample['landmarks'])} points")
                    print(f"      - Source: {synthetic_sample['source']}")
                    print(f"      - Template: {synthetic_sample['template']}")
        
        return synthetic_samples
    
    def demo_source_3_external_dataset_integration(self):
        """Demo: Integrate external datasets with format conversion."""
        print("\n" + "=" * 60)
        print("SOURCE 3: EXTERNAL DATASET INTEGRATION")
        print("=" * 60)
        
        # Create mock external dataset (simulating downloaded data)
        external_samples = self._create_mock_external_dataset()
        
        print(f"📥 Found {len(external_samples)} external samples")
        
        # Process and convert external data
        integrated_samples = []
        
        for i, external_sample in enumerate(external_samples):
            print(f"\n🔗 Processing external sample {i+1}...")
            
            # Convert format to match our structure
            converted_sample = self._convert_external_format(external_sample)
            integrated_samples.append(converted_sample)
            
            print(f"   Original format: {external_sample['format']}")
            print(f"   Converted to: {converted_sample['gesture']}")
            print(f"   ✅ Integration complete")
            
            # Show conversion example
            if i == 0:
                print(f"\n📋 Conversion example:")
                print(f"   Before: {external_sample['original_data'][:3]}...")  # Show first 3 landmarks
                print(f"   After:  {converted_sample['landmarks'][:3]}...")
        
        return integrated_samples
    
    def demo_data_combination_and_labeling(self, source1_data, source2_data, source3_data):
        """Demo: Combine all data sources and show final labeling."""
        print("\n" + "=" * 60)
        print("DATA COMBINATION & FINAL LABELING")
        print("=" * 60)
        
        # Combine all data sources
        all_data = source1_data + source2_data + source3_data
        
        print(f"📊 Combined dataset statistics:")
        print(f"   Source 1 (Raw + Augmented): {len(source1_data)} samples")
        print(f"   Source 2 (Synthetic): {len(source2_data)} samples")
        print(f"   Source 3 (External): {len(source3_data)} samples")
        print(f"   Total: {len(all_data)} samples")
        
        # Group by gesture and source
        gesture_source_stats = {}
        for sample in all_data:
            gesture = sample['gesture']
            source = sample['source']
            
            if gesture not in gesture_source_stats:
                gesture_source_stats[gesture] = {'raw': 0, 'synthetic': 0, 'external': 0}
            
            gesture_source_stats[gesture][source] += 1
        
        print(f"\n📈 Data distribution by gesture and source:")
        for gesture, stats in gesture_source_stats.items():
            print(f"   {gesture:12}: Raw({stats['raw']:3d}) | Synthetic({stats['synthetic']:3d}) | External({stats['external']:3d})")
        
        # Show labeling consistency
        print(f"\n🏷️  Labeling consistency check:")
        for sample in all_data[:3]:  # Show first 3 samples
            print(f"   Sample: {sample['gesture']} | Source: {sample['source']} | Landmarks: {len(sample['landmarks'])}")
        
        # Create final dataset structure
        final_dataset = self._create_final_dataset(all_data)
        
        print(f"\n✅ Final dataset created:")
        print(f"   - Total samples: {len(final_dataset)}")
        print(f"   - Gesture classes: {len(set(s['gesture'] for s in final_dataset))}")
        print(f"   - Data sources: {set(s['source'] for s in final_dataset)}")
        
        return final_dataset
    
    def _augment_sample(self, sample):
        """Apply augmentation to a single sample."""
        landmarks = np.array([[lm['x'], lm['y'], lm['z']] for lm in sample['landmarks']])
        
        # Apply random transformations
        augmented_landmarks = landmarks.copy()
        
        # Random rotation
        angle = np.random.uniform(-10, 10) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        augmented_landmarks = augmented_landmarks @ rotation_matrix.T
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        augmented_landmarks *= scale
        
        # Random noise
        noise = np.random.normal(0, 0.005, augmented_landmarks.shape)
        augmented_landmarks += noise
        
        # Convert back to list format
        landmarks_list = []
        for x, y, z in augmented_landmarks:
            landmarks_list.append({'x': float(x), 'y': float(y), 'z': float(z)})
        
        return {
            'gesture': sample['gesture'],
            'landmarks': landmarks_list,
            'timestamp': datetime.now().isoformat(),
            'source': 'raw',
            'augmented': True,
            'original_sample': sample.get('sample_id', 'unknown')
        }
    
    def _generate_synthetic_sample(self, gesture, template, index):
        """Generate a synthetic sample based on template."""
        # Create base hand landmarks
        base_landmarks = self._get_base_hand_landmarks()
        
        # Apply gesture template
        modified_landmarks = self._apply_gesture_template(base_landmarks, template)
        
        # Add randomness
        modified_landmarks += np.random.normal(0, 0.01, modified_landmarks.shape)
        
        # Convert to list format
        landmarks_list = []
        for x, y, z in modified_landmarks:
            landmarks_list.append({'x': float(x), 'y': float(y), 'z': float(z)})
        
        return {
            'gesture': gesture,
            'landmarks': landmarks_list,
            'timestamp': datetime.now().isoformat(),
            'source': 'synthetic',
            'template': template,
            'synthetic_id': f'synthetic_{gesture}_{index}'
        }
    
    def _get_base_hand_landmarks(self):
        """Get base hand landmark positions."""
        return np.array([
            [0.0, 0.0, 0.0],  # Wrist
            [0.1, 0.0, 0.0], [0.15, 0.0, 0.0], [0.2, 0.0, 0.0], [0.25, 0.0, 0.0],  # Thumb
            [0.0, 0.1, 0.0], [0.0, 0.15, 0.0], [0.0, 0.2, 0.0], [0.0, 0.25, 0.0],  # Index
            [0.0, 0.2, 0.0], [0.0, 0.25, 0.0], [0.0, 0.3, 0.0], [0.0, 0.35, 0.0],  # Middle
            [0.0, 0.3, 0.0], [0.0, 0.35, 0.0], [0.0, 0.4, 0.0], [0.0, 0.45, 0.0],  # Ring
            [0.0, 0.4, 0.0], [0.0, 0.45, 0.0], [0.0, 0.5, 0.0], [0.0, 0.55, 0.0]   # Pinky
        ])
    
    def _apply_gesture_template(self, landmarks, template):
        """Apply gesture template modifications."""
        modified = landmarks.copy()
        finger_extensions = template['finger_extensions']
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky tips
        
        for i, (tip_idx, extension) in enumerate(zip(finger_tips, finger_extensions)):
            if tip_idx < len(modified):
                base_pos = modified[tip_idx - 1]
                direction = modified[tip_idx] - base_pos
                modified[tip_idx] = base_pos + direction * extension
        
        return modified
    
    def _create_mock_external_dataset(self):
        """Create mock external dataset for demonstration."""
        external_formats = [
            {'format': 'CSV', 'separator': ','},
            {'format': 'JSON', 'separator': None},
            {'format': 'XML', 'separator': None}
        ]
        
        external_samples = []
        for i in range(10):
            format_info = random.choice(external_formats)
            gesture = random.choice(self.gesture_classes)
            
            # Mock external data format
            external_sample = {
                'id': f'external_{i:03d}',
                'format': format_info['format'],
                'original_data': f'[{random.random():.3f}, {random.random():.3f}, {random.random():.3f}]',
                'gesture_label': gesture,
                'confidence': random.uniform(0.8, 0.99),
                'timestamp': datetime.now().isoformat()
            }
            external_samples.append(external_sample)
        
        return external_samples
    
    def _convert_external_format(self, external_sample):
        """Convert external format to our standard format."""
        # Simulate format conversion
        gesture = external_sample['gesture_label']
        
        # Generate landmarks based on gesture
        landmarks = []
        for i in range(21):  # 21 hand landmarks
            landmarks.append({
                'x': random.uniform(-1, 1),
                'y': random.uniform(-1, 1),
                'z': random.uniform(-1, 1)
            })
        
        return {
            'gesture': gesture,
            'landmarks': landmarks,
            'timestamp': external_sample['timestamp'],
            'source': 'external',
            'original_id': external_sample['id'],
            'confidence': external_sample['confidence']
        }
    
    def _create_final_dataset(self, all_data):
        """Create final dataset with consistent structure."""
        final_dataset = []
        
        for i, sample in enumerate(all_data):
            final_sample = {
                'id': f'combined_{i:04d}',
                'gesture': sample['gesture'],
                'landmarks': sample['landmarks'],
                'timestamp': sample['timestamp'],
                'source': sample['source'],
                'metadata': {
                    'original_id': sample.get('original_sample', sample.get('synthetic_id', sample.get('original_id', 'unknown'))),
                    'confidence': sample.get('confidence', 1.0),
                    'augmented': sample.get('augmented', False),
                    'synthetic': sample.get('source') == 'synthetic'
                }
            }
            final_dataset.append(final_sample)
        
        return final_dataset
    
    def run_demo(self):
        """Run the complete data integration demo."""
        print("🚀 DATA INTEGRATION DEMO")
        print("This demo shows how data from 3 sources is processed and labeled")
        
        # Source 1: Raw data processing
        source1_data = self.demo_source_1_raw_data_processing()
        
        # Source 2: Synthetic data generation
        source2_data = self.demo_source_2_synthetic_data_generation()
        
        # Source 3: External dataset integration
        source3_data = self.demo_source_3_external_dataset_integration()
        
        # Combine and label all data
        final_dataset = self.demo_data_combination_and_labeling(source1_data, source2_data, source3_data)
        
        print("\n" + "=" * 60)
        print("SUMMARY: HOW THE 3 SOURCES ARE PROCESSED")
        print("=" * 60)
        print("1. RAW DATA (Source 1):")
        print("   - Load existing JSON files")
        print("   - Apply augmentation (rotation, scaling, noise)")
        print("   - Label: Keep original gesture labels")
        print("   - Add metadata: augmented=True, source='raw'")
        
        print("\n2. SYNTHETIC DATA (Source 2):")
        print("   - Generate from gesture templates")
        print("   - Apply hand anatomy rules")
        print("   - Label: Based on template definition")
        print("   - Add metadata: synthetic=True, source='synthetic'")
        
        print("\n3. EXTERNAL DATA (Source 3):")
        print("   - Load from various formats (CSV, JSON, XML)")
        print("   - Convert to standard landmark format")
        print("   - Label: Map external labels to our classes")
        print("   - Add metadata: source='external', confidence score")
        
        print("\n🎯 KEY LABELING PRINCIPLES:")
        print("   - All samples get consistent gesture labels")
        print("   - Source tracking for data lineage")
        print("   - Quality indicators (confidence, augmented flag)")
        print("   - Standardized landmark format (21 points)")
        
        return final_dataset

def main():
    """Main function."""
    demo = DataIntegrationDemo()
    final_dataset = demo.run_demo()
    
    print(f"\n✅ Demo completed! Final dataset has {len(final_dataset)} samples")

if __name__ == "__main__":
    main() 