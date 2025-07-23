#!/usr/bin/env python3
"""
GPU Configuration Check Script

This script checks if GPU is available and configures TensorFlow for optimal performance.

Usage:
    python check_gpu.py
"""

import sys
import os

def check_tensorflow_gpu():
    """Check TensorFlow GPU availability and configuration."""
    print("🔍 Checking TensorFlow GPU Configuration")
    print("=" * 50)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow can see GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   • GPU {i}: {gpu.name}")
            
            # Test GPU configuration
            print("\n⚙️  Configuring GPU for optimal performance...")
            
            # Enable memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✅ Memory growth enabled for {gpu.name}")
            
            # Test mixed precision
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("   ✅ Mixed precision enabled (float16)")
            except Exception as e:
                print(f"   ⚠️  Mixed precision failed: {e}")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"   ✅ GPU computation successful: {c.numpy()}")
            
            print("\n🎉 GPU configuration successful!")
            return True
            
        else:
            print("⚠️  No GPU devices found")
            print("   TensorFlow will use CPU")
            
            # Test CPU computation
            print("\n🧪 Testing CPU computation...")
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"   ✅ CPU computation successful: {c.numpy()}")
            
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        print("   Install with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_system_gpu():
    """Check system GPU information."""
    print("\n🖥️  System GPU Information")
    print("=" * 30)
    
    # Check NVIDIA GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            print(result.stdout)
        else:
            print("⚠️  NVIDIA GPU not found or nvidia-smi not available")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found (NVIDIA drivers may not be installed)")
    except Exception as e:
        print(f"⚠️  Error checking NVIDIA GPU: {e}")
    
    # Check other GPU types
    try:
        import platform
        system = platform.system()
        print(f"\n💻 System: {system}")
        
        if system == "Windows":
            try:
                import wmi
                c = wmi.WMI()
                for gpu in c.Win32_VideoController():
                    print(f"   • {gpu.Name} ({gpu.AdapterRAM} bytes)")
            except ImportError:
                print("   ⚠️  WMI not available for detailed GPU info")
        elif system == "Linux":
            try:
                result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or '3D' in line]
                    for line in gpu_lines:
                        print(f"   • {line.strip()}")
            except Exception:
                print("   ⚠️  Could not get GPU info from lspci")
                
    except Exception as e:
        print(f"⚠️  Error checking system GPU: {e}")

def main():
    """Main function."""
    print("🚀 GPU Configuration Checker")
    print("=" * 40)
    
    # Check system GPU
    check_system_gpu()
    
    # Check TensorFlow GPU
    gpu_available = check_tensorflow_gpu()
    
    print("\n📋 Summary:")
    print("=" * 20)
    if gpu_available:
        print("✅ GPU acceleration is available and configured")
        print("🚀 Your models will use GPU for faster training")
        print("💡 Benefits:")
        print("   • 10-50x faster training")
        print("   • Better memory management")
        print("   • Mixed precision optimization")
    else:
        print("⚠️  GPU acceleration not available")
        print("💻 Models will use CPU (slower but functional)")
        print("💡 To enable GPU:")
        print("   • Install NVIDIA drivers")
        print("   • Install CUDA toolkit")
        print("   • Install cuDNN")
        print("   • Install tensorflow-gpu")
    
    print("\n🎯 Next steps:")
    if gpu_available:
        print("1. Run training: python scripts/retrain_model.py")
        print("2. Test performance: python test_optimized_model.py")
    else:
        print("1. Install GPU dependencies (optional)")
        print("2. Run training with CPU: python scripts/retrain_model.py")

if __name__ == "__main__":
    main() 