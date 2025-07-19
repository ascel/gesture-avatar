"""
Basic test for gesture-controlled avatar project.
Tests core functionality without requiring all dependencies.
"""

import sys
import os
from pathlib import Path
import json


def test_project_structure():
    """Test that the project structure is correct."""
    print("Testing project structure...")
    
    required_dirs = [
        "src",
        "src/gesture_detection", 
        "src/avatar_animation",
        "src/streaming",
        "src/utils",
        "data",
        "data/raw",
        "data/processed", 
        "data/models",
        "notebooks",
        "scripts",
        "tests",
        "docs"
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        "src/main.py",
        "src/gesture_detection/__init__.py",
        "src/avatar_animation/__init__.py", 
        "src/streaming/__init__.py",
        "src/utils/__init__.py"
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            return False
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"✗ File missing: {file_path}")
            return False
    
    return True


def test_config_creation():
    """Test configuration file creation."""
    print("\nTesting configuration creation...")
    
    try:
        # Create basic config
        config = {
            "gesture_detection": {
                "model_type": "feature",
                "confidence_threshold": 0.7,
                "max_hands": 2,
                "target_fps": 15
            },
            "avatar_animation": {
                "avatar_type": "2d",
                "animation_speed": 1.0,
                "smooth_transitions": True
            },
            "streaming": {
                "output_resolution": [720, 480],
                "fps": 15,
                "delay_tolerance": 20
            },
            "hardware": {
                "gpu_acceleration": True,
                "max_memory_usage": 0.8
            }
        }
        
        # Create config directory
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(config_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✓ Configuration file created")
        return True
        
    except Exception as e:
        print(f"✗ Error creating config: {e}")
        return False


def test_sample_data_creation():
    """Test sample data creation."""
    print("\nTesting sample data creation...")
    
    try:
        # Create sample gesture data
        data_dir = Path("data/raw")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        gestures = ["wave", "thumbs_up", "point", "clap", "fist"]
        
        for gesture in gestures:
            gesture_dir = data_dir / gesture
            gesture_dir.mkdir(exist_ok=True)
            
            # Create sample landmark data
            sample_landmarks = []
            for i in range(21):  # MediaPipe hands have 21 landmarks
                sample_landmarks.append({
                    'x': 0.5 + 0.1 * (i % 3),
                    'y': 0.5 + 0.1 * (i // 3),
                    'z': 0.0 + 0.01 * i
                })
            
            # Create sample JSON files
            for i in range(5):  # Create 5 sample files per gesture
                sample_data = {
                    'gesture': gesture,
                    'landmarks': sample_landmarks,
                    'timestamp': f'2024-01-01T12:00:00.{i:03d}Z'
                }
                
                json_path = gesture_dir / f"sample_{i:04d}.json"
                with open(json_path, 'w') as f:
                    json.dump(sample_data, f, indent=2)
        
        print("✓ Sample data created")
        return True
        
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False


def test_requirements_file():
    """Test requirements.txt file."""
    print("\nTesting requirements file...")
    
    try:
        with open("requirements.txt", 'r') as f:
            requirements = f.read()
        
        # Check for essential packages
        essential_packages = [
            "tensorflow",
            "opencv-python", 
            "mediapipe",
            "numpy",
            "streamlit"
        ]
        
        for package in essential_packages:
            if package in requirements:
                print(f"✓ Package found: {package}")
            else:
                print(f"✗ Package missing: {package}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading requirements: {e}")
        return False


def test_readme_file():
    """Test README.md file."""
    print("\nTesting README file...")
    
    try:
        with open("README.md", 'r', encoding='utf-8') as f:
            readme = f.read()
        
        # Check for essential sections
        essential_sections = [
            "Real-Time Gesture-Controlled Video Avatar",
            "Features",
            "Installation",
            "Usage"
        ]
        
        for section in essential_sections:
            if section in readme:
                print(f"✓ Section found: {section}")
            else:
                print(f"✗ Section missing: {section}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error reading README: {e}")
        return False


def main():
    """Run all basic tests."""
    print("Gesture Avatar - Basic System Tests")
    print("===================================")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration", test_config_creation),
        ("Sample Data", test_sample_data_creation),
        ("Requirements", test_requirements_file),
        ("README", test_readme_file),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("BASIC TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All basic tests passed! Project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Collect gesture data: python src/main.py --collect-data")
        print("3. Preprocess data: python src/main.py --preprocess")
        print("4. Train models: python src/main.py --train")
        print("5. Run demo: python src/main.py")
    else:
        print("⚠ Some basic tests failed. Please check the project structure.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 