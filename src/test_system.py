"""
Test script for gesture-controlled avatar system.
Verifies all components work correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from gesture_detection.models import GestureDetector, FeatureBasedGestureModel
        print("✓ Gesture detection models imported")
    except Exception as e:
        print(f"✗ Error importing gesture detection: {e}")
        return False
    
    try:
        from avatar_animation.avatar_system import AvatarManager, SpriteBasedAvatar
        print("✓ Avatar animation imported")
    except Exception as e:
        print(f"✗ Error importing avatar animation: {e}")
        return False
    
    try:
        from streaming.obs_integration import StreamManager, OBSStreamer
        print("✓ Streaming imported")
    except Exception as e:
        print(f"✗ Error importing streaming: {e}")
        return False
    
    try:
        from utils.data_preprocessing import GestureDataPreprocessor
        print("✓ Data preprocessing imported")
    except Exception as e:
        print(f"✗ Error importing data preprocessing: {e}")
        return False
    
    return True


def test_gesture_detection():
    """Test gesture detection component."""
    print("\nTesting gesture detection...")
    
    try:
        # Test with default configuration
        detector = GestureDetector(model_type="feature")
        print("✓ Gesture detector created")
        
        # Test with sample data
        import numpy as np
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        gesture, confidence, info = detector.detect_gesture(sample_frame)
        print(f"✓ Gesture detection works: {gesture} (confidence: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing gesture detection: {e}")
        return False


def test_avatar_animation():
    """Test avatar animation component."""
    print("\nTesting avatar animation...")
    
    try:
        # Test 2D avatar
        avatar_2d = AvatarManager(avatar_type="2d")
        frame_2d = avatar_2d.update("wave", 0.8)
        print("✓ 2D avatar created and updated")
        
        # Test 3D avatar
        avatar_3d = AvatarManager(avatar_type="3d")
        frame_3d = avatar_3d.update("thumbs_up", 0.9)
        print("✓ 3D avatar created and updated")
        
        # Test avatar switching
        avatar_2d.switch_avatar_type("3d")
        print("✓ Avatar switching works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing avatar animation: {e}")
        return False


def test_streaming():
    """Test streaming component."""
    print("\nTesting streaming...")
    
    try:
        # Test stream manager creation
        config = {
            'output_resolution': [720, 480],
            'fps': 15,
            'delay_tolerance': 20
        }
        
        stream_manager = StreamManager(config)
        print("✓ Stream manager created")
        
        # Test performance monitor
        perf_stats = stream_manager.get_performance_report()
        print("✓ Performance monitoring works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing streaming: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing component."""
    print("\nTesting data preprocessing...")
    
    try:
        # Test preprocessor creation
        preprocessor = GestureDataPreprocessor()
        print("✓ Data preprocessor created")
        
        # Test feature extraction
        import numpy as np
        sample_landmarks = []
        for i in range(21):  # MediaPipe hands have 21 landmarks
            sample_landmarks.append({
                'x': 0.5 + 0.1 * (i % 3),
                'y': 0.5 + 0.1 * (i // 3),
                'z': 0.0 + 0.01 * i
            })
        
        features = preprocessor.extract_hand_features(sample_landmarks)
        if features:
            print(f"✓ Feature extraction works: {len(features)} features extracted")
        else:
            print("✗ Feature extraction failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing data preprocessing: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from main import GestureAvatarDemo
        
        # Test with default config
        demo = GestureAvatarDemo()
        print("✓ Demo created with default configuration")
        
        # Test config structure
        required_keys = ['gesture_detection', 'avatar_animation', 'streaming', 'hardware']
        for key in required_keys:
            if key in demo.config:
                print(f"✓ Config has {key} section")
            else:
                print(f"✗ Config missing {key} section")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing configuration: {e}")
        return False


def main():
    """Run all tests."""
    print("Gesture Avatar System - Component Tests")
    print("=======================================")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Gesture Detection", test_gesture_detection),
        ("Avatar Animation", test_avatar_animation),
        ("Streaming", test_streaming),
        ("Data Preprocessing", test_data_preprocessing),
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
    print("TEST SUMMARY")
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
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Collect gesture data: python src/main.py --collect-data")
        print("2. Preprocess data: python src/main.py --preprocess")
        print("3. Train models: python src/main.py --train")
        print("4. Run demo: python src/main.py")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 