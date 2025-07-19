"""
Setup script for gesture-controlled avatar project.
Installs dependencies and initializes the project structure.
"""

import subprocess
import sys
import os
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False


def create_sample_data():
    """Create sample data structure."""
    print("Creating sample data structure...")
    
    try:
        # Create sample gesture data
        from scripts.download_assets import create_sample_data
        create_sample_data()
        print("✓ Sample data created")
        return True
    except Exception as e:
        print(f"✗ Error creating sample data: {e}")
        return False


def create_config_files():
    """Create configuration files."""
    print("Creating configuration files...")
    
    try:
        from scripts.download_assets import create_config_files
        create_config_files()
        print("✓ Configuration files created")
        return True
    except Exception as e:
        print(f"✗ Error creating config files: {e}")
        return False


def test_system():
    """Test the system components."""
    print("Testing system components...")
    
    try:
        subprocess.check_call([sys.executable, "src/test_system.py"])
        print("✓ System test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ System test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("Gesture Avatar - Project Setup")
    print("==============================")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found. Please run this from the project root.")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create sample data
    if not create_sample_data():
        return False
    
    # Create config files
    if not create_config_files():
        return False
    
    # Test system
    if not test_system():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Collect gesture data: python src/main.py --collect-data")
    print("2. Preprocess data: python src/main.py --preprocess")
    print("3. Train models: python src/main.py --train")
    print("4. Run demo: python src/main.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 