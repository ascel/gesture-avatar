#!/usr/bin/env python3
"""
Simple retraining script to run from the project root directory.

Usage:
    python run_retraining.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the retraining process."""
    print("🚀 Starting Model Retraining")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   Current directory should contain 'src' folder")
        sys.exit(1)
    
    # Step 1: Expand dataset
    print("\n📊 Step 1: Expanding dataset...")
    try:
        result = subprocess.run([sys.executable, "scripts/expand_dataset.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Dataset expansion completed")
        else:
            print(f"❌ Dataset expansion failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running dataset expansion: {e}")
        return False
    
    # Step 2: Retrain model
    print("\n🧠 Step 2: Retraining model...")
    try:
        result = subprocess.run([sys.executable, "scripts/retrain_model.py", "--data-source", "expanded"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Model retraining completed")
        else:
            print(f"❌ Model retraining failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running model retraining: {e}")
        return False
    
    # Step 3: Test model
    print("\n🧪 Step 3: Testing model...")
    try:
        result = subprocess.run([sys.executable, "scripts/test_retrained_model.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Model testing completed")
        else:
            print(f"❌ Model testing failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running model testing: {e}")
        return False
    
    print("\n🎉 All steps completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Test with real-time recognition: python src/main.py")
    print("   2. Deploy to your avatar system")
    print("   3. Monitor performance and collect feedback")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 