# Real-Time Gesture-Controlled Video Avatar

A machine learning-driven application that enables users to control a virtual avatar (2D or 3D) using webcam-based hand and body gestures, streamed in real-time via OBS for applications like gaming, VTubing, or virtual events.

## 🎯 Project Overview

This project demonstrates advanced ML, computer vision, and real-time system integration skills through:
- Custom ML pipeline for real-time gesture recognition
- Natural, context-aware avatar animations
- OBS integration for livestreaming
- Optimized performance for consumer-grade hardware (RTX 3060)

## 🚀 Features

- **Real-time Gesture Recognition**: Detect 5+ hand and body gestures using MediaPipe + custom CNN
- **Avatar Animation**: Map gestures to smooth 2D/3D avatar movements
- **OBS Integration**: Stream avatar output with pyvirtualcam
- **Performance Optimized**: Runs smoothly on RTX 3060 with ≥15 FPS
- **Exploratory Data Analysis**: Comprehensive gesture data analysis and visualization

## 📋 Requirements

### Hardware
- RTX 3060 GPU (or equivalent)
- 16GB+ RAM
- Standard webcam (720p or higher)

### Software
- Python 3.8+
- OBS Studio (for streaming)
- CUDA-compatible drivers

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/gesture-avatar.git
   cd gesture-avatar
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models and datasets**
   ```bash
   python scripts/download_assets.py
   ```

## 🎮 Usage

### Quick Start
```bash
# Run the demo interface
python src/main.py

# Or run individual components
python src/gesture_detection.py  # Test gesture recognition
python src/avatar_animation.py   # Test avatar animations
python src/streaming.py          # Test OBS integration
```

### Demo Interface
The Streamlit demo provides:
- Real-time webcam feed
- Gesture detection visualization
- Avatar animation preview
- Performance metrics
- 2D/3D avatar switching

### OBS Integration
1. Install OBS Studio
2. Add "Virtual Camera" as a video source
3. Run the streaming script
4. Your avatar will appear in OBS

## 📊 Performance Metrics

- **Gesture Detection Accuracy**: ≥90% F1 score
- **Pipeline Latency**: ≤20 seconds
- **Frame Rate**: ≥15 FPS
- **Supported Gestures**: Wave, Thumbs-up, Point, Clap, Fist

## 📁 Project Structure

```
gesture-avatar/
├── src/                    # Main source code
│   ├── gesture_detection/  # Gesture recognition models
│   ├── avatar_animation/   # Avatar animation system
│   ├── streaming/          # OBS integration
│   └── utils/              # Utility functions
├── data/                   # Datasets and models
│   ├── raw/               # Raw gesture data
│   ├── processed/         # Processed datasets
│   └── models/            # Trained models
├── notebooks/             # Jupyter notebooks for EDA
├── scripts/               # Utility scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🔬 Technical Details

### ML Pipeline
- **Gesture Detection**: MediaPipe + MobileNet CNN
- **Animation Mapping**: Decision Tree + RNN for smooth transitions
- **Data Processing**: Custom dataset with Jester dataset augmentation

### Architecture
- Real-time webcam input processing
- GPU-accelerated inference (TensorFlow/PyTorch)
- Modular design for easy extension

## 📈 Development Timeline

- **Week 1-2**: Data collection, EDA, preprocessing
- **Week 3-4**: Gesture recognition model training
- **Week 5-6**: Avatar animation pipeline
- **Week 7**: OBS streaming integration
- **Week 8**: Demo UI and documentation

## 🤝 Contributing

This is a portfolio project demonstrating ML and computer vision skills. Feel free to fork and experiment!

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- Jester dataset for gesture recognition
- OBS Studio for streaming integration