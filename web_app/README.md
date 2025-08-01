# Gesture Avatar Web Application

A comprehensive web-based platform for real-time gesture recognition and avatar control. This application provides a complete pipeline from data collection to model training and live demonstration.

## 🚀 Features

- **Data Collection**: Capture gesture samples directly from your webcam
- **Data Preprocessing**: Configure and apply preprocessing to improve model training
- **Model Training**: Train ResNet and EfficientNet models with real-time progress visualization
- **Model Management**: Switch between trained models and manage your model library
- **Live Demo**: Real-time gesture detection with 2D/3D avatar control
- **Analytics (EDA)**: Comprehensive dataset analysis and insights

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Webcam for data collection and live demo
- GPU recommended for model training (RTX 3060 or equivalent)

## 🛠️ Installation

### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd web_app/backend
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install project dependencies**
   ```bash
   cd ../../
   pip install -r requirements.txt
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd web_app/frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

## 🚀 Running the Application

### Start Backend Server

1. **From the backend directory:**
   ```bash
   cd web_app/backend
   python main.py
   ```
   
   The backend will start on `http://localhost:8000`

### Start Frontend Development Server

1. **From the frontend directory:**
   ```bash
   cd web_app/frontend
   npm start
   ```
   
   The frontend will start on `http://localhost:3000`

## 📱 Using the Application

### 1. Dashboard
- Overview of system status and quick actions
- View dataset statistics and model information
- Quick navigation to all features

### 2. Data Collection
- **Start Collection**: Choose a gesture name (e.g., "fist", "peace")
- **Record Samples**: Use "Start Recording (10s)" for automatic capture or "Capture Single Frame" for manual control
- **Manage Sessions**: End sessions and view collected data

### 3. Data Preprocessing
- **Configure Options**: Enable/disable normalization and augmentation
- **Set Parameters**: Adjust noise level and rotation range for augmentation
- **Run Processing**: Apply preprocessing to raw data

### 4. Model Training
- **Select Backbone**: Choose between ResNet and EfficientNet
- **Configure Parameters**: Set epochs, batch size, learning rate, and validation split
- **Monitor Progress**: Real-time training and validation accuracy visualization
- **View Metrics**: Live charts showing loss and accuracy curves

### 5. Model Management
- **View Models**: See all trained models with metadata
- **Activate Models**: Switch between models for inference
- **Model Info**: Check accuracy, size, and training date

### 6. Live Demo
- **Select Model**: Choose an active model for inference
- **Choose Avatar**: Switch between 2D and 3D avatars
- **Real-time Detection**: See gesture detection and avatar response
- **Performance Metrics**: Monitor FPS and confidence scores

### 7. Analytics (EDA)
- **Dataset Overview**: View sample distribution and balance
- **Quality Insights**: Get recommendations for dataset improvement
- **Export Reports**: Download analysis results

## 🎯 Supported Gestures

The system supports detection of these gestures:
- **Fist**: Closed hand
- **Open Hand**: Open palm
- **Peace**: Victory/peace sign
- **Point**: Index finger pointing
- **Thumbs Up**: Thumb up gesture

## 🔧 Configuration

### Backend Configuration
- API endpoints are configured in `backend/main.py`
- Model paths and data directories can be adjusted
- GPU acceleration is automatically detected

### Frontend Configuration
- API proxy is configured in `package.json`
- UI theme and styling in `App.tsx`
- Chart configurations in individual page components

## 📊 API Endpoints

### Data Collection
- `POST /api/data/start-collection` - Start data collection session
- `POST /api/data/save-sample` - Save gesture sample
- `GET /api/data/gestures` - Get available gestures

### Preprocessing
- `POST /api/preprocessing/configure` - Configure preprocessing
- `POST /api/preprocessing/run` - Run preprocessing

### Training
- `POST /api/training/start` - Start model training
- `GET /api/training/status/{session_id}` - Get training status
- `WebSocket /ws/training/{session_id}` - Real-time training updates

### Models
- `GET /api/models/list` - List available models
- `POST /api/models/activate` - Activate a model

### Inference
- `POST /api/inference/detect` - Detect gesture from image

### Analytics
- `GET /api/eda/dataset-info` - Get dataset statistics

## 🐛 Troubleshooting

### Common Issues

1. **Webcam not working**
   - Check browser permissions for camera access
   - Ensure no other applications are using the camera

2. **Backend connection failed**
   - Verify backend server is running on port 8000
   - Check firewall settings

3. **Model training fails**
   - Ensure sufficient data is collected and preprocessed
   - Check GPU memory availability
   - Verify Python dependencies are installed

4. **Poor gesture detection**
   - Collect more training samples
   - Ensure good lighting conditions
   - Try different model backbones

### Performance Tips

- **For better training performance**: Use GPU acceleration
- **For smoother live demo**: Close unnecessary browser tabs
- **For better accuracy**: Collect diverse samples under different conditions

## 🤝 Contributing

This is a portfolio project demonstrating ML and web development skills. Feel free to fork and experiment!

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- Material-UI for React components
- FastAPI for backend framework
- Recharts for data visualization 