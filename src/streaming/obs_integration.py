"""
OBS integration for gesture-controlled avatar project.
Streams animated avatar output using pyvirtualcam for OBS compatibility.
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional, Callable
from pathlib import Path
import json
import queue


class OBSStreamer:
    """OBS streaming integration using pyvirtualcam."""
    
    def __init__(self, config: dict = None):
        """
        Initialize OBS streamer.
        
        Args:
            config: Streaming configuration
        """
        self.config = config or {}
        self.virtual_camera = None
        self.is_streaming = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.streaming_thread = None
        
        # Default configuration
        self.width = self.config.get('output_resolution', [720, 480])[0]
        self.height = self.config.get('output_resolution', [720, 480])[1]
        self.fps = self.config.get('fps', 15)
        self.delay_tolerance = self.config.get('delay_tolerance', 20)  # seconds
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.actual_fps = 0
        
        # Initialize virtual camera
        self._initialize_virtual_camera()
    
    def _initialize_virtual_camera(self):
        """Initialize virtual camera for OBS."""
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat
            
            # Create virtual camera
            self.virtual_camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=PixelFormat.BGR,
                device='obs-virtualcam'  # Use OBS virtual camera
            )
            
            print(f"Virtual camera initialized: {self.width}x{self.height} @ {self.fps}fps")
            
        except ImportError:
            print("Warning: pyvirtualcam not available. Install with: pip install pyvirtualcam")
            self.virtual_camera = None
        except Exception as e:
            print(f"Error initializing virtual camera: {e}")
            self.virtual_camera = None
    
    def start_streaming(self, frame_callback: Optional[Callable] = None):
        """
        Start streaming to OBS.
        
        Args:
            frame_callback: Optional callback function to get frames
        """
        if self.virtual_camera is None:
            print("Virtual camera not available. Cannot start streaming.")
            return False
        
        if self.is_streaming:
            print("Streaming already active.")
            return True
        
        self.is_streaming = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(
            target=self._streaming_loop,
            args=(frame_callback,),
            daemon=True
        )
        self.streaming_thread.start()
        
        print("OBS streaming started!")
        print("Add 'Virtual Camera' as a video source in OBS Studio")
        return True
    
    def stop_streaming(self):
        """Stop streaming to OBS."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=2)
        
        if self.virtual_camera:
            self.virtual_camera.close()
        
        # Calculate final statistics
        if self.start_time:
            total_time = time.time() - self.start_time
            self.actual_fps = self.frame_count / total_time if total_time > 0 else 0
        
        print(f"OBS streaming stopped. Average FPS: {self.actual_fps:.1f}")
    
    def _streaming_loop(self, frame_callback: Optional[Callable]):
        """Main streaming loop."""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()
        
        while self.is_streaming:
            try:
                current_time = time.time()
                
                # Check if it's time for next frame
                if current_time - last_frame_time >= frame_interval:
                    # Get frame from callback or queue
                    frame = None
                    
                    if frame_callback:
                        frame = frame_callback()
                    else:
                        try:
                            frame = self.frame_queue.get_nowait()
                        except queue.Empty:
                            # Create blank frame if no frame available
                            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    
                    if frame is not None:
                        # Resize frame to match virtual camera resolution
                        frame = self._resize_frame(frame)
                        
                        # Send frame to virtual camera
                        if self.virtual_camera:
                            self.virtual_camera.send(frame)
                            self.virtual_camera.sleep_until_next_frame()
                        
                        self.frame_count += 1
                        last_frame_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in streaming loop: {e}")
                time.sleep(0.1)
    
    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to match virtual camera resolution."""
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        return frame
    
    def send_frame(self, frame: np.ndarray):
        """
        Send frame to streaming queue.
        
        Args:
            frame: Frame to stream (numpy array)
        """
        if not self.is_streaming:
            return
        
        try:
            # Put frame in queue, remove old frame if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
            
        except Exception as e:
            print(f"Error sending frame: {e}")
    
    def get_streaming_stats(self) -> dict:
        """Get streaming statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        return {
            'is_streaming': self.is_streaming,
            'frame_count': self.frame_count,
            'elapsed_time': elapsed_time,
            'actual_fps': self.actual_fps,
            'target_fps': self.fps,
            'resolution': f"{self.width}x{self.height}",
            'queue_size': self.frame_queue.qsize()
        }


class StreamManager:
    """Manages streaming with performance optimization."""
    
    def __init__(self, config: dict = None):
        """
        Initialize stream manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.obs_streamer = OBSStreamer(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Streaming state
        self.is_active = False
        self.frame_buffer = []
        self.buffer_size = 5  # Number of frames to buffer
        
        # Performance targets
        self.target_fps = self.config.get('fps', 15)
        self.max_delay = self.config.get('delay_tolerance', 20)
    
    def start_streaming(self, frame_callback: Optional[Callable] = None):
        """Start streaming with performance monitoring."""
        if self.obs_streamer.start_streaming(frame_callback):
            self.is_active = True
            self.performance_monitor.start()
            print("Stream manager started successfully")
            return True
        return False
    
    def stop_streaming(self):
        """Stop streaming and performance monitoring."""
        self.is_active = False
        self.obs_streamer.stop_streaming()
        self.performance_monitor.stop()
        print("Stream manager stopped")
    
    def send_frame(self, frame: np.ndarray):
        """Send frame with performance optimization."""
        if not self.is_active:
            return
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > self.buffer_size:
            self.frame_buffer.pop(0)
        
        # Send frame to OBS
        self.obs_streamer.send_frame(frame)
        
        # Update performance monitor
        self.performance_monitor.update_frame()
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report."""
        obs_stats = self.obs_streamer.get_streaming_stats()
        perf_stats = self.performance_monitor.get_stats()
        
        return {
            'streaming': obs_stats,
            'performance': perf_stats,
            'targets_met': {
                'fps_target': perf_stats['current_fps'] >= self.target_fps * 0.8,
                'delay_target': perf_stats['average_latency'] <= self.max_delay,
                'buffer_healthy': len(self.frame_buffer) <= self.buffer_size
            }
        }
    
    def optimize_performance(self):
        """Apply performance optimizations based on current metrics."""
        perf_stats = self.performance_monitor.get_stats()
        
        # Adjust buffer size based on performance
        if perf_stats['current_fps'] < self.target_fps * 0.8:
            # Reduce buffer size to improve responsiveness
            self.buffer_size = max(2, self.buffer_size - 1)
            print(f"Reduced buffer size to {self.buffer_size} for better performance")
        
        elif perf_stats['current_fps'] > self.target_fps * 1.2:
            # Increase buffer size for smoother streaming
            self.buffer_size = min(10, self.buffer_size + 1)
            print(f"Increased buffer size to {self.buffer_size} for smoother streaming")


class PerformanceMonitor:
    """Monitors streaming performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = None
        self.frame_times = []
        self.max_frame_history = 100
        
        # Performance metrics
        self.current_fps = 0
        self.average_latency = 0
        self.frame_drops = 0
        self.total_frames = 0
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.frame_times = []
        self.current_fps = 0
        self.average_latency = 0
        self.frame_drops = 0
        self.total_frames = 0
    
    def stop(self):
        """Stop performance monitoring."""
        self.start_time = None
    
    def update_frame(self):
        """Update frame statistics."""
        current_time = time.time()
        self.total_frames += 1
        
        # Track frame timing
        self.frame_times.append(current_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        # Calculate current FPS
        if len(self.frame_times) > 1:
            recent_frames = self.frame_times[-30:]  # Last 30 frames
            if len(recent_frames) > 1:
                time_span = recent_frames[-1] - recent_frames[0]
                self.current_fps = (len(recent_frames) - 1) / time_span if time_span > 0 else 0
        
        # Calculate average latency
        if len(self.frame_times) > 1:
            intervals = [self.frame_times[i] - self.frame_times[i-1] 
                        for i in range(1, len(self.frame_times))]
            self.average_latency = np.mean(intervals) if intervals else 0
    
    def get_stats(self) -> dict:
        """Get current performance statistics."""
        return {
            'current_fps': self.current_fps,
            'average_latency': self.average_latency,
            'frame_drops': self.frame_drops,
            'total_frames': self.total_frames,
            'uptime': time.time() - self.start_time if self.start_time else 0
        }


def test_obs_integration():
    """Test OBS integration with a simple animation."""
    print("Testing OBS Integration")
    print("======================")
    
    # Configuration
    config = {
        'output_resolution': [720, 480],
        'fps': 15,
        'delay_tolerance': 20
    }
    
    # Initialize stream manager
    stream_manager = StreamManager(config)
    
    # Create a simple test animation
    def test_animation():
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        
        # Create a moving circle
        t = time.time()
        x = int(360 + 200 * np.sin(t))
        y = int(240 + 100 * np.cos(t))
        
        cv2.circle(frame, (x, y), 50, (0, 255, 0), -1)
        cv2.putText(frame, "OBS Test", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    try:
        # Start streaming
        if stream_manager.start_streaming(test_animation):
            print("Streaming test animation for 10 seconds...")
            print("Check OBS Studio for the virtual camera feed")
            
            # Run for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10:
                time.sleep(0.1)
                
                # Print performance stats every second
                if int(time.time() - start_time) % 2 == 0:
                    stats = stream_manager.get_performance_report()
                    print(f"FPS: {stats['performance']['current_fps']:.1f}, "
                          f"Latency: {stats['performance']['average_latency']*1000:.1f}ms")
            
            # Stop streaming
            stream_manager.stop_streaming()
            print("Test completed!")
            
        else:
            print("Failed to start streaming")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        stream_manager.stop_streaming()
    except Exception as e:
        print(f"Test error: {e}")
        stream_manager.stop_streaming()


if __name__ == "__main__":
    test_obs_integration() 