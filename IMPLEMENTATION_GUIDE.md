# Implementation Guide

This guide explains how to implement and run the LLM-based Explainable Autonomous Driving system.

## Prerequisites

Before starting, make sure you have:

1. **CARLA Simulator**: Download and install [CARLA 0.9.13+](https://carla.readthedocs.io/en/latest/start_quickstart/)
2. **Python 3.8+**: The project requires Python 3.8 or newer
3. **Hardware Requirements**: GPU recommended but CPU mode works
4. **Git**: To clone the repository

## Installation Steps

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/auto_drive_llm_explainable.git
cd auto_drive_llm_explainable
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

This installs all required packages including:
- PyTorch for YOLOv8 object detection
- OpenVINO runtime for lane detection
- OpenCV for image processing
- CARLA Python API
- Pygame for visualization

3. **Download Required Models**:

The system will automatically download the YOLOv8 model (`yolov8n.pt`) when first run. For manual download:
```bash
# YOLOv8 nano model (lightweight, fast inference)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

For OpenVINO lane detection model:
```bash
# Create model directory if needed
mkdir -p converted_model
# Place your lane detection model as: converted_model/lane_model.xml
# Contact the project team for the specific model file
```

4. **Set Environment Variables** (for LLM API):
```bash
# For Deepseek API (default)
export DEEPSEEK_API_KEY=your_api_key

# Or modify the API key directly in carla_sim_llm.py
```

5. **Configure CARLA Path**:
```bash
# Add CARLA to your system PATH
export CARLA_ROOT=/path/to/your/carla/installation
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.8-linux-x86_64.egg
```

## Running the System

### Step 1: Start CARLA Server

**Option 1: Using the provided script**:
```bash
python run_carla.py
```

**Option 2: Manual CARLA startup**:
```bash
cd $CARLA_ROOT
./CarlaUE4.sh -quality-level=Low -fps=20
```

**Important CARLA Settings**:
- Use `-quality-level=Low` for better performance
- Set `-fps=20` to match the system expectations
- Ensure port 2000 is available (default CARLA port)

### Step 2: Run the Autonomous Driving System

**Basic Usage**:
```bash
python carla_sim_llm.py
```

**Common Options**:
```bash
# Set simulation speed and map
python carla_sim_llm.py --fps 20 --map 2 --weather 1

# Disable LLM explanations for faster performance
python carla_sim_llm.py --no-llm

# Disable object detection 
python carla_sim_llm.py --no-yolo

# Run with different weather conditions
python carla_sim_llm.py --weather 3  # Options: 0-10
```

**All Available Options**:
```bash
python carla_sim_llm.py --help

Options:
  --fps INT          Simulation frame rate (default: 10)
  --map STRING       CARLA map ID 1-10 (default: '1') 
  --weather INT      Weather preset 0-10 (default: 1)
  --show-map         Show map view (default: False)
  --model STRING     Lane detection model type (default: 'openvino')
  --no-llm           Disable LLM explanations
  --no-yolo          Disable YOLO object detection
```

## System Components

### 1. Perception System

**Object Detection (YOLOv8)**
- Automatically detects 80 object classes
- Real-time bounding box visualization
- Confidence thresholding for reliable detection
- Optimized for CARLA environment

**Lane Detection (OpenVINO)**
- Deep learning-based lane segmentation
- Polynomial curve fitting for smooth trajectories
- Real-time processing at 20+ FPS
- Robust to lighting and weather conditions

**Traffic Light Detection**
- Integrated with object detection pipeline
- State recognition (red, yellow, green)
- Position-aware filtering for relevant lights
- Automatic braking for red/yellow lights

### 2. Control System

**Pure Pursuit Controller**
- Geometric path following algorithm
- Lookahead distance adaptation
- Smooth steering commands
- Curvature-based speed adjustment

**PID Speed Controller**
- Maintains desired driving speed
- Smooth acceleration/deceleration
- Emergency braking capability
- Traffic-aware speed management

### 3. LLM Explainer

**Threaded Processing**
- Non-blocking explanation generation
- 3-second explanation intervals
- Real-time status indicators
- Automatic error recovery

**Deepseek API Integration**
- Fast response times (1-3 seconds)
- Cost-effective API pricing
- Structured prompt engineering
- Rich context formatting

## Customization Options

### 1. Adjusting Detection Sensitivity

Edit detection thresholds in `carla_sim_llm.py`:
```python
# Object detection confidence threshold
confidence_threshold = 0.5  # Default

# Obstacle stopping distances
min_area_for_stop = 1500  # Bounding box area threshold
pedestrian_area_threshold = 800  # More sensitive for pedestrians
```

### 2. Modifying Control Parameters

Adjust controller settings:
```python
# Pure Pursuit parameters
controller = PurePursuitPlusPID(
    pure_pursuit=PurePursuit(K_dd=0.4, wheel_base=2.65),
    pid=PIDController(Kp=0.3, Ki=0.1, Kd=0.05, set_point=5.56)
)
```

### 3. Changing LLM Configuration

Update LLM settings:
```python
llm_config = {
    'api_key': "your_api_key",
    'api_base_url': 'https://api.deepseek.com/v1',
    'api_model': 'deepseek-chat',
    'max_tokens': 150,  # Explanation length
    'temperature': 0.7   # Response creativity
}
```

## Troubleshooting

### 1. CARLA Connection Issues

**Problem**: "Connection refused" or timeout errors
**Solutions**:
- Verify CARLA is running before starting the script
- Check if port 2000 is available: `netstat -an | grep 2000`
- Try restarting CARLA server
- Increase timeout in the script if needed

### 2. Model Loading Errors

**Problem**: YOLOv8 model not found
**Solutions**:
- Ensure `yolov8n.pt` is in the project root directory
- Check internet connection for automatic download
- Manually download the model file

**Problem**: OpenVINO lane model missing
**Solutions**:
- Verify `converted_model/lane_model.xml` exists
- Check file permissions
- Contact project team for model file

### 3. Performance Issues

**Problem**: Low frame rate or stuttering
**Solutions**:
- Lower CARLA quality: `--quality-level=Low`
- Reduce simulation FPS: `--fps 10`
- Disable LLM: `--no-llm`
- Use CPU-only mode if GPU memory is limited

**Problem**: LLM explanations slow or failing
**Solutions**:
- Check internet connection
- Verify API key is correct
- Monitor API rate limits
- Use `--no-llm` for testing

### 4. Camera and Detection Issues

**Problem**: Poor lane detection
**Solutions**:
- Check camera pitch angle settings
- Verify OpenVINO model compatibility
- Adjust lighting conditions in CARLA

**Problem**: Missing object detections
**Solutions**:
- Lower confidence threshold
- Check YOLOv8 model version compatibility
- Verify camera image quality

## Performance Optimization

### 1. Hardware Optimization

**For CPU-only systems**:
- Use `--fps 10` for lower computational load
- Disable resource-heavy features: `--no-yolo --no-llm`
- Close unnecessary applications

**For GPU systems**:
- Monitor GPU memory usage
- Use CUDA for PyTorch operations
- Consider larger batch sizes for detection

### 2. Software Optimization

**Memory Usage**:
- The system typically uses 2-4GB RAM
- GPU memory usage: 1-2GB for YOLO + OpenVINO
- Monitor with `nvidia-smi` or system tools

**Network Usage**:
- LLM API calls: ~1KB per request every 3 seconds
- Total network usage: minimal (<1MB/hour)

## Development and Extension

### 1. Adding New Features

**New Object Classes**:
- Modify YOLO class filtering
- Update detection visualization
- Add new blocking object logic

**Enhanced Control**:
- Implement advanced controllers (MPC, etc.)
- Add lane change capabilities
- Improve emergency maneuvers

### 2. Integration with Other Systems

**ROS Integration**:
- Wrap components as ROS nodes
- Use ROS messages for communication
- Enable distributed processing

**Real Vehicle Integration**:
- Replace CARLA sensors with real cameras
- Adapt coordinate systems
- Add safety monitoring

## System Evaluation

### 1. Performance Metrics

Monitor these key indicators:
- Frame rate: Should maintain 20+ FPS
- Detection accuracy: >90% for relevant objects
- LLM response rate: >95% successful
- Control smoothness: Low jerk and acceleration

### 2. Safety Testing

Test in various scenarios:
- Different weather conditions
- Heavy traffic situations
- Pedestrian crossings
- Emergency braking scenarios
- Traffic light compliance

Verify explanations match actual system behavior and are understandable to users. 