# YOLO Object Detection Integration

## Overview

The working simplified autonomous driving system now includes YOLO object detection capabilities. The system can detect and track various objects in real-time while maintaining the successful lane-following behavior.

## Features

- **Real-time Object Detection**: Uses YOLOv8n for fast and accurate object detection
- **Traffic Light Color Recognition**: Analyzes traffic light regions to detect red/green/yellow states
- **Visual Feedback**: Bounding boxes and labels displayed on the camera feed with color-coded traffic lights
- **LLM Integration**: Detected objects and traffic light states included in driving explanations
- **Traffic-aware**: Focuses on traffic-relevant objects (cars, trucks, persons, traffic lights, etc.)
- **Flexible Configuration**: Can disable YOLO detection if needed

## Usage

### Basic Usage
```bash
python working_simple.py
```

### Command Line Options
```bash
# Run without YOLO detection
python working_simple.py --no-yolo

# Run without LLM explanations
python working_simple.py --no-llm  

# Run with both disabled
python working_simple.py --no-yolo --no-llm

# Set custom FPS
python working_simple.py --fps 30
```

## Key Components

### Object Detection
- Model: YOLOv8n (yolov8n.pt)
- Confidence threshold: 0.5
- Detects all 80 COCO classes
- Focuses on traffic-relevant objects for explanations

### Traffic Light Recognition
- **Color Detection**: Analyzes HSV color space to identify red, green, yellow lights
- **State Confidence**: Provides confidence scores for each detected state
- **Brightness Analysis**: Considers light brightness to distinguish active vs inactive lights
- **Visual Feedback**: Traffic light bounding boxes are color-coded (red, green, yellow, gray)

### Visual Display
- Window title: "Autonomous Driving with YOLO"
- Color-coded bounding boxes:
  - **Green**: Regular objects (cars, persons, etc.)
  - **Red**: Red traffic lights
  - **Green**: Green traffic lights  
  - **Yellow**: Yellow traffic lights
  - **Gray**: Unknown traffic light state
- Enhanced labels showing traffic light states
- Driving information overlay (speed, steering, object count)

### LLM Explanations
Enhanced explanations now include detected objects and traffic light states:
```
"Driving straight at 15.2 km/h. Traffic lights: red light. Detected: car (0.89)"
"Turning right at 12.4 km/h (curvature: 0.0234). Traffic lights: green light. Detected: truck (0.92), person (0.76)"
"Driving straight at 20.1 km/h. Detected: car (0.85), bus (0.78)"
```

## Technical Details

### Performance
- YOLO detection runs in parallel with lane detection
- Traffic light color analysis adds minimal overhead
- Minimal impact on control system performance
- Graceful fallback if YOLO or traffic light detection fails

### Object Classes
The system detects 80 COCO classes but focuses on traffic-relevant ones:
- Vehicles: car, truck, bus, motorcycle, bicycle
- People: person
- Traffic infrastructure: traffic light (with color state), stop sign

### Traffic Light Color Detection Algorithm
1. **Region Extraction**: Extract traffic light region from YOLO bounding box
2. **HSV Conversion**: Convert RGB to HSV for better color separation
3. **Color Range Detection**: Apply HSV masks for red, green, and yellow ranges
4. **Brightness Analysis**: Consider pixel brightness to identify active lights
5. **Score Calculation**: Combine area ratio and brightness for final confidence
6. **State Determination**: Select color with highest confidence score

### Configuration
The YOLO detector is configured in the main function:
```python
yolo_config = {
    'model': 'yolov8n.pt',
    'device': 'cuda',  # Falls back to CPU if CUDA unavailable
    'confidence': 0.5,
    'classes': None  # Detect all classes
}
```

## Troubleshooting

### YOLO Initialization Issues
If YOLO fails to initialize:
- Check that yolov8n.pt exists in the project root
- Ensure ultralytics package is installed
- Try running with --no-yolo flag

### Performance Issues
If detection is too slow:
- Lower the FPS: --fps 15
- Use CPU instead of GPU (edit device in code)
- Disable YOLO: --no-yolo

### Dependencies
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## Integration Benefits

1. **Enhanced Situational Awareness**: The system now knows what objects are around
2. **Better Explanations**: LLM provides more contextual driving explanations
3. **Visual Feedback**: Easy to see what the system detects
4. **Modular Design**: YOLO can be easily disabled without affecting core functionality
5. **Future Extensions**: Foundation for more advanced behaviors (object avoidance, traffic light response, etc.)

The system maintains the same reliable lane-following behavior while adding intelligent object awareness capabilities. 