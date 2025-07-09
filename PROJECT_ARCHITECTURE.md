# LLM-Based Explainable Autonomous Driving Project Architecture

This document shows the architecture of the LLM-based explainable autonomous driving system.

## Overview

The project creates an autonomous driving system that:
1. Runs in the CARLA simulator
2. Uses deep learning models for perception 
3. Has an integrated control system
4. Explains its actions using Large Language Models (LLMs)

## Current System Architecture

The system now uses an integrated architecture in `carla_sim_llm.py`. This design combines all components in a single file for better performance and simplicity.

```
CARLA Simulator <---> carla_sim_llm.py (Integrated System)
    |
    v
+------------------------+    +------------------+    +-----------------+
| Perception             |    | Control          |    | Explainability |
| - YOLOv8 Object Detect |---→| - Pure Pursuit   |    | - Threaded LLM  |
| - OpenVINO Lane Detect |    | - PID Controller |    | - Deepseek API  |
| - Traffic Light Detect |    | - Brake Control  |    | - Real-time     |
+------------------------+    +------------------+    +-----------------+
         |                              ↑
         ↓                              |
+------------------------+              |
| Planning & Decision    |              |
| - Obstacle Avoidance   |--------------+
| - Traffic Light Rules  |
| - Lane Following       |
| - Emergency Braking    |
+------------------------+
```

## Component Details

### 1. Perception System

**Object Detection (YOLOv8)**
- Detects vehicles, pedestrians, traffic lights, and other objects
- Pre-trained on COCO dataset with 80 object classes
- Returns bounding boxes, class IDs, and confidence scores
- Real-time processing with CPU/GPU support

**Lane Detection (OpenVINO)**
- Uses pre-trained deep learning model for lane segmentation
- Processes camera images to detect left and right lane lines
- Fits polynomial curves to detected lane pixels
- Generates smooth trajectory for vehicle following

**Traffic Light Detection**
- Integrated with object detection pipeline
- Identifies traffic light states (red, yellow, green)
- Only considers front-facing traffic lights in driving path
- Triggers appropriate braking behavior

### 2. Control System

**Pure Pursuit Controller**
- Geometric path tracking algorithm
- Uses lookahead distance for smooth steering
- Calculates steering angle based on target waypoint
- Adapts to vehicle speed and path curvature

**PID Controller**
- Controls throttle and brake for speed management
- Maintains desired speed based on road conditions
- Adjusts for traffic light stops and obstacle avoidance
- Provides smooth acceleration and deceleration

**Integrated Control Logic**
- Combines steering from Pure Pursuit and speed from PID
- Handles emergency braking for obstacles
- Manages traffic light compliance
- Ensures safe driving behavior

### 3. Planning & Decision Making

**Rule-Based Behavior System**
- Lane following as primary behavior
- Obstacle detection and avoidance
- Traffic light compliance
- Emergency stopping for pedestrians

**Obstacle Avoidance Logic**
- Detects blocking objects in vehicle's path
- Uses bounding box size to estimate distance
- Different thresholds for different object types
- Immediate braking for close obstacles

**Traffic Rules Implementation**
- Stops for red and yellow traffic lights
- Only considers traffic lights in front center area
- Continues on green lights
- Handles unknown traffic light states safely

### 4. LLM Explainability System

**Threaded Processing**
- Runs LLM explanations in separate thread
- Non-blocking operation maintains real-time performance
- Processes explanations every 3 seconds (60 frames)
- Queues requests to avoid overwhelming the API

**Deepseek API Integration**
- Uses Deepseek Chat model for explanation generation
- Formats driving context into structured prompts
- Processes vehicle state, environment, and decisions
- Returns natural language explanations

**Context Processing**
- Categorizes detected objects by relevance
- Includes vehicle dynamics and control inputs
- Provides road information and waypoint data
- Formats blocking objects and traffic light states

## Data Flow

1. CARLA provides camera sensor data
2. YOLOv8 detects objects and traffic lights
3. OpenVINO processes lane detection
4. Planning system decides vehicle actions
5. Pure Pursuit + PID generates control commands
6. LLM explainer creates natural language explanations
7. Visual feedback shows system status

## Key Advantages of Integrated Architecture

**Performance Benefits**
- Reduced inter-module communication overhead
- Faster data processing and decision making
- Direct access to all system variables
- Optimized memory usage

**Simplicity Benefits**
- Single file contains complete system
- Easier debugging and modification
- Clear data flow and dependencies
- Reduced complexity for new developers

**Reliability Benefits**
- Fewer failure points
- Better error handling
- Consistent system state
- Easier testing and validation

## File Structure

**Main Implementation**
- `carla_sim_llm.py`: Complete integrated system
- `src/explainability/llm_explainer.py`: LLM processing modules
- `src/perception/object_detection.py`: YOLO detector wrapper
- `src/perception/traffic_light_detector.py`: Traffic light processing

**Legacy Modules** (for reference)
- `src/control/`: PID and Pure Pursuit implementations
- `src/planning/`: Behavior and path planning modules
- `src/utils/`: CARLA utilities and visualization

## Configuration

The system uses command-line arguments for configuration:
- `--fps`: Simulation frame rate
- `--map`: CARLA map selection (1-10)
- `--weather`: Weather preset index
- `--no-llm`: Disable LLM explanations
- `--no-yolo`: Disable object detection

## Extensibility

The integrated architecture still supports extensions:

1. **Perception**: Add new detection models in perception functions
2. **Control**: Modify controller parameters or add new algorithms  
3. **Planning**: Extend decision-making logic
4. **Explainability**: Change LLM models or improve prompts
5. **Environment**: Add more NPC vehicles or complex scenarios 