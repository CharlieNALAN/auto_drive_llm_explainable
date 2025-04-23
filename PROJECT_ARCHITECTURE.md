# LLM-Based Explainable Autonomous Driving Project Architecture

This document shows the architecture of the LLM-based explainable autonomous driving system.

## Overview

The project creates an autonomous driving system that:
1. Runs in the CARLA simulator
2. Uses pre-trained models for perception
3. Has a rule-based planning system
4. Explains its actions using Large Language Models (LLMs)

## System Architecture

```
CARLA Simulator <---> Main Loop (main.py)
    |
    v
+---------------------+    +-----------------+    +------------------+   +--------------+
| Perception          |    | Prediction      |    | Planning         |   | Control      |
| - Object Detection  |--->| - Trajectory    |--->| - Behavior       |-->| - PID        |
| - Lane Detection    |    |   Prediction    |    | - Path Planning  |   |   Controller |
+---------------------+    +-----------------+    +------------------+   +--------------+
                                                         |
                                                         v
                                              +---------------------------+
                                              | LLM Explainability Module |
                                              | - Natural Language        |
                                              |   Explanations            |
                                              +---------------------------+
```

## Component Explanation

### 1. Perception Module

**Object Detection**
- Uses YOLOv8 for detecting objects like vehicles, pedestrians, traffic lights
- Pre-trained on COCO dataset
- Returns bounding boxes, class IDs, and confidence scores

**Lane Detection**
- Uses DeepLabV3+ for semantic segmentation
- Also includes traditional computer vision methods as fallback
- Returns lane line information

### 2. Prediction Module

**Trajectory Prediction**
- Uses Kalman filters to track objects
- Predicts future trajectories of detected objects
- Helps the planning module anticipate the movement of other road users

### 3. Planning Module

**Behavior Planner**
- Rule-based behavior planning
- Handles decisions like lane following, vehicle following, lane changes
- Provides clear explanations for each decision

**Path Planner**
- Generates trajectories based on behavior decisions
- Uses simplified Frenet frame planning
- Creates path points and speed profiles

### 4. Control Module

**PID Controller**
- Simple PID controller for lateral (steering) and longitudinal (throttle/brake) control
- Follows the planned trajectory

### 5. LLM Explainability Module

The core innovation of this system:

- Receives inputs from all other modules
- Formats this information into a natural language prompt
- Uses a Large Language Model to generate human-readable explanations
- Supports multiple LLM options:
  - Local LLM (Llama 2/3, quantized for efficient inference)
  - OpenAI API (GPT-3.5/4)
  - Anthropic API (Claude)

## Data Flow

1. CARLA provides sensor data (camera, lidar)
2. Perception models process this data to identify objects and lanes
3. Prediction models track objects and predict trajectories
4. Behavior planner decides what the vehicle should do next
5. Path planner creates a detailed trajectory
6. Controller executes the trajectory
7. LLM explainer creates natural language explanations

## Key Files

- `main.py`: Main entry point, orchestrates all components
- `run_carla.py`: Script to start CARLA simulator
- `src/perception/`: Object and lane detection
- `src/prediction/`: Trajectory prediction
- `src/planning/`: Behavior and path planning
- `src/control/`: Vehicle control
- `src/explainability/`: LLM-based explanation generation
- `src/utils/`: Utility functions for CARLA, visualization, etc.

## Configuration

The system is configured using:
- Command-line arguments in `main.py`
- Configuration files in the `configs/` directory
- The default configuration file is `configs/default.json`

## Extensibility

The system is designed to be modular and extensible:

1. **Perception**: New detection models can be added
2. **Prediction**: Alternative prediction algorithms can be implemented
3. **Planning**: More sophisticated planning algorithms can replace the rule-based approach
4. **Control**: Advanced controllers like MPC can be added
5. **Explainability**: Different LLMs can be used by changing configuration 