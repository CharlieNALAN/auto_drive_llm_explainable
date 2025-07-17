# LLM-Based Explainable Autonomous Driving System

This project builds an explainable autonomous driving system using Large Language Models. The system can drive in the CARLA simulator. It explains its actions in natural language.

## Project Structure

```
├── configs/             # Configuration files
├── src/                 # Source code modules
│   ├── perception/      # Object detection and lane detection
│   ├── prediction/      # Trajectory prediction
│   ├── planning/        # Path and behavior planning
│   ├── control/         # Vehicle control algorithms
│   ├── explainability/  # LLM-based explanation modules
│   └── utils/           # Utility functions
├── carla_sim_llm.py     # Main entry point (integrated implementation)
├── run_carla.py         # Script to start CARLA server
└── requirements.txt     # Python dependencies
```

## Current Implementation

The main system now runs from `carla_sim_llm.py`. This file contains a complete integrated implementation. It includes:

- **Object Detection**: YOLOv8 for detecting vehicles, pedestrians, and traffic lights
- **Lane Detection**: OpenVINO-based lane line detection with polynomial fitting
- **Vehicle Control**: Pure Pursuit controller combined with PID for smooth driving
- **Obstacle Avoidance**: Real-time detection and stopping for blocking objects
- **LLM Explanations**: Threaded LLM explainer using Deepseek API
- **Traffic Management**: Automatic spawning of NPC vehicles and pedestrians

## Setup

1. Install CARLA simulator. Follow the [official documentation](https://carla.readthedocs.io/en/latest/start_quickstart/).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model:
   ```bash
   # The system uses yolov8n.pt which should be in the project root
   # If not present, it will be downloaded automatically
   ```

## Usage

1. Start CARLA server:
   ```bash
   python run_carla.py
   ```

2. Then Run the autonomous driving system:
   ```bash
   python carla_sim_llm.py
   ```

3. Available command line options:
   ```bash
   python carla_sim_llm.py --help
   
   # Examples:
   python carla_sim_llm.py --fps 20 --map 2 --weather 1
   python carla_sim_llm.py --no-llm --no-yolo  # Disable LLM and YOLO
   ```

## Features

- **Intelligent Lane Detection**: Uses OpenVINO for fast and accurate lane detection
- **Object Detection**: YOLOv8-based detection of vehicles, pedestrians, and traffic lights  
- **Smart Obstacle Avoidance**: Automatically stops for vehicles, pedestrians, and red lights
- **Smooth Control**: Pure Pursuit + PID controller for natural driving behavior
- **Real-time Explanations**: LLM generates natural language explanations of driving decisions
- **Realistic Traffic**: Automatic spawning of NPC vehicles and pedestrians
- **Visual Feedback**: Real-time display of detection results and vehicle status

## System Requirements

- Python 3.8+
- CARLA simulator 0.9.13+
- OpenCV for image processing
- PyTorch for YOLO detection
- OpenVINO runtime for lane detection
- GPU recommended for better performance 