# LLM-Based Explainable Autonomous Driving System

This project builds an explainable autonomous driving system using Large Language Models. The system can drive in the CARLA simulator and explain its actions in natural language.

## Project Structure

```
├── configs/             # Configuration files
├── data/                # Data storage
├── logs/                # Logs for training and testing
├── src/                 # Source code
│   ├── perception/      # Perception module (object detection, lane detection)
│   ├── prediction/      # Prediction module (trajectory prediction)
│   ├── planning/        # Planning module (behavior planning, path planning)
│   ├── control/         # Control module (PID controller, etc.)
│   ├── explainability/  # LLM-based explainability module
│   └── utils/           # Utility functions
├── main.py              # Main entry point
├── run_carla.py         # Script to run CARLA
└── requirements.txt     # Dependencies
```

## Setup

1. Install CARLA simulator following the [official documentation](https://carla.readthedocs.io/en/latest/start_quickstart/).
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download pre-trained models:
   ```
   python src/utils/download_models.py
   ```

## Usage

1. Start CARLA server:
   ```
   ./CarlaUE4.sh -quality-level=Low  
   ```
   or
   ```
   ./CarlaUE4.exe -quality-level=Low  # For windows
   ```

2. Run the autonomous driving system:
   ```
   python main.py
   ```

## Features

- Object detection using YOLOv8
- Lane detection using pre-trained segmentation models
- Trajectory prediction using Kalman filters
- Rule-based planning with explainable decision making
- PID controller for vehicle control
- LLM-based natural language explanations of driving decisions 