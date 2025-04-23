# Implementation Guide

This guide explains how to implement and run the LLM-based Explainable Autonomous Driving system.

## Prerequisites

Before starting, make sure you have:

1. **CARLA Simulator**: Download and install [CARLA 0.9.13](https://carla.readthedocs.io/en/latest/start_quickstart/)
2. **Python 3.8+**: The project is built with Python 3.8 or newer
3. **GPU with CUDA support**: For faster model inference (CPU mode also works)
4. **Git**: To clone the repository

## Installation Steps

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/llm-explainable-autonomous-driving.git
cd llm-explainable-autonomous-driving
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download Pre-trained Models**:
```bash
python src/utils/download_models.py --all --install-deps
```

This will:
- Download YOLOv8 for object detection
- Provide instructions for downloading LLaMA models
- Install additional dependencies

4. **Set Environment Variables** (optional, for using API-based LLMs):
```bash
# For OpenAI API
export OPENAI_API_KEY=your_api_key

# For Anthropic API
export ANTHROPIC_API_KEY=your_api_key
```

5. **Configure CARLA Path**:
```bash
# Add CARLA to your Python path (modify based on your installation)
export CARLA_PATH=/path/to/your/carla/installation
```

## Running the System

### Starting CARLA

1. **Start CARLA Server**:
```bash
python run_carla.py --quality-level=Low
```

Or manually start CARLA:
```bash
cd $CARLA_PATH
./CarlaUE4.sh -quality-level=Low
```

### Running the Autonomous Driving System

1. **Basic Run**:
```bash
python main.py
```

2. **With Different LLM Options**:
```bash
# Use local LLM
python main.py --llm-model=local

# Use OpenAI API
python main.py --llm-model=openai

# Use Anthropic API
python main.py --llm-model=anthropic
```

3. **Additional Options**:
```bash
# Change resolution
python main.py --res=800x600

# Disable visualization
python main.py --no-rendering

# Use a different configuration
python main.py --config=configs/custom.json
```

## Creating Custom Configurations

You can modify the default configuration or create a new one:

1. **Copy the Default Configuration**:
```bash
cp configs/default.json configs/custom.json
```

2. **Edit the Configuration File** to adjust parameters for:
   - Perception models
   - Tracking parameters
   - Planning behavior
   - Control parameters
   - LLM settings

## Troubleshooting

1. **CARLA Connection Issues**:
   - Make sure CARLA is running before starting the main script
   - Check if the ports match (default is 2000)
   - Try restarting CARLA

2. **Model Loading Errors**:
   - Ensure models are downloaded correctly
   - For LLaMA models, follow the instructions from `download_models.py`
   - If using GPUs, check CUDA availability with `torch.cuda.is_available()`

3. **Performance Issues**:
   - Lower the resolution with `--res=640x480`
   - Use smaller models with `download_models.py --small`
   - Run CARLA with lower quality `--quality-level=Low`

## Extending the System

### Adding New Perception Models

To add a new object detection model:

1. Modify `src/perception/object_detection.py` to include your model
2. Update the configuration in `configs/default.json`

### Customizing the LLM Explainer

To modify explanations:

1. Edit the prompt template in `configs/default.json` under the `explainability` section
2. Adjust the data formatting in `src/explainability/llm_explainer.py`

### Adding New Planning Behaviors

To add new driving behaviors:

1. Update `src/planning/behavior_planner.py` with new behavior states
2. Implement the corresponding decision logic
3. Ensure the new behaviors have appropriate explanations

## System Evaluation

To evaluate the system's performance:

1. **Perception Accuracy**:
   - Monitor detection results in the visualization
   - Compare with ground truth in CARLA

2. **Planning Quality**:
   - Observe the vehicle's behavior in different scenarios
   - Check if it follows traffic rules

3. **Explanation Quality**:
   - Assess if explanations match the vehicle's actions
   - Check if they are human-understandable

## Next Steps

After implementing the basic system, consider these enhancements:

1. Train models on CARLA data for better performance
2. Implement more sophisticated planning algorithms
3. Add support for complex scenarios like intersections
4. Enhance the explainability module with visual explanations
5. Conduct user studies to evaluate explanation quality 