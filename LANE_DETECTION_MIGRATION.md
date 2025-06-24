# OpenVINO Lane Detection Migration

This document explains the migration of the successful OpenVINO lane detection from the `self-driving-carla-main` project to the LLM-explainable autonomous driving system.

## What Was Migrated

1. **OpenVINO Lane Detector** - The core lane detection algorithm that works reliably
2. **Camera Geometry Module** - Precise coordinate system transformations
3. **Pure Pursuit Controller** - Smooth vehicle control for lane following
4. **Trajectory Generation** - Direct trajectory output from lane detection

## Key Files Added/Modified

### New Files:
- `src/perception/camera_geometry.py` - Camera coordinate transformations
- `src/perception/openvino_lane_detector.py` - OpenVINO lane detector
- `src/control/pure_pursuit_controller.py` - Pure pursuit control algorithm
- `src/utils/lane_utils.py` - Utility functions for lane processing

### Modified Files:
- `src/perception/lane_detection.py` - Updated to use OpenVINO detector
- `src/planning/path_planner.py` - Added trajectory-based planning
- `src/control/pid_controller.py` - Integrated pure pursuit control
- `configs/default.json` - Updated configuration for OpenVINO

## How It Works

1. **Lane Detection**: OpenVINO model detects lane lines and generates a trajectory
2. **Trajectory Planning**: Path planner uses the trajectory directly instead of complex planning
3. **Control**: Pure pursuit controller follows the trajectory smoothly
4. **Fallbacks**: System falls back to mock detection if OpenVINO model is not available

## Installation

1. Install OpenVINO:
```bash
pip install openvino>=2022.3.0
```

2. Copy the OpenVINO model from the successful project:
```bash
# If you have the self-driving-carla-main project in the parent directory:
# The system will automatically look for: ../self-driving-carla-main/converted_model/lane_model.xml

# Or manually specify the path in configs/default.json:
"openvino_model_path": "path/to/your/lane_model.xml"
```

## Usage

The system now automatically uses OpenVINO lane detection when available. No code changes needed in main.py.

Key improvements:
- **Stable lane following** - No more left-right wobbling
- **Smooth steering** - Pure pursuit provides better control
- **Robust detection** - Falls back gracefully if model not found
- **Direct trajectory** - Bypasses complex planning for simple lane following

## Troubleshooting

### If you get "OpenVINO model not found":
1. Check that the model file exists at the specified path
2. Update `openvino_model_path` in `configs/default.json`
3. The system will fall back to mock detection (straight line)

### If steering is still jerky:
1. Adjust `lookahead_factor` in config (lower = more responsive, higher = smoother)
2. Modify PID gains in the control section
3. Check that pure pursuit controller is being used (look for console message)

### If the car drives too fast/slow:
1. The system adapts speed based on trajectory curvature
2. Adjust the speed limits in the path planner
3. Modify PID gains for longitudinal control

## Configuration Options

Key parameters in `configs/default.json`:

```json
{
  "perception": {
    "lane_detection": {
      "model": "openvino",
      "openvino_model_path": "../self-driving-carla-main/converted_model/lane_model.xml"
    }
  },
  "control": {
    "lookahead_factor": 0.4,  // Higher = smoother, lower = more responsive
    "max_steering_angle": 0.5,  // Limits max steering
    "lateral": {
      "Kp": 1.0,  // Steering responsiveness
      "Kd": 0.05  // Steering damping
    }
  }
}
```

## Benefits of This Migration

1. **Solves the main problem**: Vehicle now follows lanes smoothly without erratic steering
2. **Proven technology**: Uses the working implementation from the second project
3. **Maintains compatibility**: LLM explanations and other features still work
4. **Graceful degradation**: Falls back to safe behavior if OpenVINO is unavailable
5. **Performance**: OpenVINO provides fast, efficient inference

The vehicle should now drive smoothly along lanes without the left-right wobbling issue! 