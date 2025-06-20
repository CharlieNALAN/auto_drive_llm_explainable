#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import logging
import json
import pygame
from pathlib import Path
import numpy as np

try:
    import carla
except ImportError:
    raise ImportError("CARLA Python API module not found. Make sure it's in your PYTHONPATH")

from src.perception.object_detection import ObjectDetector

try:
    from src.perception.openvino_lane_detector import OpenVINOLaneDetector
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from src.perception.lane_detection import LaneDetector
from src.perception.camera_geometry import CameraGeometry
from src.prediction.trajectory_prediction import TrajectoryPredictor
from src.planning.behavior_planner import BehaviorPlanner
from src.planning.path_planner import PathPlanner
from src.control.pid_controller import PIDController
from src.control.pure_pursuit import PurePursuitPlusPID
from src.explainability.llm_explainer import LLMExplainer
from src.utils.carla_utils import CarlaWorld, CarlaSensors
from src.utils.visualization import Visualization
from src.utils.trajectory_utils import (
    get_trajectory_from_lane_detector, 
    get_trajectory_from_map, 
    send_control,
    dist_point_linestring
)
from src.control.get_target_point import get_curvature

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-based Explainable Autonomous Driving System")
    
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--tm_port', default=8000, type=int, help='CARLA Traffic Manager port')
    parser.add_argument('--config', default='configs/default.json', help='Configuration file')
    parser.add_argument('--sync', action='store_true', help='Enable synchronous mode')
    parser.add_argument('--no-rendering', action='store_true', help='Disable rendering')
    parser.add_argument('--res', default='1280x720', help='Window resolution')
    parser.add_argument('--filter', default='vehicle.tesla.model3', help='Actor filter')
    parser.add_argument('--llm-model', default='local', choices=['local', 'openai', 'anthropic'], 
                        help='LLM model to use for explanations')
    parser.add_argument('--log', default='info', help='Logging level')
    
    return parser.parse_args()

def setup_logging(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/system.log"),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    setup_logging(args.log)
    logger = logging.getLogger(__name__)
    logger.info("Starting LLM-based Explainable Autonomous Driving System")
    
    # Load configuration
    config = load_config(args.config)
    
    # Parse resolution
    width, height = map(int, args.res.split('x'))
    
    try:
        # Connect to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        # Get CARLA world
        world = CarlaWorld(client, args.sync, args.tm_port)
        
        # Set up sensors
        sensors = CarlaSensors(world.world, world.player, width, height)
        
        # Initialize camera geometry for lane detection
        cg = CameraGeometry()
        
        # Initialize modules with new approach
        object_detector = ObjectDetector(config["perception"]["object_detection"])
        
        # Try OpenVINO lane detector first, fallback to regular lane detector
        try:
            if OPENVINO_AVAILABLE:
                lane_detector = OpenVINOLaneDetector(cam_geom=cg)
                print("✓ Using OpenVINO lane detector")
            else:
                raise ImportError("OpenVINO not available")
        except Exception as e:
            print(f"ℹ OpenVINO lane detector failed ({e}), using fallback detector")
            lane_detector = LaneDetector(config.get("perception", {}).get("lane_detection"), cam_geom=cg)
        trajectory_predictor = TrajectoryPredictor(config["prediction"])
        behavior_planner = BehaviorPlanner(config["planning"]["behavior"])
        path_planner = PathPlanner(config["planning"]["path"])
        
        # Use proven pure pursuit controller from working project
        controller = PurePursuitPlusPID()
        
        # Initialize LLM explainer
        explainer = LLMExplainer(config["explainability"], args.llm_model)
        
        # Initialize visualization if rendering is enabled
        if not args.no_rendering:
            viz = Visualization(width, height)
        
        # Main loop
        clock = pygame.time.Clock()
        
        try:
            while True:
                if args.sync:
                    world.world.tick()
                clock.tick_busy_loop(30)
                
                if sensors.get_data():
                    # Process sensor data
                    camera_img = sensors.camera_data
                    
                    # Get vehicle state
                    vehicle_speed = sensors.get_vehicle_speed()
                    
                    # Simplified lane-following approach from working project
                    try:
                        # Primary approach: Use lane detector
                        trajectory, img = get_trajectory_from_lane_detector(lane_detector, camera_img)
                        lane_detection_success = True
                    except Exception as e:
                        print(f"Lane detection failed: {e}. Using map fallback.")
                        # Fallback: Use CARLA map navigation
                        trajectory = get_trajectory_from_map(world.world.get_map(), world.player)
                        img = None
                        lane_detection_success = False
                    
                    # Adaptive speed control based on road curvature
                    max_curvature = get_curvature(np.array(trajectory))
                    if max_curvature > 0.005:
                        # Limit speed when turning (adapted from working project)
                        move_speed = np.abs(5.56 - 20 * max_curvature)
                        move_speed = max(move_speed, 3.0)  # Minimum speed of 3 m/s (10.8 km/h)
                    else:
                        move_speed = 5.56  # 20 km/h in m/s for straight roads
                    
                    # Pure pursuit control (proven approach)
                    dt = 1.0 / 30.0  # 30 FPS
                    throttle, steer = controller.get_control(trajectory, vehicle_speed, move_speed, dt)
                    
                    # Apply control to vehicle
                    brake = 0.0
                    if throttle < 0:
                        brake = -throttle
                        throttle = 0.0
                    
                    send_control(world.player, throttle, steer, brake)
                    
                    # Calculate performance metrics
                    cross_track_error = dist_point_linestring(np.array([0, 0]), trajectory)
                    
                    # For LLM explanation: still run perception and planning for context
                    detected_objects = object_detector.detect(camera_img)
                    
                    # Format lane_info properly for visualization
                    if trajectory is not None:
                        lane_info = {
                            'lanes': [{'points': trajectory.tolist() if hasattr(trajectory, 'tolist') else trajectory}],
                            'trajectory': trajectory,
                            'success': lane_detection_success,
                            'cross_track_error': cross_track_error
                        }
                    else:
                        lane_info = {
                            'lanes': [],
                            'trajectory': [],
                            'success': False,
                            'cross_track_error': cross_track_error
                        }
                    
                    predicted_trajectories = trajectory_predictor.predict(detected_objects)
                    
                    # Simplified behavior planning for explanation
                    behavior_command, behavior_reason = behavior_planner.plan(
                        detected_objects, lane_info, predicted_trajectories, world.player
                    )
                    
                    # Create control data for recording
                    control_data = {
                        'throttle': throttle,
                        'steer': steer,
                        'brake': brake,
                        'speed': vehicle_speed,
                        'target_speed': move_speed,
                        'cross_track_error': cross_track_error,
                        'curvature': max_curvature
                    }
                    
                    # Explainability with enhanced context
                    explainability_input = {
                        "vehicle_state": {
                            "speed": vehicle_speed,
                            "target_speed": move_speed,
                            "location": world.player.get_location(),
                            "control": control_data,
                            "cross_track_error": cross_track_error,
                            "road_curvature": max_curvature
                        },
                        "perception": {
                            "detected_objects": detected_objects,
                            "lane_info": lane_info,
                            "lane_detection_method": "deep_learning" if lane_detection_success else "map_fallback"
                        },
                        "prediction": {
                            "trajectories": predicted_trajectories
                        },
                        "planning": {
                            "behavior": behavior_command,
                            "reason": behavior_reason,
                            "trajectory": trajectory,
                            "control_method": "pure_pursuit"
                        }
                    }
                    
                    explanation = explainer.explain(explainability_input)
                    logger.info(f"Explanation: {explanation}")
                    
                    # Visualization
                    if not args.no_rendering:
                        try:
                            # Ensure data structures are properly formatted
                            if camera_img is None or not isinstance(camera_img, np.ndarray):
                                logger.warning("Camera image is missing or invalid for visualization")
                                continue
                                
                            # Create visualization data structures
                            objects_to_display = detected_objects if detected_objects is not None else []
                            lanes_to_display = lane_info if lane_info is not None else {'lanes': []}
                            trajectories_to_display = predicted_trajectories if predicted_trajectories is not None else {}
                            
                            # Path to display - convert trajectory numpy array to expected format
                            if trajectory is not None:
                                path_to_display = {
                                    'points': trajectory.tolist() if hasattr(trajectory, 'tolist') else trajectory,
                                    'control_data': control_data
                                }
                            else:
                                path_to_display = {'points': []}
                            
                            explanation_to_display = explanation if explanation is not None else "No explanation available"
                            
                            # Additional debug info for new system
                            debug_info = {
                                'lane_detection_success': lane_detection_success,
                                'cross_track_error': cross_track_error,
                                'curvature': max_curvature,
                                'target_speed': move_speed,
                                'actual_speed': vehicle_speed
                            }
                            
                            viz.display(
                                camera_img, objects_to_display, lanes_to_display, 
                                trajectories_to_display, path_to_display, explanation_to_display
                            )
                        except Exception as e:
                            logger.error(f"Error in visualization: {e}", exc_info=True)
                    
                # Check for exit conditions
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_ESCAPE:
                            return
        
        finally:
            # Clean up
            sensors.destroy()
            world.destroy()
            if not args.no_rendering:
                viz.destroy()
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!') 