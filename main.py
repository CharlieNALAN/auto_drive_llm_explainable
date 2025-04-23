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

try:
    import carla
except ImportError:
    raise ImportError("CARLA Python API module not found. Make sure it's in your PYTHONPATH")

from src.perception.object_detection import ObjectDetector
from src.perception.lane_detection import LaneDetector
from src.prediction.trajectory_prediction import TrajectoryPredictor
from src.planning.behavior_planner import BehaviorPlanner
from src.planning.path_planner import PathPlanner
from src.control.pid_controller import PIDController
from src.explainability.llm_explainer import LLMExplainer
from src.utils.carla_utils import CarlaWorld, CarlaSensors
from src.utils.visualization import Visualization

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
        
        # Initialize modules
        object_detector = ObjectDetector(config["perception"]["object_detection"])
        lane_detector = LaneDetector(config["perception"]["lane_detection"])
        trajectory_predictor = TrajectoryPredictor(config["prediction"])
        behavior_planner = BehaviorPlanner(config["planning"]["behavior"])
        path_planner = PathPlanner(config["planning"]["path"])
        controller = PIDController(config["control"])
        
        # Initialize LLM explainer
        explainer = LLMExplainer(config["explainability"], args.llm_model)
        
        # Initialize visualization if rendering is enabled
        if not args.no_rendering:
            visualization = Visualization(width, height)
        
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
                    lidar_data = sensors.lidar_data
                    
                    # Perception
                    detected_objects = object_detector.detect(camera_img)
                    lane_info = lane_detector.detect(camera_img)
                    
                    # Prediction
                    predicted_trajectories = trajectory_predictor.predict(detected_objects)
                    
                    # Planning
                    behavior_command, behavior_reason = behavior_planner.plan(
                        detected_objects, lane_info, predicted_trajectories, world.player
                    )
                    
                    trajectory = path_planner.plan(
                        behavior_command, detected_objects, lane_info, predicted_trajectories, world.player
                    )
                    
                    # Control
                    control = controller.control(trajectory, world.player)
                    world.player.apply_control(control)
                    
                    # Explainability
                    explainability_input = {
                        "vehicle_state": {
                            "speed": sensors.get_vehicle_speed(),
                            "location": world.player.get_location(),
                            "control": control
                        },
                        "perception": {
                            "detected_objects": detected_objects,
                            "lane_info": lane_info
                        },
                        "prediction": {
                            "trajectories": predicted_trajectories
                        },
                        "planning": {
                            "behavior": behavior_command,
                            "reason": behavior_reason,
                            "trajectory": trajectory
                        }
                    }
                    
                    explanation = explainer.explain(explainability_input)
                    logger.info(f"Explanation: {explanation}")
                    
                    # Visualization
                    if not args.no_rendering:
                        visualization.display(
                            camera_img, detected_objects, lane_info, 
                            predicted_trajectories, trajectory, explanation
                        )
                    
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
                pygame.quit()
    
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!') 