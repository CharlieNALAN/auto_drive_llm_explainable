#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import carla
from src.planning.behavior_planner import BehaviorState

class PathPlanner:
    """Path planner that generates a trajectory based on behavior command."""
    
    def __init__(self, config):
        """
        Initialize the path planner.
        
        Args:
            config: Configuration dictionary for path planning.
        """
        self.config = config
        self.method = config.get('method', 'frenet')
        self.planning_horizon = config.get('planning_horizon', 50.0)  # meters
        self.lateral_safety_margin = config.get('lateral_safety_margin', 1.0)  # meters
        self.longitudinal_safety_margin = config.get('longitudinal_safety_margin', 5.0)  # meters
        
        # Waypoints for the current plan
        self.current_waypoints = []
        
        # Target speed from behavior planner
        self.target_speed = 30.0  # km/h, will be updated by behavior
    
    def plan(self, behavior_command, detected_objects, lane_info, predicted_trajectories, ego_vehicle):
        """
        Plan a trajectory based on the behavior command.
        
        Args:
            behavior_command: Behavior command from the behavior planner.
            detected_objects: List of detected objects.
            lane_info: Lane information.
            predicted_trajectories: Dictionary of predicted trajectories.
            ego_vehicle: Ego vehicle object.
            
        Returns:
            Planned trajectory.
        """
        # Get ego vehicle state
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)  # km/h
        
        # Extract lane information for planning
        lanes = self._get_lanes_for_planning(lane_info)
        
        # Determine target lane based on behavior
        target_lane = self._get_target_lane(behavior_command, lanes)
        
        # Generate trajectory based on the planning method
        if self.method == 'frenet':
            trajectory = self._plan_frenet(behavior_command, target_lane, ego_location, 
                                          ego_transform, ego_speed, detected_objects, 
                                          predicted_trajectories)
        else:
            raise ValueError(f"Unknown planning method: {self.method}")
        
        return trajectory
    
    def _get_lanes_for_planning(self, lane_info):
        """Extract lane information for planning."""
        lanes = []
        
        if lane_info and 'lanes' in lane_info:
            for lane in lane_info['lanes']:
                if 'points' in lane and len(lane['points']) >= 2:
                    lanes.append(lane)
        
        # If no lanes detected, create a simple forward lane
        if not lanes:
            # Create a simple straight lane ahead
            points = [(320, 720), (320, 360)]
            lanes.append({
                'points': points,
                'confidence': 1.0,
                'side': 'center'
            })
        
        return lanes
    
    def _get_target_lane(self, behavior_command, lanes):
        """Determine target lane based on behavior command."""
        # Default to the first lane
        if not lanes:
            return None
        
        # Find lanes by side
        left_lanes = [lane for lane in lanes if lane.get('side') == 'left']
        right_lanes = [lane for lane in lanes if lane.get('side') == 'right']
        center_lanes = [lane for lane in lanes if lane.get('side') == 'center']
        
        # If sides not specified, try to infer from position
        if not (left_lanes or right_lanes):
            # Sort lanes by x-position of first point
            lanes_sorted = sorted(lanes, key=lambda lane: lane['points'][0][0])
            if len(lanes_sorted) >= 2:
                left_lanes = [lanes_sorted[0]]
                right_lanes = [lanes_sorted[-1]]
                if len(lanes_sorted) >= 3:
                    center_lanes = lanes_sorted[1:-1]
            elif len(lanes_sorted) == 1:
                center_lanes = lanes_sorted
        
        # Select lane based on behavior
        if behavior_command in [BehaviorState.EXECUTE_LANE_CHANGE_LEFT, BehaviorState.PREPARE_LANE_CHANGE_LEFT]:
            return left_lanes[0] if left_lanes else (center_lanes[0] if center_lanes else lanes[0])
        elif behavior_command in [BehaviorState.EXECUTE_LANE_CHANGE_RIGHT, BehaviorState.PREPARE_LANE_CHANGE_RIGHT]:
            return right_lanes[0] if right_lanes else (center_lanes[0] if center_lanes else lanes[0])
        else:
            return center_lanes[0] if center_lanes else lanes[0]
    
    def _plan_frenet(self, behavior_command, target_lane, ego_location, ego_transform, 
                    ego_speed, detected_objects, predicted_trajectories):
        """
        Plan trajectory using Frenet coordinates.
        
        This is a simplified implementation of Frenet path planning.
        """
        if not target_lane or 'points' not in target_lane or len(target_lane['points']) < 2:
            # No valid lane, create a straight path ahead
            forward_vector = ego_transform.get_forward_vector()
            points = []
            
            # Sample points along the forward direction
            for i in range(10):
                distance = i * 5.0  # 5m intervals
                point = (
                    ego_location.x + forward_vector.x * distance,
                    ego_location.y + forward_vector.y * distance
                )
                points.append(point)
            
            return {
                'points': points,
                'speed_profile': [ego_speed] * len(points),
                'behavior': behavior_command
            }
        
        # Extract lane points
        lane_points = target_lane['points']
        
        # Convert to 3D world coordinates (simplified, would use actual projection in practice)
        # In a real system, lane points would already be in world coordinates from mapping
        world_points = []
        for i, (x, y) in enumerate(lane_points):
            # Simple conversion based on image coordinates to world (highly simplified)
            # In practice, you would use camera calibration and 3D reconstruction
            world_x = ego_location.x + (i * 5.0)  # Assume 5m spacing
            world_y = ego_location.y + (x - 320) / 10.0  # Center offset
            world_points.append((world_x, world_y))
        
        # Create speed profile based on behavior
        speed_profile = []
        target_speed = ego_speed
        
        if behavior_command == BehaviorState.FOLLOW_LANE:
            # Maintain or increase speed
            target_speed = min(30.0, ego_speed + 5.0)
        elif behavior_command == BehaviorState.FOLLOW_VEHICLE:
            # Adjust speed to follow vehicle ahead
            target_speed = max(0.0, ego_speed - 2.0)
        elif behavior_command == BehaviorState.STOP_FOR_TRAFFIC_LIGHT or behavior_command == BehaviorState.STOP_FOR_OBSTACLE:
            # Decelerate to stop
            target_speed = max(0.0, ego_speed - 10.0)
        elif behavior_command == BehaviorState.EMERGENCY_STOP:
            # Hard brake
            target_speed = 0.0
        
        # Create speed profile
        for i in range(len(world_points)):
            # Gradually adjust speed to target
            if i == 0:
                speed_profile.append(ego_speed)
            else:
                prev_speed = speed_profile[-1]
                if prev_speed < target_speed:
                    # Accelerate gradually
                    speed_profile.append(min(target_speed, prev_speed + 2.0))
                elif prev_speed > target_speed:
                    # Decelerate gradually
                    speed_profile.append(max(target_speed, prev_speed - 5.0))
                else:
                    # Maintain speed
                    speed_profile.append(target_speed)
        
        # Create trajectory
        trajectory = {
            'points': world_points,
            'speed_profile': speed_profile,
            'behavior': behavior_command
        }
        
        return trajectory 