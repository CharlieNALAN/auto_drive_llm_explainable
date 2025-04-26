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
        
        This implementation includes better obstacle avoidance and curve handling.
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
        
        # Check if we have polynomial curve coefficients from enhanced lane detection
        has_curve_data = 'curve_coeffs' in target_lane
        
        # Convert to 3D world coordinates with improved method for curves
        world_points = []
        
        if has_curve_data:
            # Use polynomial coefficients for smoother curve representation
            coeffs = target_lane['curve_coeffs']
            
            # Generate more points for smoother curves, especially in turns
            num_points = 20  # Increased from typical 10 points
            
            if len(coeffs) == 3:  # Quadratic fit (ax^2 + bx + c)
                # Calculate y range in image coordinates
                y_min = min([p[1] for p in lane_points])
                y_max = max([p[1] for p in lane_points])
                
                # Generate points along the curve
                y_values = np.linspace(y_min, y_max, num_points)
                
                for y in y_values:
                    # Calculate x using polynomial: x = ay^2 + by + c
                    x = coeffs[0] * (y**2) + coeffs[1] * y + coeffs[2]
                    
                    # Transform to world coordinates (with improved scaling)
                    # More aggressive mapping from image to world for better curve response
                    world_x = ego_location.x + (y_max - y) / (y_max - y_min) * self.planning_horizon
                    world_y = ego_location.y + (x - 640) / 8.0  # Reduced divisor for more pronounced steering
                    
                    world_points.append((world_x, world_y))
            else:  # Linear fit (mx + b)
                # Similar process for linear fit
                y_min = min([p[1] for p in lane_points])
                y_max = max([p[1] for p in lane_points])
                
                y_values = np.linspace(y_min, y_max, num_points)
                
                for y in y_values:
                    # Calculate x using linear equation: x = my + b
                    x = coeffs[0] * y + coeffs[1]
                    
                    # Transform to world coordinates
                    world_x = ego_location.x + (y_max - y) / (y_max - y_min) * self.planning_horizon
                    world_y = ego_location.y + (x - 640) / 10.0
                    
                    world_points.append((world_x, world_y))
        else:
            # Fall back to original method for basic lanes
            for i, (x, y) in enumerate(lane_points):
                # Simple conversion based on image coordinates to world (highly simplified)
                world_x = ego_location.x + (i * 5.0)  # Assume 5m spacing
                world_y = ego_location.y + (x - 320) / 10.0  # Center offset
                world_points.append((world_x, world_y))
        
        # Ensure we have enough points
        if len(world_points) < 5:
            # Interpolate to get more points
            x_values = [p[0] for p in world_points]
            y_values = [p[1] for p in world_points]
            
            # Create interpolation function
            if len(world_points) >= 2:
                x_interp = np.linspace(min(x_values), max(x_values), 10)
                y_interp = np.interp(x_interp, x_values, y_values)
                
                world_points = [(x, y) for x, y in zip(x_interp, y_interp)]
        
        # Find potential obstacles in our path
        path_obstacles = self._find_obstacles_in_path(world_points, detected_objects, predicted_trajectories, ego_transform)
        
        # If we have obstacles in our path, adjust the trajectory for collision avoidance
        if path_obstacles and behavior_command not in [BehaviorState.EXECUTE_LANE_CHANGE_LEFT, BehaviorState.EXECUTE_LANE_CHANGE_RIGHT]:
            # Sort obstacles by distance
            path_obstacles.sort(key=lambda x: x['distance'])
            
            # Get closest obstacle
            closest_obstacle = path_obstacles[0]
            obstacle_distance = closest_obstacle['distance']
            obstacle_lateral = closest_obstacle['lateral_offset']
            
            # If obstacle is too close, apply evasive maneuver
            if obstacle_distance < self.planning_horizon / 2:
                # Amount of lateral offset needed (opposite to obstacle's lateral position)
                evasion_offset = -1.0 * obstacle_lateral / max(1.0, abs(obstacle_lateral)) * min(1.5, self.lateral_safety_margin * 1.5)
                
                # Apply offset to the world points after the obstacle position
                obstacle_idx = int(obstacle_distance / 5.0)  # Approximate index in world_points
                
                # Gradually apply the offset to create a smooth evasive trajectory
                for i in range(len(world_points)):
                    if i > obstacle_idx:
                        # Increase offset gradually, then decrease after passing
                        offset_factor = min(1.0, (i - obstacle_idx) / 3.0) * max(0, 1.0 - (i - obstacle_idx - 5) / 5.0)
                        world_points[i] = (world_points[i][0], world_points[i][1] + evasion_offset * offset_factor)
        
        # Create speed profile based on path curvature and obstacles
        speed_profile = self._create_speed_profile_with_curvature(world_points, behavior_command, ego_speed, path_obstacles)
        
        # Create trajectory
        trajectory = {
            'points': world_points,
            'speed_profile': speed_profile,
            'behavior': behavior_command
        }
        
        return trajectory
        
    def _create_speed_profile_with_curvature(self, path_points, behavior_command, ego_speed, path_obstacles):
        """
        Create a speed profile that accounts for path curvature and obstacles.
        
        Args:
            path_points: List of (x, y) points in the planned path.
            behavior_command: Behavior command from the behavior planner.
            ego_speed: Current speed of the ego vehicle.
            path_obstacles: List of obstacles in the path.
            
        Returns:
            List of target speeds for each point in the path.
        """
        # Initialize speed profile with current speed
        speed_profile = [ego_speed] * len(path_points)
        
        # Base target speed on behavior command
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
        else:
            # Default
            target_speed = ego_speed
        
        # Calculate path curvature to adjust speed in turns
        if len(path_points) >= 3:
            # Calculate curvature at each point (except first and last)
            for i in range(1, len(path_points) - 1):
                prev_point = path_points[i-1]
                current_point = path_points[i]
                next_point = path_points[i+1]
                
                # Calculate vectors between points
                v1 = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
                v2 = (next_point[0] - current_point[0], next_point[1] - current_point[1])
                
                # Calculate the change in direction (approximation of curvature)
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
                v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
                
                if v1_mag * v2_mag > 0:
                    # Calculate the cosine of the angle between vectors
                    cos_angle = dot_product / (v1_mag * v2_mag)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1]
                    
                    # Calculate the angle (in radians)
                    angle = math.acos(cos_angle)
                    
                    # Higher angle means sharper turn
                    # Reduce speed based on curvature
                    curvature_factor = 1.0 - (angle / math.pi) * 1.5  # Scale to get reasonable reduction
                    curvature_factor = max(0.5, curvature_factor)  # Don't slow down beyond 50%
                    
                    # Apply curvature-based speed reduction
                    speed_profile[i] = min(speed_profile[i], target_speed * curvature_factor)
                    
                    # Propagate reduced speed to nearby points for smoother deceleration/acceleration
                    if angle > 0.2:  # Only for significant turns
                        # Slow down before the turn
                        for j in range(max(0, i-3), i):
                            approach_factor = 0.7 + 0.3 * (j - (i-3)) / 3  # Gradually slow down
                            speed_profile[j] = min(speed_profile[j], target_speed * approach_factor)
                        
                        # Speed up after the turn
                        for j in range(i+1, min(i+4, len(speed_profile))):
                            exit_factor = 0.7 + 0.3 * (j - i) / 3  # Gradually speed up
                            speed_profile[j] = min(speed_profile[j], target_speed * exit_factor)
        
        # Adjust speed profile based on obstacles
        if path_obstacles:
            closest_obstacle = path_obstacles[0]
            obstacle_distance = closest_obstacle['distance']
            
            # Adjust speed based on distance to obstacle
            if obstacle_distance < self.longitudinal_safety_margin:
                # Very close - emergency stop
                speed_profile = [0.0] * len(speed_profile)
            elif obstacle_distance < self.longitudinal_safety_margin * 2:
                # Close - slow down significantly
                for i in range(len(speed_profile)):
                    speed_profile[i] = min(speed_profile[i], 5.0)
            elif obstacle_distance < self.longitudinal_safety_margin * 4:
                # Moderate distance - slow down
                for i in range(len(speed_profile)):
                    speed_profile[i] = min(speed_profile[i], 15.0)
        
        # Ensure smooth transitions in speed profile
        smoothed_profile = [speed_profile[0]]
        for i in range(1, len(speed_profile)):
            prev_speed = smoothed_profile[-1]
            current_target = speed_profile[i]
            
            if prev_speed < current_target:
                # Accelerate gradually
                smoothed_profile.append(min(current_target, prev_speed + 2.0))
            elif prev_speed > current_target:
                # Decelerate gradually (faster than acceleration)
                smoothed_profile.append(max(current_target, prev_speed - 5.0))
            else:
                # Maintain speed
                smoothed_profile.append(current_target)
        
        return smoothed_profile
        
    def _find_obstacles_in_path(self, path_points, detected_objects, predicted_trajectories, ego_transform):
        """
        Find obstacles that are in the planned path.
        
        Args:
            path_points: List of (x, y) points in the planned path.
            detected_objects: List of detected objects.
            predicted_trajectories: Dictionary of predicted trajectories.
            ego_transform: Ego vehicle transform.
            
        Returns:
            List of obstacles in the path with their distance and lateral offset.
        """
        obstacles_in_path = []
        
        # Get ego forward vector for reference
        forward_vector = ego_transform.get_forward_vector()
        right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0)
        
        # Define the path corridor width (wider to reduce false positives)
        path_width = 3.0  # meters, increased from 2.5
        
        for obj in detected_objects:
            if obj['class_id'] in [0, 1, 2, 3, 5, 7]:  # Person, bicycle, car, motorcycle, bus, truck
                # Get object position (simplified)
                bbox = obj['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Skip if object is near the edges of the screen (likely false detections)
                if center_x < 50 or center_x > 1230 or center_y < 50 or center_y > 670:
                    continue
                
                # Convert to world coordinates (simplified)
                obj_x = ego_transform.location.x + 40 * (720 - center_y) / 720  # Increased from 30 to 40
                obj_y = ego_transform.location.y + (center_x - 640) / 10
                
                # Check if object is close to any point in the path
                min_distance = float('inf')
                min_lateral_offset = 0
                
                for i, (path_x, path_y) in enumerate(path_points):
                    # Calculate distance to this path point
                    dx = obj_x - path_x
                    dy = obj_y - path_y
                    
                    # Project onto path tangent and normal vectors
                    if i < len(path_points) - 1:
                        # Use the path segment direction
                        next_x, next_y = path_points[i + 1]
                        path_dx = next_x - path_x
                        path_dy = next_y - path_y
                        path_length = math.sqrt(path_dx**2 + path_dy**2)
                        
                        if path_length > 0:
                            path_dx /= path_length
                            path_dy /= path_length
                            
                            # Tangential (longitudinal) and normal (lateral) projections
                            longitudinal = dx * path_dx + dy * path_dy
                            lateral = dx * (-path_dy) + dy * path_dx
                            
                            # Total distance to the path segment
                            if 0 <= longitudinal <= path_length:
                                # Object is alongside this segment
                                distance = abs(lateral)
                                if distance < min_distance:
                                    min_distance = distance
                                    min_lateral_offset = lateral
                
                # If the object is within the path corridor and not too close (avoid false positives)
                if min_distance < path_width and min_distance > 0.5:
                    # Calculate longitudinal distance along the path (simplified)
                    # Using a less strict distance check
                    for i, (path_x, path_y) in enumerate(path_points):
                        dx = obj_x - path_x
                        dy = obj_y - path_y
                        distance = math.sqrt(dx**2 + dy**2)
                        
                        if distance < 7.0:  # Increased from 5.0 for more tolerance
                            # Only consider obstacles ahead and not too far
                            if i > 1 and i < len(path_points) - 2:
                                obstacles_in_path.append({
                                    'id': obj['id'],
                                    'type': obj['class_id'],
                                    'distance': i * 5.0,  # Approximate distance along path
                                    'lateral_offset': min_lateral_offset
                                })
                                break 
        
        return obstacles_in_path 