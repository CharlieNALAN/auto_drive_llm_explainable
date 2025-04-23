#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
import carla

class BehaviorState:
    """Enum-like class for behavior states."""
    FOLLOW_LANE = "follow_lane"
    FOLLOW_VEHICLE = "follow_vehicle"
    PREPARE_LANE_CHANGE_LEFT = "prepare_lane_change_left"
    PREPARE_LANE_CHANGE_RIGHT = "prepare_lane_change_right"
    EXECUTE_LANE_CHANGE_LEFT = "execute_lane_change_left"
    EXECUTE_LANE_CHANGE_RIGHT = "execute_lane_change_right"
    STOP_FOR_TRAFFIC_LIGHT = "stop_for_traffic_light"
    STOP_FOR_OBSTACLE = "stop_for_obstacle"
    WAIT_FOR_CLEAR = "wait_for_clear"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    EMERGENCY_STOP = "emergency_stop"


class BehaviorPlanner:
    """Rule-based behavior planner."""
    
    def __init__(self, config):
        """
        Initialize the behavior planner.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.max_speed = config.get('max_speed', 30.0)  # km/h
        self.safe_distance = config.get('safe_distance', 10.0)  # meters
        self.lane_change_threshold = config.get('lane_change_threshold', 5.0)  # seconds
        
        # Current state
        self.current_state = BehaviorState.FOLLOW_LANE
        
        # Previous decisions for smoothing (prevent rapid changes)
        self.prev_states = []
        self.state_ttl = 10  # How many frames a state remains active (minimum)
        self.state_counter = 0
        
        # Variables to track intention explanations
        self.state_explanations = {
            BehaviorState.FOLLOW_LANE: "staying in current lane",
            BehaviorState.FOLLOW_VEHICLE: "following the vehicle ahead",
            BehaviorState.PREPARE_LANE_CHANGE_LEFT: "preparing to change to the left lane",
            BehaviorState.PREPARE_LANE_CHANGE_RIGHT: "preparing to change to the right lane",
            BehaviorState.EXECUTE_LANE_CHANGE_LEFT: "changing to the left lane",
            BehaviorState.EXECUTE_LANE_CHANGE_RIGHT: "changing to the right lane",
            BehaviorState.STOP_FOR_TRAFFIC_LIGHT: "stopping for a red traffic light",
            BehaviorState.STOP_FOR_OBSTACLE: "stopping for an obstacle",
            BehaviorState.WAIT_FOR_CLEAR: "waiting for the path to clear",
            BehaviorState.TURN_LEFT: "turning left at intersection",
            BehaviorState.TURN_RIGHT: "turning right at intersection",
            BehaviorState.EMERGENCY_STOP: "performing emergency stop"
        }
        
        self.current_explanation = "Driving normally"
    
    def plan(self, detected_objects, lane_info, predicted_trajectories, ego_vehicle):
        """
        Plan the next behavior.
        
        Args:
            detected_objects: List of detected objects.
            lane_info: Information about lanes.
            predicted_trajectories: Dictionary of predicted trajectories.
            ego_vehicle: Ego vehicle object.
            
        Returns:
            Tuple (behavior_command, explanation)
        """
        # Get ego vehicle state
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)  # km/h
        ego_location = ego_vehicle.get_location()
        ego_transform = ego_vehicle.get_transform()
        
        # If state counter is active, maintain current state
        if self.state_counter > 0:
            self.state_counter -= 1
            return self.current_state, self.current_explanation
        
        # Check for traffic lights
        traffic_light = self._detect_traffic_light(detected_objects)
        if traffic_light and traffic_light.get('state') == 'red':
            new_state = BehaviorState.STOP_FOR_TRAFFIC_LIGHT
            reason = f"Stopping for red traffic light at {traffic_light.get('distance', 'unknown')} meters"
            self._update_state(new_state, reason)
            return new_state, reason
        
        # Check for obstacles ahead
        obstacle = self._detect_obstacles_ahead(detected_objects, predicted_trajectories, ego_location, ego_transform)
        if obstacle:
            if obstacle.get('distance', float('inf')) < self.safe_distance / 2:
                # Very close obstacle - emergency stop
                new_state = BehaviorState.EMERGENCY_STOP
                reason = f"Emergency stopping for {obstacle.get('type', 'obstacle')} at {obstacle.get('distance', 'unknown')} meters"
                self._update_state(new_state, reason)
                return new_state, reason
            elif obstacle.get('distance', float('inf')) < self.safe_distance:
                # Close enough to follow or stop
                if obstacle.get('speed', 0) > 0:
                    # Moving obstacle - follow it
                    new_state = BehaviorState.FOLLOW_VEHICLE
                    reason = f"Following {obstacle.get('type', 'vehicle')} at {obstacle.get('distance', 'unknown')} meters, traveling at {obstacle.get('speed', 'unknown')} km/h"
                    self._update_state(new_state, reason)
                    return new_state, reason
                else:
                    # Stopped obstacle - wait
                    new_state = BehaviorState.STOP_FOR_OBSTACLE
                    reason = f"Stopping for stationary {obstacle.get('type', 'obstacle')} at {obstacle.get('distance', 'unknown')} meters"
                    self._update_state(new_state, reason)
                    return new_state, reason
        
        # Check for lane change opportunities if we're traveling at a reasonable speed
        if ego_speed > 15:  # Only consider lane changes above 15 km/h
            left_lane_viable = self._check_lane_change_viable('left', detected_objects, predicted_trajectories, lane_info)
            right_lane_viable = self._check_lane_change_viable('right', detected_objects, predicted_trajectories, lane_info)
            
            if left_lane_viable.get('viable', False) and obstacle:
                # Obstacle ahead and left lane is clear - prepare for left lane change
                new_state = BehaviorState.PREPARE_LANE_CHANGE_LEFT
                reason = f"Preparing to change to left lane to pass {obstacle.get('type', 'vehicle')}"
                self._update_state(new_state, reason)
                return new_state, reason
            
            if right_lane_viable.get('viable', False) and not obstacle and ego_speed > self.max_speed - 5:
                # No immediate obstacle, already at good speed, right lane clear - move right (traffic rule)
                new_state = BehaviorState.PREPARE_LANE_CHANGE_RIGHT
                reason = "Preparing to change to right lane (keep right rule)"
                self._update_state(new_state, reason)
                return new_state, reason
        
        # Execute lane changes that were prepared
        if self.current_state == BehaviorState.PREPARE_LANE_CHANGE_LEFT:
            new_state = BehaviorState.EXECUTE_LANE_CHANGE_LEFT
            reason = "Executing lane change to left lane"
            self._update_state(new_state, reason)
            return new_state, reason
        elif self.current_state == BehaviorState.PREPARE_LANE_CHANGE_RIGHT:
            new_state = BehaviorState.EXECUTE_LANE_CHANGE_RIGHT
            reason = "Executing lane change to right lane"
            self._update_state(new_state, reason)
            return new_state, reason
        
        # Default behavior: follow lane at target speed
        target_speed = min(self.max_speed, ego_speed + 2)  # Gradually increase speed if below max
        new_state = BehaviorState.FOLLOW_LANE
        reason = f"Following lane at target speed of {target_speed:.1f} km/h"
        self._update_state(new_state, reason)
        return new_state, reason
    
    def _update_state(self, new_state, explanation):
        """Update the current state and explanation."""
        # Only reset counter if state has changed
        if new_state != self.current_state:
            self.state_counter = self.state_ttl
        
        self.current_state = new_state
        self.current_explanation = explanation
        
        # Add to previous states for smoothing
        self.prev_states.append(new_state)
        if len(self.prev_states) > 5:
            self.prev_states.pop(0)
    
    def _detect_traffic_light(self, detected_objects):
        """Detect and analyze traffic lights."""
        for obj in detected_objects:
            if obj.get('class_id') == 9:  # Traffic light in COCO
                # Determine traffic light state (this would need a specialized detector in practice)
                # Here we just simulate different states
                state = 'green'  # Default to green
                bbox = obj['bbox']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                # Approximate distance (would use actual 3D position in practice)
                distance = 5000 / (area + 1)  # Simple inverse relationship
                
                return {
                    'id': obj['id'],
                    'state': state, 
                    'distance': distance
                }
        
        return None
    
    def _detect_obstacles_ahead(self, detected_objects, predicted_trajectories, ego_location, ego_transform):
        """Detect obstacles ahead of the ego vehicle."""
        obstacles = []
        
        # Calculate ego vehicle's forward vector
        forward_vector = ego_transform.get_forward_vector()
        
        for obj in detected_objects:
            if obj['class_id'] in [0, 1, 2, 3, 5, 7]:  # Person, bicycle, car, motorcycle, bus, truck
                # Get center point of the bbox
                bbox = obj['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Convert pixel position to 3D location (simplified)
                # In practice, you would use depth information or 3D detection
                obj_location_relative = carla.Location(x=30, y=(center_x - 640) / 10, z=0)
                
                # Check if the object is ahead (positive dot product with forward vector)
                direction = carla.Vector3D(obj_location_relative.x, obj_location_relative.y, 0)
                dot_product = forward_vector.x * direction.x + forward_vector.y * direction.y
                
                if dot_product > 0:  # Object is ahead
                    # Calculate distance (simplified)
                    distance = obj_location_relative.x
                    
                    # Determine object speed from trajectories
                    speed = 0
                    if obj['id'] in predicted_trajectories:
                        # Calculate speed from predicted trajectory
                        traj = predicted_trajectories[obj['id']]
                        if len(traj['points']) >= 2:
                            p1 = traj['points'][0]
                            p2 = traj['points'][-1]
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            displacement = math.sqrt(dx**2 + dy**2)
                            speed = displacement / traj['time_horizon'] * 3.6  # m/s to km/h
                    
                    obstacles.append({
                        'id': obj['id'],
                        'type': ['pedestrian', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'][obj['class_id']],
                        'distance': distance,
                        'speed': speed
                    })
        
        # Return the closest obstacle
        if obstacles:
            return min(obstacles, key=lambda x: x['distance'])
        else:
            return None
    
    def _check_lane_change_viable(self, direction, detected_objects, predicted_trajectories, lane_info):
        """Check if a lane change to the specified direction is viable."""
        # In a real implementation, this would:
        # 1. Check if there's a lane in the target direction
        # 2. Check if that lane is clear of obstacles
        # 3. Predict if any vehicles will occupy that space in the next few seconds
        
        # Simplified implementation - randomly decide based on detected objects
        if detected_objects:
            # If many objects detected, less likely to change lanes
            viable = len(detected_objects) < 3
        else:
            viable = True
        
        return {
            'viable': viable,
            'reason': f"Lane {'clear' if viable else 'occupied'}"
        } 