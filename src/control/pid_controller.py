#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import carla

class PID:
    """PID controller for a single control variable."""
    
    def __init__(self, Kp, Ki, Kd, dt=0.05, min_output=-1.0, max_output=1.0):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: Time step
            min_output: Minimum output value
            max_output: Maximum output value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.min_output = min_output
        self.max_output = max_output
        
        # Initialize error terms
        self.prev_error = 0.0
        self.integral = 0.0
        
        # For diagnostics
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        
        # Timestamp for dt calculation (if dt is not fixed)
        self.prev_time = time.time()
    
    def reset(self):
        """Reset the controller."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
    
    def update(self, error, dt=None):
        """
        Update the controller.
        
        Args:
            error: Current error
            dt: Time step (optional, if not provided, calculated from last call)
            
        Returns:
            Control output
        """
        # Use provided dt or calculate from time difference
        if dt is None:
            current_time = time.time()
            dt = current_time - self.prev_time
            self.prev_time = current_time
        else:
            dt = self.dt
        
        # Calculate terms
        self.p_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.i_term = self.Ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        self.d_term = self.Kd * derivative
        
        # Update previous error
        self.prev_error = error
        
        # Calculate output
        output = self.p_term + self.i_term + self.d_term
        
        # Clamp output
        output = max(self.min_output, min(self.max_output, output))
        
        return output


class PIDController:
    """PID Controller for vehicle control."""
    
    def __init__(self, config):
        """
        Initialize PID controller.
        
        Args:
            config: Configuration dictionary for PID controller.
        """
        self.config = config
        
        # Extract PID parameters from config
        lateral_config = config.get('lateral', {})
        longitudinal_config = config.get('longitudinal', {})
        
        # Lateral (steering) PID - reduce Kp for smoother steering
        self.lateral_pid = PID(
            Kp=lateral_config.get('Kp', 1.0),  # Reduced from 1.5 for less aggressive steering
            Ki=lateral_config.get('Ki', 0.0),
            Kd=lateral_config.get('Kd', 0.05),  # Added small derivative term for stability
            min_output=-1.0,
            max_output=1.0
        )
        
        # Longitudinal (throttle/brake) PID - reduce Kp for gentler acceleration
        self.longitudinal_pid = PID(
            Kp=longitudinal_config.get('Kp', 0.7),  # Reduced from 1.0 for gentler throttle
            Ki=longitudinal_config.get('Ki', 0.05),  # Reduced from 0.1 for less integral windup
            Kd=longitudinal_config.get('Kd', 0.02),  # Added small derivative term
            min_output=-1.0,  # -1 for full brake
            max_output=0.7    # Reduced from 1.0 to limit max throttle
        )
        
        # Safety parameters
        self.max_steering_angle = config.get('max_steering_angle', 0.5)  # Reduced from 0.7 for less aggressive turning
        self.max_acceleration = config.get('max_acceleration', 1.5)  # Reduced from 2.0
        self.max_deceleration = config.get('max_deceleration', 3.0)  # Reduced from 5.0
        
        # Emergency maneuver parameters
        self.emergency_brake_threshold = config.get('emergency_brake_threshold', 5.0)  # meters
        self.collision_avoidance_steering = config.get('collision_avoidance_steering', 0.6)  # Reduced from 0.8
        
        # Previous control values for smoothing
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        
        # Counters for emergency maneuvers
        self.emergency_counter = 0
    
    def control(self, trajectory, ego_vehicle):
        """
        Calculate control commands based on the planned trajectory.
        
        Args:
            trajectory: Planned trajectory.
            ego_vehicle: Ego vehicle object.
            
        Returns:
            carla.VehicleControl object with throttle, brake, and steering values.
        """
        # Extract vehicle state
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)  # km/h
        
        # Create default control (no action)
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.0
        control.steer = 0.0
        
        # Check if trajectory exists and has points
        if not trajectory or 'points' not in trajectory or not trajectory['points']:
            # Default to very small throttle if no trajectory to prevent stalling
            control.throttle = 0.05  # Reduced from 0.1
            return control
        
        # Get trajectory information
        waypoints = trajectory['points']
        speed_profile = trajectory.get('speed_profile', [ego_speed] * len(waypoints))
        behavior = trajectory.get('behavior', None)
        
        # Determine if we're in an emergency situation
        is_emergency = behavior == "emergency_stop" if behavior else False
        
        # Update emergency counter
        if is_emergency:
            self.emergency_counter = 15  # Reduced from 20 to make recovery faster
        elif self.emergency_counter > 0:
            self.emergency_counter -= 1
        
        # Get the target waypoint with lookahead based on speed
        # Higher speed requires looking further ahead for stability
        lookahead_factor = min(3, max(1, int(ego_speed / 20.0)))  # Increased from 15.0 to look further ahead
        target_idx = min(lookahead_factor, len(waypoints) - 1)
        target_point = waypoints[target_idx]
        
        # For emergency situations or very low speeds, look at closer waypoints
        if is_emergency or ego_speed < 5.0:
            target_idx = 0
            target_point = waypoints[0]
        
        # Get target speed (with more aggressive deceleration for emergencies)
        target_speed = speed_profile[target_idx]
        if is_emergency or self.emergency_counter > 0:
            target_speed = 0.0  # Hard stop
        
        # For very low speeds, add a minimum target speed to prevent stalling
        # but keep it very low
        if target_speed < 2.0 and not is_emergency and self.emergency_counter == 0:
            target_speed = 2.0
        
        # Cap the target speed to reduce excessive acceleration
        if not is_emergency:
            # Limit acceleration by capping target speed relative to current speed
            target_speed = min(target_speed, ego_speed + 5.0)  # Limit acceleration to ~5 km/h per second
        
        # Calculate lateral error (cross-track error)
        lateral_error = self._calculate_lateral_error(ego_location, ego_rotation, target_point)
        
        # Calculate longitudinal error (speed error)
        longitudinal_error = target_speed - ego_speed  # km/h
        
        # Apply more aggressive control for emergencies
        if is_emergency or self.emergency_counter > 0:
            # Emergency steering to avoid collision if lateral error is significant
            if abs(lateral_error) > 1.0:
                steering = self.collision_avoidance_steering * (lateral_error / abs(lateral_error))
            else:
                steering = self.lateral_pid.update(lateral_error)
            
            # Hard braking, but not too extreme
            throttle_brake = -0.7  # Reduced from -0.8 for smoother stops
        else:
            # Normal steering control with PID
            steering = self.lateral_pid.update(lateral_error)
            
            # Add predictive steering based on path curvature, but with smaller weight
            if len(waypoints) > target_idx + 1:
                next_point = waypoints[target_idx + 1]
                path_angle = math.atan2(next_point[1] - target_point[1], next_point[0] - target_point[0])
                vehicle_angle = math.radians(ego_rotation.yaw)
                angle_diff = self._normalize_angle(path_angle - vehicle_angle)
                steering += 0.15 * angle_diff  # Reduced from 0.2 for even smoother steering
            
            # Normal throttle/brake control with PID
            throttle_brake = self.longitudinal_pid.update(longitudinal_error)
            
            # Add a minimum throttle at low speeds to prevent stalling, but keep it small
            if ego_speed < 5.0 and throttle_brake >= 0:
                throttle_brake = max(throttle_brake, 0.1)  # Reduced from 0.2
                
            # Limit maximum throttle to avoid jerky acceleration
            if throttle_brake > 0:
                max_throttle = min(0.6, 0.3 + (ego_speed / 50.0))  # Progressive throttle limit based on speed
                throttle_brake = min(throttle_brake, max_throttle)
        
        # Apply steering limits and smoothing
        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
        
        # Apply stronger steering smoothing to reduce oscillations
        smoothing_factor = 0.25 if is_emergency else 0.3  # Reduced from 0.3/0.4 for more smoothing
        steering = steering * smoothing_factor + self.prev_steering * (1 - smoothing_factor)
        
        # Apply second-order damping to reduce oscillation
        if abs(steering - self.prev_steering) > 0.02:  # If steering change is significant
            # Dampen the change to reduce oscillation
            steering = 0.7 * steering + 0.3 * self.prev_steering
            
        self.prev_steering = steering
        
        # Apply throttle and brake with smoothing for non-emergency situations
        if not is_emergency and self.emergency_counter == 0:
            if throttle_brake >= 0:
                # Positive value: throttle
                throttle = throttle_brake
                brake = 0.0
                
                # Stronger throttle smoothing to prevent jerky acceleration
                smoothing_factor = 0.25  # Reduced from 0.4 for stronger smoothing
                throttle = throttle * smoothing_factor + self.prev_throttle * (1 - smoothing_factor)
                self.prev_throttle = throttle
                self.prev_brake = 0.0
            else:
                # Negative value: brake
                throttle = 0.0
                brake = -throttle_brake
                
                # Smooth brake changes (less smoothing for more responsive braking)
                smoothing_factor = 0.5  # Reduced from 0.6 for more responsive braking
                brake = brake * smoothing_factor + self.prev_brake * (1 - smoothing_factor)
                self.prev_throttle = 0.0
                self.prev_brake = brake
        else:
            # Emergency situation - still use some smoothing for stability
            if throttle_brake >= 0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake
            
            # Even in emergency, add some smoothing
            brake = 0.7 * brake + 0.3 * self.prev_brake
            
            self.prev_throttle = throttle
            self.prev_brake = brake
        
        # Set control values
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steering)
        
        # Store current control values for explanation
        self.current_control = {
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'lateral_error': lateral_error,
            'speed_error': longitudinal_error,
            'target_speed': target_speed,
            'is_emergency': is_emergency or self.emergency_counter > 0
        }
        
        return control
    
    def _calculate_lateral_error(self, ego_location, ego_rotation, target_point):
        """
        Calculate lateral error (cross-track error) to the target point.
        
        Args:
            ego_location: Current vehicle location.
            ego_rotation: Current vehicle rotation.
            target_point: Target point as (x, y) tuple.
            
        Returns:
            Lateral error in meters.
        """
        # Convert ego location to numpy array
        ego_pos = np.array([ego_location.x, ego_location.y])
        
        # Convert target point to numpy array
        target_pos = np.array([target_point[0], target_point[1]])
        
        # Calculate vector from vehicle to target
        vec_to_target = target_pos - ego_pos
        
        # Calculate vehicle's forward vector
        forward = ego_rotation.get_forward_vector()
        forward_vec = np.array([forward.x, forward.y])
        forward_vec = forward_vec / np.linalg.norm(forward_vec)
        
        # Calculate vehicle's right vector
        right_vec = np.array([-forward_vec[1], forward_vec[0]])
        
        # Calculate lateral error (projection onto right vector)
        lateral_error = np.dot(vec_to_target, right_vec)
        
        return lateral_error
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle 