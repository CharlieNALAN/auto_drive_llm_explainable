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
        
        # Lateral (steering) PID
        self.lateral_pid = PID(
            Kp=lateral_config.get('Kp', 1.5),
            Ki=lateral_config.get('Ki', 0.0),
            Kd=lateral_config.get('Kd', 0.0),
            min_output=-1.0,
            max_output=1.0
        )
        
        # Longitudinal (throttle/brake) PID
        self.longitudinal_pid = PID(
            Kp=longitudinal_config.get('Kp', 1.0),
            Ki=longitudinal_config.get('Ki', 0.1),
            Kd=longitudinal_config.get('Kd', 0.0),
            min_output=-1.0,  # -1 for full brake
            max_output=1.0    # +1 for full throttle
        )
        
        # Safety parameters
        self.max_steering_angle = config.get('max_steering_angle', 0.7)  # radians
        self.max_acceleration = config.get('max_acceleration', 2.0)  # m/s^2
        self.max_deceleration = config.get('max_deceleration', 5.0)  # m/s^2
    
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
            return control
        
        # Get target point and speed (from the first few waypoints)
        waypoints = trajectory['points']
        speed_profile = trajectory.get('speed_profile', [ego_speed] * len(waypoints))
        
        # Get the target waypoint (a bit ahead of the first one for smoother control)
        target_idx = min(2, len(waypoints) - 1)
        target_point = waypoints[target_idx]
        target_speed = speed_profile[target_idx]
        
        # Calculate lateral error (cross-track error)
        lateral_error = self._calculate_lateral_error(ego_location, ego_rotation, target_point)
        
        # Calculate longitudinal error (speed error)
        longitudinal_error = target_speed - ego_speed  # km/h
        
        # Calculate steering control
        steering = self.lateral_pid.update(lateral_error)
        steering = np.clip(steering, -self.max_steering_angle, self.max_steering_angle)
        
        # Calculate throttle/brake control
        throttle_brake = self.longitudinal_pid.update(longitudinal_error)
        
        # Apply throttle and brake based on the longitudinal control value
        if throttle_brake >= 0:
            # Positive value: throttle
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            # Negative value: brake
            control.throttle = 0.0
            control.brake = -throttle_brake
        
        # Apply steering
        control.steer = steering
        
        # Store current control values for explanation
        self.current_control = {
            'throttle': control.throttle,
            'brake': control.brake,
            'steer': control.steer,
            'lateral_error': lateral_error,
            'speed_error': longitudinal_error,
            'target_speed': target_speed
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