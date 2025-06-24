#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

def get_target_point(lookahead, polyline):
    """
    Get target point for pure pursuit controller.
    
    Args:
        lookahead: Lookahead distance
        polyline: List of (x, y) trajectory points
        
    Returns:
        Target point (x, y) or None if no intersection found
    """
    intersections = []
    for j in range(len(polyline)-1):
        pt1 = polyline[j]
        pt2 = polyline[j+1]
        intersections += circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
    
    # Filter intersections to only include points ahead of the vehicle
    filtered = [p for p in intersections if p[0] > 0]
    return filtered[0] if len(filtered) > 0 else None

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True):
    """
    Find intersection points between a circle and a line segment.
    
    Args:
        circle_center: Center of the circle (x, y)
        circle_radius: Radius of the circle
        pt1: First point of line segment (x, y)
        pt2: Second point of line segment (x, y)
        full_line: If True, treat as infinite line; if False, treat as segment
        
    Returns:
        List of intersection points
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** 0.5
    
    if dr == 0:
        return []
    
    big_d = x1 * y2 - x2 * y1
    discriminant = (circle_radius ** 2) * (dr ** 2) - (big_d ** 2)

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** 0.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** 0.5) / dr ** 2)
            for sign in ((1, -1) if discriminant > 0 else (1,))
        ]  # This makes sure the order along the segment is correct
        
        if not full_line:  # If only considering the segment, filter the intersections
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        
        return intersections

class PurePursuitController:
    """Pure pursuit controller for lateral control."""
    
    def __init__(self, K_dd=0.4, wheel_base=2.65, waypoint_shift=1.4):
        """
        Initialize pure pursuit controller.
        
        Args:
            K_dd: Look-ahead distance coefficient
            wheel_base: Vehicle wheelbase in meters
            waypoint_shift: Waypoint shift for coordinate adjustment
        """
        self.K_dd = K_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift
    
    def get_control(self, waypoints, speed):
        """
        Calculate steering control using pure pursuit algorithm.
        
        Args:
            waypoints: Array of waypoints [(x, y), ...]
            speed: Current vehicle speed in m/s
            
        Returns:
            Steering angle in radians
        """
        if len(waypoints) < 2:
            return 0.0
        
        # Convert to numpy array for easier manipulation
        waypoints = np.array(waypoints)
        
        # Transform x coordinates (coordinate origin is in rear wheel)
        waypoints[:, 0] += self.waypoint_shift
        
        # Calculate look-ahead distance
        look_ahead_distance = np.clip(self.K_dd * speed, 3, 20)
        
        # Get target point
        track_point = get_target_point(look_ahead_distance, waypoints.tolist())
        
        if track_point is None:
            # Undo transform
            waypoints[:, 0] -= self.waypoint_shift
            return 0.0
        
        # Calculate steering angle
        alpha = np.arctan2(track_point[1], track_point[0])
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)
        
        # Undo transform to waypoints
        waypoints[:, 0] -= self.waypoint_shift
        
        return steer

class PIDController:
    """PID controller for longitudinal control."""
    
    def __init__(self, Kp, Ki, Kd, set_point):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain  
            Kd: Derivative gain
            set_point: Desired setpoint
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.int_term = 0
        self.derivative_term = 0
        self.last_error = None
    
    def get_control(self, measurement, dt):
        """
        Calculate PID control output.
        
        Args:
            measurement: Current measurement
            dt: Time step
            
        Returns:
            Control output
        """
        error = self.set_point - measurement
        self.int_term += error * self.Ki * dt
        
        if self.last_error is not None:
            self.derivative_term = (error - self.last_error) / dt * self.Kd
            
        self.last_error = error
        return self.Kp * error + self.int_term + self.derivative_term

class PurePursuitPlusPIDController:
    """Combined pure pursuit and PID controller."""
    
    def __init__(self, Kp=2.0, Ki=0.0, Kd=0.0, K_dd=0.4):
        """
        Initialize combined controller.
        
        Args:
            Kp: PID proportional gain
            Ki: PID integral gain
            Kd: PID derivative gain  
            K_dd: Pure pursuit look-ahead coefficient
        """
        self.pure_pursuit = PurePursuitController(K_dd=K_dd)
        self.pid = PIDController(Kp, Ki, Kd, 0)
    
    def get_control(self, waypoints, speed, desired_speed, dt):
        """
        Calculate combined steering and throttle control.
        
        Args:
            waypoints: Array of waypoints
            speed: Current speed
            desired_speed: Desired speed
            dt: Time step
            
        Returns:
            Tuple of (throttle/brake, steering)
        """
        # Update PID setpoint
        self.pid.set_point = desired_speed
        
        # Get longitudinal control (throttle/brake)
        acceleration = self.pid.get_control(speed, dt)
        
        # Get lateral control (steering)
        steering = self.pure_pursuit.get_control(waypoints, speed)
        
        return acceleration, steering 