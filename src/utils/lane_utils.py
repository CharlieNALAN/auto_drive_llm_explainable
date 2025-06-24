#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def carla_img_to_array(image):
    """
    Convert CARLA image to numpy array.
    
    Args:
        image: CARLA image object
        
    Returns:
        Numpy array in RGB format
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    # Remove alpha channel and convert BGRA to RGB
    array = array[:, :, :3][:, :, ::-1]
    return array

def get_curvature(trajectory):
    """
    Calculate maximum curvature of a trajectory.
    
    Args:
        trajectory: List of (x, y) points
        
    Returns:
        Maximum curvature value
    """
    if len(trajectory) < 3:
        return 0.0
    
    trajectory_array = np.array(trajectory)
    x, y = trajectory_array[:, 0], trajectory_array[:, 1]
    
    # Calculate first and second derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Calculate curvature at each point
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    # Handle division by zero
    curvature = np.nan_to_num(curvature)
    
    return np.max(curvature)

def trajectory_to_carla_waypoints(trajectory, vehicle_transform):
    """
    Convert trajectory points to CARLA world coordinates.
    
    Args:
        trajectory: List of (x, y) points in vehicle coordinates
        vehicle_transform: CARLA Transform of the vehicle
        
    Returns:
        List of CARLA Location objects
    """
    import carla
    
    waypoints = []
    
    # Get vehicle transformation matrix
    transform = vehicle_transform
    forward_vector = transform.get_forward_vector()
    right_vector = transform.get_right_vector()
    
    for x, y in trajectory:
        # Convert from vehicle coordinates to world coordinates
        world_x = transform.location.x + x * forward_vector.x + y * right_vector.x
        world_y = transform.location.y + x * forward_vector.y + y * right_vector.y
        world_z = transform.location.z
        
        waypoints.append(carla.Location(x=world_x, y=world_y, z=world_z))
    
    return waypoints

def smooth_trajectory(trajectory, alpha=0.3):
    """
    Apply exponential smoothing to trajectory.
    
    Args:
        trajectory: List of (x, y) points
        alpha: Smoothing factor (0 = no smoothing, 1 = no memory)
        
    Returns:
        Smoothed trajectory
    """
    if len(trajectory) < 2:
        return trajectory
    
    smoothed = [trajectory[0]]  # First point stays the same
    
    for i in range(1, len(trajectory)):
        smooth_x = alpha * trajectory[i][0] + (1 - alpha) * smoothed[i-1][0]
        smooth_y = alpha * trajectory[i][1] + (1 - alpha) * smoothed[i-1][1]
        smoothed.append((smooth_x, smooth_y))
    
    return smoothed

def validate_trajectory(trajectory, max_steering_angle=0.5, max_distance=50.0):
    """
    Validate trajectory for safety and feasibility.
    
    Args:
        trajectory: List of (x, y) points
        max_steering_angle: Maximum allowed steering angle (radians)
        max_distance: Maximum trajectory distance (meters)
        
    Returns:
        Boolean indicating if trajectory is valid
    """
    if not trajectory or len(trajectory) < 2:
        return False
    
    # Check if trajectory points are reasonable
    for x, y in trajectory:
        if abs(y) > 10.0:  # Lateral deviation too large
            return False
        if x < 0 or x > max_distance:  # Longitudinal bounds
            return False
    
    # Check for reasonable curvature
    curvature = get_curvature(trajectory)
    if curvature > max_steering_angle:
        return False
    
    return True

def trajectory_to_path_points(trajectory, num_points=20):
    """
    Convert trajectory to evenly spaced path points.
    
    Args:
        trajectory: List of (x, y) points
        num_points: Number of output points
        
    Returns:
        List of evenly spaced points
    """
    if len(trajectory) < 2:
        return trajectory
    
    trajectory_array = np.array(trajectory)
    
    # Calculate cumulative distance along trajectory
    distances = np.cumsum(np.sqrt(np.sum(np.diff(trajectory_array, axis=0)**2, axis=1)))
    distances = np.insert(distances, 0, 0)  # Insert 0 at the beginning
    
    # Create evenly spaced distance points
    total_distance = distances[-1]
    even_distances = np.linspace(0, total_distance, num_points)
    
    # Interpolate x and y coordinates
    x_interp = np.interp(even_distances, distances, trajectory_array[:, 0])
    y_interp = np.interp(even_distances, distances, trajectory_array[:, 1])
    
    return list(zip(x_interp, y_interp)) 