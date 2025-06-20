#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import carla
import cv2

def carla_img_to_array(image):
    """Convert CARLA image to numpy array."""
    # Check if the input is already a numpy array
    if isinstance(image, np.ndarray):
        return image
    
    # Handle CARLA image object
    if hasattr(image, 'raw_data'):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        array = array[:, :, ::-1]  # Convert BGR to RGB
        return array
    else:
        # If it's already an array or unknown format, return as is
        return np.array(image)

def carla_vec_to_np_array(carla_vector):
    """Convert CARLA Vector3D to numpy array."""
    return np.array([carla_vector.x, carla_vector.y, carla_vector.z])

def get_trajectory_from_lane_detector(lane_detector, image):
    """
    Get trajectory from lane detector using the proven approach from project 2.
    
    Args:
        lane_detector: Lane detector instance
        image: CARLA image
        
    Returns:
        Tuple (trajectory, visualization_image)
    """
    # Convert CARLA image to array
    image_arr = carla_img_to_array(image)

    try:
        # Use the lane detector
        poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
        
        # Create visualization image
        img = img_left + img_right if img_left is not None and img_right is not None else None
        if img is not None:
            img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img = img.astype(np.uint8)
            img = cv2.resize(img, (600, 400))
        else:
            # Create a basic visualization
            img = cv2.resize(image_arr, (600, 400))
        
        # Generate trajectory from lane polynomials
        # Note: multiply with -0.5 instead of 0.5 because of coordinate system differences
        x = np.arange(-2, 60, 1.0)
        y = -0.5 * (poly_left(x) + poly_right(x))
        
        # Adjust x coordinates (camera is 0.5m in front of vehicle center)
        x += 0.5
        trajectory = np.stack((x, y)).T
        
        return trajectory, img
        
    except Exception as e:
        print(f"Lane detection failed: {e}")
        raise e

def get_trajectory_from_map(carla_map, vehicle):
    """
    Get trajectory from CARLA map as fallback when lane detection fails.
    
    Args:
        carla_map: CARLA map object
        vehicle: Vehicle actor
        
    Returns:
        Trajectory as numpy array
    """
    # Get 20 waypoints each 1m apart
    waypoint = carla_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    # Transform waypoints to vehicle reference frame
    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())
    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:, :2]
    
    return trajectory

def send_control(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    """
    Send control commands to the vehicle.
    
    Args:
        vehicle: CARLA vehicle actor
        throttle: Throttle value [0.0, 1.0]
        steer: Steering value [-1.0, 1.0]
        brake: Brake value [0.0, 1.0]
        hand_brake: Hand brake boolean
        reverse: Reverse boolean
    """
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

def dist_point_linestring(point, linestring):
    """
    Calculate distance from a point to a linestring.
    
    Args:
        point: Point as numpy array [x, y]
        linestring: Array of points forming a line
        
    Returns:
        Minimum distance from point to linestring
    """
    if len(linestring) < 2:
        return float('inf')
    
    min_distance = float('inf')
    
    for i in range(len(linestring) - 1):
        # Get line segment
        p1 = np.array(linestring[i])
        p2 = np.array(linestring[i + 1])
        
        # Calculate distance from point to line segment
        segment_vec = p2 - p1
        point_vec = point - p1
        
        # Project point onto line segment
        if np.allclose(segment_vec, 0):
            # Degenerate line segment
            distance = np.linalg.norm(point_vec)
        else:
            t = np.dot(point_vec, segment_vec) / np.dot(segment_vec, segment_vec)
            t = np.clip(t, 0.0, 1.0)  # Clamp to line segment
            
            # Calculate closest point on segment
            closest_point = p1 + t * segment_vec
            distance = np.linalg.norm(point - closest_point)
        
        min_distance = min(min_distance, distance)
    
    return min_distance 