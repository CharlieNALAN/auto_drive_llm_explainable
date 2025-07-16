#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from ..utils.geometry import carla_vec_to_np_array, carla_img_to_array

def get_trajectory_from_lane_detector(lane_detector, image):
    """
    Generate trajectory from lane detector output.
    
    Args:
        lane_detector: OpenVINO lane detector instance
        image: CARLA image
        
    Returns:
        Tuple of (trajectory, visualization_image)
    """
    image_arr = carla_img_to_array(image)
    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    
    # Get original image, convert RGB to BGR, and resize it
    img = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (600, 400))
    
    # Get lane detection results and overlay them on the original image
    lane_img = img_left + img_right
    lane_img = cv2.normalize(lane_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    lane_img = lane_img.astype(np.uint8)
    lane_img_resized = cv2.resize(lane_img, (600, 400))
    
    # Convert lane detection to 3-channel for overlay
    lane_img_colored = cv2.cvtColor(lane_img_resized, cv2.COLOR_GRAY2BGR)
    
    # Create overlay: original image + lane lines
    # Make lane lines more visible by using green color
    lane_mask = lane_img_resized > 50  # Threshold for lane pixels
    img[lane_mask] = [0, 255, 0]  # Green color for lane lines (BGR format)
    
    x = np.arange(-2, 60, 1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    x += 0.5
    trajectory = np.stack((x,y)).T
    return trajectory, img

def get_trajectory_from_map(CARLA_map, vehicle):
    """
    Generate trajectory from CARLA map waypoints.
    
    Args:
        CARLA_map: CARLA map instance
        vehicle: CARLA vehicle actor
        
    Returns:
        Trajectory array
    """
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())
    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:,:2]
    return trajectory 