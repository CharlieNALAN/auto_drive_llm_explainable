#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ..utils.geometry import get_target_point

class PurePursuit:
    """Pure pursuit path tracking controller."""
    
    def __init__(self, K_dd=0.4, wheel_base=2.65, waypoint_shift=1.4):
        self.K_dd = K_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift
    
    def get_control(self, waypoints, speed):
        """
        Calculate steering control for pure pursuit.
        
        Args:
            waypoints: Array of waypoint coordinates
            speed: Current vehicle speed
            
        Returns:
            Steering angle in radians
        """
        waypoints[:,0] += self.waypoint_shift
        look_ahead_distance = np.clip(self.K_dd * speed, 3, 20)
        track_point = get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            waypoints[:,0] -= self.waypoint_shift
            return 0
        alpha = np.arctan2(track_point[1], track_point[0])
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)
        waypoints[:,0] -= self.waypoint_shift
        return steer 