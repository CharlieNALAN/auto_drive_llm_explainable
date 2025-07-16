#!/usr/bin/env python
# -*- coding: utf-8 -*-

import carla
import numpy as np
from .pure_pursuit_controller import PurePursuit
from .pid_controller import PIDController

class PurePursuitPlusPID:
    """Combined pure pursuit and PID controller."""
    
    def __init__(self, pure_pursuit=None, pid=None):
        self.pure_pursuit = pure_pursuit or PurePursuit()
        self.pid = pid or PIDController(2, 0, 0, 0)

    def get_control(self, waypoints, speed, desired_speed, dt):
        """
        Calculate combined control output.
        
        Args:
            waypoints: Array of waypoint coordinates
            speed: Current vehicle speed
            desired_speed: Target speed
            dt: Time step
            
        Returns:
            Tuple of (acceleration, steering)
        """
        self.pid.set_point = desired_speed
        a = self.pid.get_control(speed, dt)
        steer = self.pure_pursuit.get_control(waypoints, speed)
        return a, steer

def send_control(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    """
    Send control commands to the vehicle.
    
    Args:
        vehicle: CARLA vehicle actor
        throttle: Throttle value (0-1)
        steer: Steering angle (-1 to 1)
        brake: Brake value (0-1)
        hand_brake: Hand brake flag
        reverse: Reverse flag
    """
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control) 