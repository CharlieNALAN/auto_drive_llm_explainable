#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class KalmanTracker:
    """
    Kalman filter-based tracker for a single object.
    
    Uses a constant velocity model to predict future positions.
    """
    
    def __init__(self, bbox, dt=0.1, id=None):
        """
        Initialize a Kalman tracker.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            dt: Time step
            id: Object ID
        """
        self.id = id
        self.dt = dt
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        
        # Compute center coordinates and dimensions from bbox
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Create Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State vector [x, y, width, height, vx, vy, vw, vh]
        self.kf.x = np.array([x_center, y_center, width, height, 0, 0, 0, 0])
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.eye(8)
        self.kf.F[0, 4] = dt  # x += vx * dt
        self.kf.F[1, 5] = dt  # y += vy * dt
        self.kf.F[2, 6] = dt  # width += vw * dt
        self.kf.F[3, 7] = dt  # height += vh * dt
        
        # Measurement matrix (we only observe position and size)
        self.kf.H = np.zeros((4, 8))
        self.kf.H[0, 0] = 1.0  # x
        self.kf.H[1, 1] = 1.0  # y
        self.kf.H[2, 2] = 1.0  # width
        self.kf.H[3, 3] = 1.0  # height
        
        # Measurement noise covariance
        self.kf.R = np.diag([10, 10, 10, 10])  # Uncertainty in measurements
        
        # Process noise covariance
        q_pos = Q_discrete_white_noise(dim=2, dt=dt, var=30)
        q_size = Q_discrete_white_noise(dim=2, dt=dt, var=10)
        self.kf.Q = block_diag(q_pos, q_size, q_pos, q_size)
        
        # Initial state covariance
        self.kf.P = np.diag([100, 100, 25, 25, 50, 50, 25, 25])
    
    def update(self, bbox):
        """
        Update the Kalman filter with a new measurement.
        
        Args:
            bbox: New bounding box [x1, y1, x2, y2]
        """
        # Compute center coordinates and dimensions from bbox
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Measurement
        z = np.array([x_center, y_center, width, height])
        
        # Update Kalman filter
        self.kf.update(z)
        
        # Update tracking state
        self.time_since_update = 0
        self.hit_streak += 1
        self.age += 1
    
    def predict(self):
        """
        Predict the state for the next time step.
        
        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Predict new state
        self.kf.predict()
        
        # Update tracking state
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        
        # Convert center coordinates and dimensions to bbox
        x_center = self.kf.x[0]
        y_center = self.kf.x[1]
        width = self.kf.x[2]
        height = self.kf.x[3]
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return [x1, y1, x2, y2]
    
    def predict_future(self, time_horizon, steps):
        """
        Predict future trajectory over multiple time steps.
        
        Args:
            time_horizon: Time horizon to predict into the future (seconds)
            steps: Number of steps to predict
            
        Returns:
            List of predicted positions (x, y) for each time step
        """
        # Save current state
        current_x = self.kf.x.copy()
        current_P = self.kf.P.copy()
        
        # Time step size
        dt_future = time_horizon / steps
        
        # Update transition matrix with new dt
        F_future = np.eye(8)
        F_future[0, 4] = dt_future
        F_future[1, 5] = dt_future
        F_future[2, 6] = dt_future
        F_future[3, 7] = dt_future
        
        # Predict future positions
        future_positions = []
        x = current_x.copy()
        
        for _ in range(steps):
            # Predict next state using future transition matrix
            x = np.dot(F_future, x)
            
            # Extract position (center coordinates)
            position = (x[0], x[1])
            future_positions.append(position)
        
        # Restore saved state
        self.kf.x = current_x
        self.kf.P = current_P
        
        return future_positions


class TrajectoryPredictor:
    """
    Class to predict the future trajectories of objects.
    """
    
    def __init__(self, config):
        """
        Initialize the trajectory predictor.
        
        Args:
            config: Configuration dictionary for trajectory prediction.
        """
        self.config = config
        self.method = config.get('method', 'kalman')
        self.time_horizon = config.get('time_horizon', 3.0)  # seconds
        self.dt = config.get('dt', 0.1)  # seconds
        self.steps = int(self.time_horizon / self.dt)
        
        # Dictionary to store trackers for each object
        self.trackers = {}
    
    def predict(self, detected_objects):
        """
        Predict trajectories for detected objects.
        
        Args:
            detected_objects: List of detected objects from the perception module.
            
        Returns:
            Dictionary mapping object IDs to predicted trajectories.
        """
        if self.method == 'kalman':
            return self._predict_kalman(detected_objects)
        else:
            raise ValueError(f"Unknown prediction method: {self.method}")
    
    def _predict_kalman(self, detected_objects):
        """
        Predict trajectories using Kalman filters.
        
        Args:
            detected_objects: List of detected objects from the perception module.
            
        Returns:
            Dictionary mapping object IDs to predicted trajectories.
        """
        # Update existing trackers and create new ones
        current_ids = set()
        
        for obj in detected_objects:
            obj_id = obj['id']
            current_ids.add(obj_id)
            
            if obj_id in self.trackers:
                # Update existing tracker
                self.trackers[obj_id].update(obj['bbox'])
            else:
                # Create new tracker
                self.trackers[obj_id] = KalmanTracker(obj['bbox'], dt=self.dt, id=obj_id)
        
        # Remove trackers for objects that are no longer detected
        trackers_to_remove = []
        for tracker_id in self.trackers:
            if tracker_id not in current_ids:
                trackers_to_remove.append(tracker_id)
        
        for tracker_id in trackers_to_remove:
            del self.trackers[tracker_id]
        
        # Predict trajectories for each tracker
        trajectories = {}
        
        for tracker_id, tracker in self.trackers.items():
            # Get current state (for current position)
            current_bbox = tracker.predict()
            
            # Predict future trajectory
            future_positions = tracker.predict_future(self.time_horizon, self.steps)
            
            # Create trajectory dictionary
            trajectory = {
                'current_bbox': current_bbox,
                'points': future_positions,
                'time_horizon': self.time_horizon,
                'step_size': self.dt
            }
            
            trajectories[tracker_id] = trajectory
        
        return trajectories 