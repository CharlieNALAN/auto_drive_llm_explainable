#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from pathlib import Path
from .camera_geometry import CameraGeometry

# Try to import OpenVINO, use fallback if not available
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    print("Warning: OpenVINO not available. Using mock implementation.")
    OPENVINO_AVAILABLE = False

class OpenVINOLaneDetector:
    """OpenVINO-based lane detector migrated from the successful second project."""
    
    def __init__(self, cam_geom=None, model_path=None, device="CPU"):
        """
        Initialize OpenVINO lane detector.
        
        Args:
            cam_geom: Camera geometry object
            model_path: Path to the OpenVINO model
            device: Device to run inference on
        """
        # Initialize camera geometry
        if cam_geom is None:
            self.cg = CameraGeometry()
        else:
            self.cg = cam_geom
            
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        # Set default model path if not provided
        if model_path is None:
            # Try to find the model in various locations
            possible_paths = [
                'converted_model/lane_model.xml',
                'models/lane_model.xml',
                '../self-driving-carla-main/converted_model/lane_model.xml',
                '../../self-driving-carla-main/converted_model/lane_model.xml'
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                print("Warning: OpenVINO model not found. Using mock implementation.")
                self._use_mock = True
                return
        
        self.model_path = model_path
        self.device = device
        self._use_mock = False
        
        # Initialize OpenVINO model
        if OPENVINO_AVAILABLE:
            try:
                ie = Core()
                model_ir = ie.read_model(model_path)
                self.compiled_model_ir = ie.compile_model(model=model_ir, device_name=device)
                print(f"OpenVINO lane detector initialized with model: {model_path}")
            except Exception as e:
                print(f"Error loading OpenVINO model: {e}. Using mock implementation.")
                self._use_mock = True
        else:
            self._use_mock = True

    def read_imagefile_to_array(self, filename):
        """Read image file and convert to array."""
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_from_file(self, filename):
        """Detect lanes from image file."""
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        """
        Perform lane detection on image array.
        
        Args:
            img_array: Input image as numpy array (RGB format)
            
        Returns:
            Tuple of (background, left_lane, right_lane) probability maps
        """
        if self._use_mock:
            return self._mock_detect(img_array)
            
        try:
            # Preprocess image for OpenVINO model
            img_array = np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
            
            # Run inference
            output_layer_ir = next(iter(self.compiled_model_ir.outputs))
            model_output = self.compiled_model_ir([img_array])[output_layer_ir]
            
            # Extract lane probability maps
            background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
            return background, left, right
            
        except Exception as e:
            print(f"Error in OpenVINO inference: {e}. Using mock detection.")
            return self._mock_detect(img_array)

    def _mock_detect(self, img_array):
        """Mock detection for when OpenVINO is not available."""
        h, w = img_array.shape[:2]
        
        # Create simple mock lane maps
        background = np.zeros((h, w), dtype=np.float32)
        left = np.zeros((h, w), dtype=np.float32)
        right = np.zeros((h, w), dtype=np.float32)
        
        # Add some mock lane lines
        # Left lane (roughly 1/3 from left)
        left_x = w // 3
        for y in range(h//2, h):
            if 0 <= left_x < w:
                left[y, max(0, left_x-5):min(w, left_x+5)] = 0.8
        
        # Right lane (roughly 2/3 from left)
        right_x = 2 * w // 3
        for y in range(h//2, h):
            if 0 <= right_x < w:
                right[y, max(0, right_x-5):min(w, right_x+5)] = 0.8
                
        return background, left, right

    def detect_and_fit(self, img_array):
        """
        Detect lanes and fit polynomials.
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            Tuple of (left_poly, right_poly, left_prob_map, right_prob_map)
        """
        _, left, right = self.detect(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def fit_poly(self, probs):
        """
        Fit polynomial to lane probability map.
        
        Args:
            probs: Lane probability map
            
        Returns:
            Polynomial function
        """
        try:
            probs_flat = np.ravel(probs[self.cut_v:, :])
            mask = probs_flat > 0.3
            
            if np.sum(mask) < 10:  # Not enough points
                # Return a default straight line
                return np.poly1d([0, 0, 0, 0])  # y = 0 (straight ahead)
            
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
            return np.poly1d(coeffs)
            
        except Exception as e:
            print(f"Error in polynomial fitting: {e}")
            # Return a default straight line
            return np.poly1d([0, 0, 0, 0])

    def __call__(self, img):
        """Make the detector callable."""
        if isinstance(img, str):
            img = self.read_imagefile_to_array(img)
        return self.detect_and_fit(img)

    def get_lane_trajectory(self, img_array):
        """
        Get lane trajectory for path planning.
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            Trajectory as list of (x, y) points in vehicle coordinates
        """
        try:
            left_poly, right_poly, _, _ = self.detect_and_fit(img_array)
            
            # Generate trajectory points
            x_points = np.arange(-2, 60, 1.0)  # From -2m to 60m ahead
            
            # Calculate left and right lane positions
            left_y = left_poly(x_points)
            right_y = right_poly(x_points)
            
            # Take center line as trajectory
            center_y = -0.5 * (left_y + right_y)  # Note: negative because of coordinate system
            
            # Adjust x coordinates (camera is 0.5m in front of vehicle center)
            x_points += 0.5
            
            # Create trajectory as list of points
            trajectory = []
            for i in range(len(x_points)):
                trajectory.append((float(x_points[i]), float(center_y[i])))
                
            return trajectory
            
        except Exception as e:
            print(f"Error generating trajectory: {e}")
            # Return straight trajectory as fallback
            x_points = np.arange(0, 30, 2.0)
            trajectory = [(float(x), 0.0) for x in x_points]
            return trajectory 