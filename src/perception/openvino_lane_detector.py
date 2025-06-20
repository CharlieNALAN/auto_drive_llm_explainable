#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenVINO-based lane detector copied from working project.
"""
from .camera_geometry import CameraGeometry
from openvino.runtime import Core
import numpy as np
import cv2
from pathlib import Path

class OpenVINOLaneDetector:
    """Lane detector using OpenVINO runtime - copied from working project."""
    
    def __init__(self, cam_geom=None, model_path=None, device="CPU"):
        """
        Initialize OpenVINO lane detector.
        
        Args:
            cam_geom: Camera geometry object
            model_path: Path to OpenVINO XML model file
            device: Inference device ("CPU", "GPU", etc.)
        """
        self.cg = cam_geom if cam_geom else CameraGeometry()
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        # Handle model path
        if model_path is None:
            potential_paths = [
                '../self-driving-carla-main/converted_model/lane_model.xml',
                'converted_model/lane_model.xml',
                'models/lane_model.xml',
                "D:\TPG sem2\Project\self-driving-carla-main\converted_model\lane_model.xml"
            ]
            for path in potential_paths:
                if Path(path).exists():
                    model_path = path
                    break
        
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"OpenVINO model not found. Tried paths: {potential_paths}")
        
        # Initialize OpenVINO
        ie = Core()
        model_ir = ie.read_model(model_path)
        self.compiled_model_ir = ie.compile_model(model=model_ir, device_name=device)
        
        print(f"✓ OpenVINO lane detector loaded from: {model_path}")

    def read_imagefile_to_array(self, filename):
        """Read image file to numpy array."""
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_from_file(self, filename):
        """Detect lanes from image file."""
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)

    def detect(self, img_array):
        """
        Detect lanes and return trajectory information (compatible interface).
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            Dictionary containing lane information compatible with original interface
        """
        try:
            # Use OpenVINO deep learning approach
            left_poly, right_poly, left_mask, right_mask = self.detect_and_fit(img_array)
            trajectory = self._generate_trajectory_from_lanes(left_poly, right_poly)
            
            return {
                'lanes': [trajectory] if trajectory else [],
                'trajectory': trajectory,
                'left_poly': left_poly,
                'right_poly': right_poly,
                'left_mask': left_mask,
                'right_mask': right_mask,
                'success': trajectory is not None
            }
        except Exception as e:
            print(f"OpenVINO lane detection failed: {e}. Using straight trajectory.")
            trajectory = self._straight_trajectory()
            return {
                'lanes': [trajectory],
                'trajectory': trajectory,
                'fallback': True,
                'success': False
            }

    def detect_raw(self, img_array):
        """Raw detection using OpenVINO model."""
        # Preprocessing for OpenVINO - resize to expected model input size
        img_resized = cv2.resize(img_array, (1024, 512))  # Resize to model expected size
        img_array = np.expand_dims(np.transpose(img_resized, (2, 0, 1)), 0)
        
        # Run inference
        output_layer_ir = next(iter(self.compiled_model_ir.outputs))
        model_output = self.compiled_model_ir([img_array])[output_layer_ir]
        
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right

    def detect_and_fit(self, img_array):
        """Detect lanes and fit polynomials."""
        _, left, right = self.detect_raw(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def fit_poly(self, probs):
        """Fit polynomial to lane detection probabilities."""
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        
        if np.sum(mask) < 10:  # Not enough points
            return np.poly1d([0, 0, 0, 0])  # Straight line as fallback
            
        try:
            coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
            return np.poly1d(coeffs)
        except:
            return np.poly1d([0, 0, 0, 0])  # Straight line as fallback

    def _generate_trajectory_from_lanes(self, left_poly, right_poly):
        """Generate vehicle trajectory from left and right lane polynomials."""
        x = np.arange(-2, 60, 1.0)
        
        try:
            # Calculate y coordinates for left and right lanes
            y_left = left_poly(x)
            y_right = right_poly(x)
            
            # Calculate center line (trajectory to follow)
            # Note: multiply by -0.5 to account for coordinate system differences
            y = -0.5 * (y_left + y_right)
            
            # Adjust x coordinates (camera is 0.5m in front of vehicle center)
            x += 0.5
            
            # Create trajectory as numpy array
            trajectory = np.stack((x, y)).T
            return trajectory
            
        except Exception as e:
            print(f"Error generating trajectory from polynomials: {e}")
            return self._straight_trajectory()

    def _straight_trajectory(self):
        """Generate a straight trajectory as fallback."""
        x = np.arange(-2, 60, 1.0)
        y = np.zeros_like(x)  # Straight ahead
        return np.stack((x, y)).T

    def __call__(self, img):
        """Make the detector callable."""
        if isinstance(img, str):
            img = self.read_imagefile_to_array(img)
        return self.detect_and_fit(img) 