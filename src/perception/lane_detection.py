#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from pathlib import Path
from .openvino_lane_detector import OpenVINOLaneDetector
from .camera_geometry import CameraGeometry

# Import segmentation models
try:
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
except ImportError:
    print("Warning: Could not import segmentation models from torchvision. Make sure you have the correct version installed.")

class LaneDetector:
    """Class to handle lane detection using OpenVINO implementation from successful project."""
    
    def __init__(self, config):
        """
        Initialize lane detector.
        
        Args:
            config: Configuration dictionary for lane detection.
        """
        self.config = config
        self.device = config.get('device', 'cpu')
        self.model_name = config.get('model', 'openvino')
        
        # Initialize camera geometry for CARLA
        self.camera_geometry = CameraGeometry(
            height=1.3,  # Camera height in meters
            pitch_deg=5,  # Camera pitch angle
            image_width=1024,  # Image width
            image_height=512,  # Image height  
            field_of_view_deg=45  # Field of view
        )
        
        # Initialize OpenVINO lane detector
        self.openvino_detector = OpenVINOLaneDetector(
            cam_geom=self.camera_geometry,
            model_path=config.get('openvino_model_path'),
            device="CPU"  # OpenVINO typically runs on CPU
        )
        
        print(f"Lane detector initialized with OpenVINO implementation")
        
        # Store previous lane detections for temporal smoothing
        self.prev_lanes = None
        self.smoothing_factor = 0.7  # Weight for current detection vs previous
    
    def _load_model(self):
        """
        Load the lane detection model.
        
        Returns:
            The lane detection model.
        """
        if self.model_name == 'deeplabv3_resnet50':
            model = deeplabv3_resnet50(pretrained=True)
        elif self.model_name == 'deeplabv3_resnet101':
            model = deeplabv3_resnet101(pretrained=True)
        elif self.model_name == 'lraspp_mobilenet_v3_large':
            model = lraspp_mobilenet_v3_large(pretrained=True)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
        
        return model
    
    def detect(self, image):
        """
        Detect lane lines in the input image using OpenVINO implementation.
        
        Args:
            image: Input image as a numpy array (RGB format).
            
        Returns:
            Dictionary containing lane information compatible with the original interface.
        """
        try:
            # Get lane trajectory from OpenVINO detector
            trajectory = self.openvino_detector.get_lane_trajectory(image)
        
            # Convert trajectory to the expected lane format
            lane_info = self._convert_trajectory_to_lanes(trajectory)
        
            # Apply temporal smoothing
            smoothed_lanes = self._apply_temporal_smoothing(lane_info)
        
        return smoothed_lanes
    
        except Exception as e:
            print(f"Error in OpenVINO lane detection: {e}")
            # Return fallback straight lane
            return self._get_fallback_lanes()

    def _convert_trajectory_to_lanes(self, trajectory):
        """
        Convert trajectory from OpenVINO detector to lane format expected by the system.
        
        Args:
            trajectory: List of (x, y) points in vehicle coordinates
            
        Returns:
            Dictionary with lane information
        """
        lane_info = {
            'lanes': [],
            'trajectory': trajectory,
            'center_line': trajectory
        }
        
        # If we have a valid trajectory, create lane representations
        if trajectory and len(trajectory) > 1:
            # Convert trajectory points to lane format
            lane_points = [(int(x*50 + 320), int(400 - y*10)) for x, y in trajectory[:10]]  # Simple conversion to image coordinates
            
            lane_info['lanes'] = [{
                'points': lane_points,
                'confidence': 0.8,
                'type': 'center',
                'polynomial_coeffs': None  # Could add polynomial fitting if needed
            }]
        
        return lane_info

    def _get_fallback_lanes(self):
        """Get fallback straight lane when detection fails."""
        # Create a straight forward trajectory
        trajectory = [(float(x), 0.0) for x in range(0, 30, 2)]
        
        return {
            'lanes': [{
                'points': [(320, 400), (320, 300), (320, 200)],  # Straight line in image coordinates
                'confidence': 0.5,
                'type': 'center',
                'polynomial_coeffs': None
            }],
            'trajectory': trajectory,
            'center_line': trajectory
        }
    
    def get_trajectory_points(self, image):
        """
        Get trajectory points for path planning.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of (x, y) trajectory points in vehicle coordinates
        """
        try:
            # Use OpenVINO detector to get trajectory
            trajectory = self.openvino_detector.get_lane_trajectory(image)
            return trajectory
        except Exception as e:
            print(f"Error getting trajectory: {e}")
            # Return straight fallback trajectory
            return [(float(x), 0.0) for x in range(0, 30, 2)]
    
    def _apply_temporal_smoothing(self, current_lanes):
        """Apply temporal smoothing to reduce jitter in lane detection."""
        if self.prev_lanes is None:
            self.prev_lanes = current_lanes
            return current_lanes
            
        smoothed_lanes = current_lanes.copy()
        
        # Smooth the trajectory if available
        if 'trajectory' in current_lanes and 'trajectory' in self.prev_lanes:
            current_traj = current_lanes['trajectory']
            prev_traj = self.prev_lanes['trajectory']
            
            if current_traj and prev_traj and len(current_traj) == len(prev_traj):
                smoothed_trajectory = []
                for i in range(len(current_traj)):
                    smooth_x = current_traj[i][0] * self.smoothing_factor + \
                              prev_traj[i][0] * (1 - self.smoothing_factor)
                    smooth_y = current_traj[i][1] * self.smoothing_factor + \
                              prev_traj[i][1] * (1 - self.smoothing_factor)
                    smoothed_trajectory.append((smooth_x, smooth_y))
                
                smoothed_lanes['trajectory'] = smoothed_trajectory
                smoothed_lanes['center_line'] = smoothed_trajectory
                
        # Save current smoothed result for next iteration
        self.prev_lanes = smoothed_lanes
        return smoothed_lanes
    
    def _enhance_with_curve_fitting(self, lane_info):
        """Enhance lane detection with polynomial curve fitting for curved roads."""
        enhanced_lane_info = {'lanes': []}
        
        for lane in lane_info['lanes']:
            # Get original points
            original_points = lane['points']
            
            # Need at least 2 points for any fitting
            if len(original_points) < 2:
                enhanced_lane_info['lanes'].append(lane)
                continue
                
            # Extract x and y coordinates
            x_points = np.array([p[0] for p in original_points])
            y_points = np.array([p[1] for p in original_points])
            
            # If we have enough points, fit a second-degree polynomial (curved lane)
            if len(original_points) >= 3:
                try:
                    # Fit polynomial: y = ax^2 + bx + c
                    coeffs = np.polyfit(y_points, x_points, 2)
                    
                    # Generate points along the polynomial curve
                    y_new = np.linspace(min(y_points), max(y_points), 10)
                    x_new = np.polyval(coeffs, y_new)
                    
                    # Create new points list
                    new_points = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
                    
                    # Create enhanced lane with more points for better curve representation
                    enhanced_lane = lane.copy()
                    enhanced_lane['points'] = new_points
                    enhanced_lane['curve_coeffs'] = coeffs.tolist()  # Store coefficients for path planning
                    enhanced_lane_info['lanes'].append(enhanced_lane)
                    continue
                except:
                    # If curve fitting fails, fall back to linear interpolation
                    pass
            
            # For lanes with only 2 points or if curve fitting failed, use linear interpolation
            try:
                # Fit a line: y = mx + b
                coeffs = np.polyfit(y_points, x_points, 1)
                
                # Generate points along the line
                y_new = np.linspace(min(y_points), max(y_points), 10)
                x_new = np.polyval(coeffs, y_new)
                
                # Create new points list
                new_points = [(int(x), int(y)) for x, y in zip(x_new, y_new)]
                
                # Create enhanced lane
                enhanced_lane = lane.copy()
                enhanced_lane['points'] = new_points
                enhanced_lane['curve_coeffs'] = coeffs.tolist()  # Store coefficients for path planning
                enhanced_lane_info['lanes'].append(enhanced_lane)
            except:
                # If all fitting fails, use original points
                enhanced_lane_info['lanes'].append(lane)
                
        return enhanced_lane_info
    
    def _detect_lanes_segmentation(self, image):
        """
        Detect lanes using semantic segmentation.
        
        Args:
            image: Input image as a numpy array (RGB format).
            
        Returns:
            Dictionary containing lane information.
        """
        # Resize image if needed
        original_size = image.shape[:2]
        img = cv2.resize(image, (640, 384))
        
        # Transform image for the model
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model output
            output = self.model(img_tensor)['out'][0]
            
            # Get road and lane classes (indices may vary depending on the model)
            road_mask = (output.argmax(0) == 0).cpu().numpy()  # Assuming 0 is road
            
            # Process the mask to get lane lines
            # This is a simplification - in practice, you would use a more sophisticated approach
            road_mask = cv2.resize(road_mask.astype(np.uint8) * 255, original_size[::-1])
            
            # Use edge detection to find boundaries in the road mask
            edges = cv2.Canny(road_mask, 100, 200)
            
            # Find lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
            
            # Process lines to get lane information
            lane_info = {
                'lanes': []
            }
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Create a simple lane representation
                    lane = {
                        'points': [(x1, y1), (x2, y2)],
                        'confidence': 0.8  # Placeholder confidence value
                    }
                    lane_info['lanes'].append(lane)
            
            return lane_info
    
    def _detect_lanes_cv(self, image):
        """
        Detect lanes using traditional computer vision techniques with improvements.
        
        Args:
            image: Input image as a numpy array (RGB format).
            
        Returns:
            Dictionary containing lane information.
        """
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create masks for yellow and white colors
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Combine masks
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Apply preprocessing to improve detection
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Apply region of interest mask (focus on lower part of the image)
        height, width = blurred.shape
        roi_vertices = np.array([
            [(0, height), (width // 3, height // 2), (2 * width // 3, height // 2), (width, height)]
        ], dtype=np.int32)
        
        roi_mask = np.zeros_like(blurred)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(blurred, roi_mask)
        
        # Apply Canny edge detection with improved parameters
        edges = cv2.Canny(masked_edges, 30, 150)  # Lower threshold to detect more edges
        
        # Apply dilation to connect broken lane lines
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply Hough transform with improved parameters for curved roads
        lines = cv2.HoughLinesP(
            dilated_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,  # Lower threshold to detect more lines
            minLineLength=20,  # Shorter minimum line length for curved segments
            maxLineGap=50  # Larger gap to connect segments in curves
        )
        
        # Process lines to get lane information
        lane_info = {
            'lanes': []
        }
        
        if lines is not None:
            # Group lines into left and right lanes based on slope
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                if x2 - x1 == 0:
                    continue  # Skip vertical lines
                
                slope = (y2 - y1) / (x2 - x1)
                
                # Improved slope thresholds to better capture curved lanes
                if -0.9 < slope < -0.1:  # Less restrictive negative slope for left lanes
                    left_lines.append(line[0])
                elif 0.1 < slope < 0.9:  # Less restrictive positive slope for right lanes
                    right_lines.append(line[0])
            
            # Average left lanes
            if left_lines:
                x1_avg, y1_avg, x2_avg, y2_avg = self._average_lines(left_lines)
                lane_info['lanes'].append({
                    'points': [(x1_avg, y1_avg), (x2_avg, y2_avg)],
                    'confidence': 0.9,
                    'side': 'left'
                })
            
            # Average right lanes
            if right_lines:
                x1_avg, y1_avg, x2_avg, y2_avg = self._average_lines(right_lines)
                lane_info['lanes'].append({
                    'points': [(x1_avg, y1_avg), (x2_avg, y2_avg)],
                    'confidence': 0.9,
                    'side': 'right'
                })
        
        return lane_info
    
    def _average_lines(self, lines):
        """
        Average multiple lines into a single line.
        
        Args:
            lines: List of lines, each represented as [x1, y1, x2, y2].
            
        Returns:
            Averaged line as (x1, y1, x2, y2).
        """
        x1_sum = 0
        y1_sum = 0
        x2_sum = 0
        y2_sum = 0
        
        for line in lines:
            x1, y1, x2, y2 = line
            x1_sum += x1
            y1_sum += y1
            x2_sum += x2
            y2_sum += y2
        
        n = len(lines)
        x1_avg = int(x1_sum / n)
        y1_avg = int(y1_sum / n)
        x2_avg = int(x2_sum / n)
        y2_avg = int(y2_sum / n)
        
        return x1_avg, y1_avg, x2_avg, y2_avg 