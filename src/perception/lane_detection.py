#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from pathlib import Path

# Import segmentation models
try:
    from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
except ImportError:
    print("Warning: Could not import segmentation models from torchvision. Make sure you have the correct version installed.")

class LaneDetector:
    """Class to handle lane detection using semantic segmentation."""
    
    def __init__(self, config):
        """
        Initialize lane detector.
        
        Args:
            config: Configuration dictionary for lane detection.
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = config.get('model', 'deeplabv3_resnet50')
        
        # Initialize model
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Lane detector initialized with model: {self.model_name} on device: {self.device}")
        
        # Lane color range in HSV - expanded ranges for better detection
        self.yellow_lower = np.array([15, 60, 60])  # Lower saturation and value thresholds
        self.yellow_upper = np.array([40, 255, 255])  # Wider hue range
        
        self.white_lower = np.array([0, 0, 180])  # Lower value threshold
        self.white_upper = np.array([180, 40, 255])  # Higher saturation threshold
        
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
        Detect lane lines in the input image.
        
        Args:
            image: Input image as a numpy array (RGB format).
            
        Returns:
            Dictionary containing lane information.
        """
        # Use both segmentation-based and traditional computer vision approaches
        lanes_cv = self._detect_lanes_cv(image)
        lanes_segmentation = self._detect_lanes_segmentation(image)
        
        # Combine the results with improved fusion logic
        combined_lanes = self._fuse_lane_detections(lanes_cv, lanes_segmentation)
        
        # Apply temporal smoothing with previous detections
        smoothed_lanes = self._apply_temporal_smoothing(combined_lanes)
        
        return smoothed_lanes
    
    def _fuse_lane_detections(self, lanes_cv, lanes_segmentation):
        """
        Fuse lane detections from multiple methods with improved logic.
        
        Args:
            lanes_cv: Lane information from CV approach.
            lanes_segmentation: Lane information from segmentation approach.
            
        Returns:
            Combined lane information.
        """
        combined_lanes = {'lanes': []}
        
        # Start with CV lanes (generally more reliable for lane markings)
        if lanes_cv['lanes']:
            combined_lanes['lanes'].extend(lanes_cv['lanes'])
            
        # Add segmentation lanes that don't overlap with CV lanes
        if lanes_segmentation['lanes']:
            for seg_lane in lanes_segmentation['lanes']:
                is_duplicate = False
                for cv_lane in lanes_cv['lanes']:
                    # Check if lanes are similar (simplified)
                    if self._are_lanes_similar(seg_lane, cv_lane):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    # Add confidence field if not present
                    if 'confidence' not in seg_lane:
                        seg_lane['confidence'] = 0.7
                    combined_lanes['lanes'].append(seg_lane)
        
        # Add curve fitting for improved detection of curved lanes
        if combined_lanes['lanes']:
            combined_lanes = self._enhance_with_curve_fitting(combined_lanes)
            
        return combined_lanes
    
    def _are_lanes_similar(self, lane1, lane2):
        """Check if two lane detections are similar and likely duplicates."""
        # Compare endpoints of lane segments
        if len(lane1['points']) < 2 or len(lane2['points']) < 2:
            return False
            
        p1_start, p1_end = lane1['points'][0], lane1['points'][-1]
        p2_start, p2_end = lane2['points'][0], lane2['points'][-1]
        
        # Calculate distances between endpoints
        start_dist = np.sqrt((p1_start[0] - p2_start[0])**2 + (p1_start[1] - p2_start[1])**2)
        end_dist = np.sqrt((p1_end[0] - p2_end[0])**2 + (p1_end[1] - p2_end[1])**2)
        
        # Check if both start and end points are close
        return start_dist < 50 and end_dist < 50
    
    def _apply_temporal_smoothing(self, current_lanes):
        """Apply temporal smoothing to reduce jitter in lane detection."""
        if self.prev_lanes is None:
            self.prev_lanes = current_lanes
            return current_lanes
            
        smoothed_lanes = {'lanes': []}
        
        # For each current lane, find matching previous lane and apply smoothing
        for current_lane in current_lanes['lanes']:
            matched = False
            
            for prev_lane in self.prev_lanes['lanes']:
                if self._are_lanes_similar(current_lane, prev_lane):
                    # Apply smoothing to points
                    smoothed_points = []
                    for i in range(min(len(current_lane['points']), len(prev_lane['points']))):
                        smoothed_x = current_lane['points'][i][0] * self.smoothing_factor + \
                                    prev_lane['points'][i][0] * (1 - self.smoothing_factor)
                        smoothed_y = current_lane['points'][i][1] * self.smoothing_factor + \
                                    prev_lane['points'][i][1] * (1 - self.smoothing_factor)
                        smoothed_points.append((int(smoothed_x), int(smoothed_y)))
                    
                    # Create smoothed lane
                    smoothed_lane = current_lane.copy()
                    smoothed_lane['points'] = smoothed_points
                    smoothed_lanes['lanes'].append(smoothed_lane)
                    matched = True
                    break
            
            # If no match found, add current lane as is
            if not matched:
                smoothed_lanes['lanes'].append(current_lane)
                
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