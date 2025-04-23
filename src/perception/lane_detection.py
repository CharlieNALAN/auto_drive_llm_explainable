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
        
        # Lane color range in HSV
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([35, 255, 255])
        
        self.white_lower = np.array([0, 0, 200])
        self.white_upper = np.array([180, 25, 255])
    
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
        
        # Combine the results (in practice, you would want a more sophisticated fusion)
        lanes = lanes_cv.copy()  # Start with CV-based lanes
        if not lanes['lanes'] and lanes_segmentation['lanes']:
            # If CV didn't find lanes but segmentation did
            lanes = lanes_segmentation
        
        return lanes
    
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
        Detect lanes using traditional computer vision techniques.
        
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
        
        # Apply region of interest mask (focus on lower part of the image)
        height, width = mask.shape
        roi_vertices = np.array([
            [(0, height), (width // 2, height // 2), (width, height)]
        ], dtype=np.int32)
        
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(mask, roi_mask)
        
        # Apply Canny edge detection
        edges = cv2.Canny(masked_edges, 50, 150)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
        
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
                
                if slope < 0:
                    left_lines.append(line[0])
                else:
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