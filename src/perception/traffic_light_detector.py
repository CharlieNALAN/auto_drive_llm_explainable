#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class TrafficLightDetector:
    """
    Traffic light color detection system.
    Analyzes YOLO-detected traffic light regions to determine red/green/yellow state.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize traffic light detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Color ranges in HSV
        self.red_ranges = [
            # Red range 1 (lower hue)
            (np.array([0, 50, 50]), np.array([10, 255, 255])),
            # Red range 2 (upper hue)
            (np.array([170, 50, 50]), np.array([180, 255, 255]))
        ]
        
        self.green_range = (np.array([40, 50, 50]), np.array([80, 255, 255]))
        self.yellow_range = (np.array([15, 50, 50]), np.array([35, 255, 255]))
        
        # Minimum area for valid color detection
        self.min_color_area = 20
        
        # Brightness threshold for active lights
        self.brightness_threshold = 100
        
    def detect_traffic_light_state(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Detect the state of a traffic light in the given bounding box.
        
        Args:
            image: Input image (RGB format)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with traffic light state information
        """
        x1, y1, x2, y2 = bbox
        
        # Extract traffic light region
        traffic_light_roi = image[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return {'state': 'unknown', 'confidence': 0.0}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_RGB2HSV)
        
        # Detect each color
        red_score = self._detect_red_light(hsv)
        green_score = self._detect_green_light(hsv)
        yellow_score = self._detect_yellow_light(hsv)
        
        # Determine the dominant color
        scores = {'red': red_score, 'green': green_score, 'yellow': yellow_score}
        
        # Get the color with highest score
        max_color = max(scores, key=scores.get)
        max_score = scores[max_color]
        
        # Check if the score is significant enough
        if max_score < 0.1:
            return {'state': 'unknown', 'confidence': 0.0}
        
        return {
            'state': max_color,
            'confidence': max_score,
            'scores': scores
        }
    
    def _detect_red_light(self, hsv: np.ndarray) -> float:
        """Detect red light in HSV image."""
        red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Combine both red ranges
        for lower, upper in self.red_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            red_mask = cv2.bitwise_or(red_mask, mask)
        
        return self._calculate_color_score(hsv, red_mask)
    
    def _detect_green_light(self, hsv: np.ndarray) -> float:
        """Detect green light in HSV image."""
        lower, upper = self.green_range
        green_mask = cv2.inRange(hsv, lower, upper)
        
        return self._calculate_color_score(hsv, green_mask)
    
    def _detect_yellow_light(self, hsv: np.ndarray) -> float:
        """Detect yellow light in HSV image."""
        lower, upper = self.yellow_range
        yellow_mask = cv2.inRange(hsv, lower, upper)
        
        return self._calculate_color_score(hsv, yellow_mask)
    
    def _calculate_color_score(self, hsv: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate color score based on mask area and brightness.
        
        Args:
            hsv: HSV image
            mask: Binary mask for the color
            
        Returns:
            Color score (0.0 to 1.0)
        """
        # Count pixels in the mask
        color_pixels = cv2.countNonZero(mask)
        
        if color_pixels < self.min_color_area:
            return 0.0
        
        # Calculate relative area
        total_pixels = hsv.shape[0] * hsv.shape[1]
        area_ratio = color_pixels / total_pixels
        
        # Calculate average brightness in the masked region
        masked_v = cv2.bitwise_and(hsv[:, :, 2], mask)
        if color_pixels > 0:
            avg_brightness = np.sum(masked_v) / color_pixels
            brightness_score = min(avg_brightness / 255.0, 1.0)
        else:
            brightness_score = 0.0
        
        # Combine area and brightness scores
        final_score = area_ratio * brightness_score
        
        return min(final_score * 10, 1.0)  # Scale up and cap at 1.0
    
    def analyze_traffic_lights(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Analyze all traffic lights in the detection list.
        
        Args:
            image: Input image (RGB format)
            detections: List of YOLO detections
            
        Returns:
            List of traffic light detections with state information
        """
        traffic_lights = []
        
        for detection in detections:
            # Check if this is a traffic light (class_id 9 in COCO)
            if detection.get('class_id') == 9:
                bbox = detection['bbox']
                state_info = self.detect_traffic_light_state(image, bbox)
                
                # Create enhanced detection with state
                enhanced_detection = detection.copy()
                enhanced_detection.update({
                    'traffic_light_state': state_info['state'],
                    'state_confidence': state_info['confidence'],
                    'color_scores': state_info.get('scores', {})
                })
                
                traffic_lights.append(enhanced_detection)
        
        return traffic_lights
    
    def get_traffic_light_color_for_visualization(self, state: str) -> Tuple[int, int, int]:
        """
        Get BGR color for visualization based on traffic light state.
        
        Args:
            state: Traffic light state ('red', 'green', 'yellow', 'unknown')
            
        Returns:
            BGR color tuple
        """
        color_map = {
            'red': (0, 0, 255),      # Red
            'green': (0, 255, 0),    # Green  
            'yellow': (0, 255, 255), # Yellow
            'unknown': (128, 128, 128) # Gray
        }
        
        return color_map.get(state, (128, 128, 128)) 