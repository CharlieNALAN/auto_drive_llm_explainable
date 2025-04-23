#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO

class ObjectDetector:
    """Class to handle object detection using YOLOv8."""
    
    def __init__(self, config):
        """
        Initialize object detector.
        
        Args:
            config: Configuration dictionary for object detection.
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence = config.get('confidence', 0.5)
        self.classes = config.get('classes', None)  # If None, detect all classes
        
        # Initialize YOLO model
        model_path = self._get_model_path(config['model'])
        self.model = YOLO(model_path)
        
        print(f"Object detector initialized with model: {model_path} on device: {self.device}")
    
    def _get_model_path(self, model_name):
        """
        Get the path to the YOLO model.
        
        Args:
            model_name: Name of the model file.
            
        Returns:
            Path to the model.
        """
        # Check if model is a full path
        if os.path.exists(model_name):
            return model_name
        
        # Check if model is a built-in YOLO model (like 'yolov8n.pt')
        if model_name in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']:
            return model_name
        
        # Otherwise, look in the models directory
        models_dir = Path('models')
        model_path = models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return str(model_path)
    
    def detect(self, image):
        """
        Perform object detection on the input image.
        
        Args:
            image: Input image as a numpy array (RGB format).
            
        Returns:
            List of detected objects with their properties.
        """
        # Make predictions
        results = self.model.predict(
            image, 
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        # Convert results to a list of dictionaries
        detected_objects = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Create object dictionary
                obj = {
                    'bbox': [x1, y1, x2, y2],  # [x1, y1, x2, y2] format
                    'confidence': conf,
                    'class_id': cls_id,
                    'id': i  # Simple ID based on detection order
                }
                
                detected_objects.append(obj)
        
        return detected_objects 