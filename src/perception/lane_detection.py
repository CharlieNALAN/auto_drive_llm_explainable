#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .camera_geometry import CameraGeometry
import numpy as np
import cv2
import torch
import os
from pathlib import Path

# Try to import advanced dependencies, fallback gracefully if not available
try:
    import segmentation_models_pytorch as smp
    import albumentations as albu
    SMP_AVAILABLE = True
except ImportError:
    print("Warning: segmentation_models_pytorch not available. Using basic lane detection.")
    SMP_AVAILABLE = False

class LaneDetector:
    """Lane detector using proven architecture from working project."""
    
    def __init__(self, config=None, cam_geom=None, model_path=None, 
                 encoder='efficientnet-b0', encoder_weights='imagenet'):
        """
        Initialize lane detector.
        
        Args:
            config: Configuration dictionary (optional, for compatibility)
            cam_geom: Camera geometry object
            model_path: Path to the trained model
            encoder: Encoder architecture
            encoder_weights: Pretrained weights
        """
        self.cg = cam_geom if cam_geom else CameraGeometry()
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        # Handle model path - try OpenVINO first, then PyTorch
        self.model_available = False
        self.use_openvino = False
        
        if model_path is None:
            # Try OpenVINO model first (more reliable)
            openvino_paths = [
                '../self-driving-carla-main/converted_model/lane_model.xml',
                'converted_model/lane_model.xml',
                'models/lane_model.xml'
            ]
            
            # Try PyTorch models as fallback
            pytorch_paths = [
                '../self-driving-carla-main/lane_detection/Deeplabv3+(MobilenetV2).pth',
                'lane_detection/Deeplabv3+(MobilenetV2).pth',
                'models/lane_model.pth'
            ]
            
            # First try OpenVINO
            if OPENVINO_AVAILABLE:
                for path in openvino_paths:
                    if Path(path).exists():
                        model_path = path
                        self.use_openvino = True
                        break
            
            # If no OpenVINO model found, try PyTorch
            if not model_path and SMP_AVAILABLE:
                for path in pytorch_paths:
                    if Path(path).exists():
                        model_path = path
                        self.use_openvino = False
                        break
        
        # Initialize model
        if model_path and Path(model_path).exists():
            try:
                if self.use_openvino and OPENVINO_AVAILABLE:
                    # Use OpenVINO
                    ie = Core()
                    model_ir = ie.read_model(model_path)
                    self.model = ie.compile_model(model=model_ir, device_name="CPU")
                    self.device = "CPU"
                    self.model_available = True
                    print(f"✓ Loaded OpenVINO lane detection model from: {model_path}")
                elif not self.use_openvino and SMP_AVAILABLE:
                    # Use PyTorch
                    if torch.cuda.is_available():
                        self.device = "cuda"
                        self.model = torch.load(model_path).to(self.device)
                    else:
                        self.model = torch.load(model_path, map_location=torch.device("cpu"))
                        self.device = "cpu"
                    self.model_available = True
                    print(f"✓ Loaded PyTorch lane detection model from: {model_path}")
                else:
                    print("✗ Model loading failed: No compatible runtime available")
                    self.device = "cpu"
            except Exception as e:
                print(f"✗ Failed to load model from {model_path}: {e}")
                self.device = "cpu"
        else:
            print("ℹ No suitable model found, using computer vision fallback")
            self.device = "cpu"
        
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        
        # Setup preprocessing
        if self.model_available:
            if self.use_openvino:
                # OpenVINO uses simple preprocessing
                self.to_tensor_func = self._simple_preprocessing
            elif SMP_AVAILABLE:
                # PyTorch uses SMP preprocessing
                try:
                    preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
                    self.to_tensor_func = self._get_preprocessing(preprocessing_fn)
                except:
                    print("Warning: Could not setup SMP preprocessing. Using simple preprocessing.")
                    self.to_tensor_func = self._simple_preprocessing
            else:
                self.to_tensor_func = self._simple_preprocessing
        else:
            self.to_tensor_func = self._simple_preprocessing
        
        print(f"Lane detector initialized on device: {self.device}")

    def _get_preprocessing(self, preprocessing_fn):
        """Get preprocessing function for the model."""
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')
        
        transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        return albu.Compose(transform)
    
    def _simple_preprocessing(self, image):
        """Simple preprocessing fallback."""
        image = cv2.resize(image, (1024, 512))
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return {"image": image}
    
    def detect(self, img_array):
        """
        Detect lanes and return trajectory information.
        
        Args:
            img_array: Input image as numpy array
            
        Returns:
            Dictionary containing lane information compatible with original interface
        """
        try:
            if self.model_available:
                # Use the proven deep learning approach
                left_poly, right_poly, left_mask, right_mask = self.detect_and_fit(img_array)
                trajectory = self._generate_trajectory_from_lanes(left_poly, right_poly)
            else:
                # Use computer vision fallback
                trajectory = self._cv_lane_detection(img_array)
                left_poly = right_poly = None
                left_mask = right_mask = None
            
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
            print(f"Lane detection failed: {e}. Using straight trajectory.")
            trajectory = self._straight_trajectory()
            return {
                'lanes': [trajectory],
                'trajectory': trajectory,
                'fallback': True,
                'success': False
            }
    
    def detect_and_fit(self, img_array):
        """Detect lanes and fit polynomials (deep learning approach)."""
        if not self.model_available:
            raise RuntimeError("Model not available for deep learning detection")
            
        _, left, right = self.detect_raw(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def detect_raw(self, img_array):
        """Raw detection using the deep learning model."""
        image_tensor = self.to_tensor_func(image=img_array)["image"]
        x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            model_output = self.model.predict(x_tensor).cpu().numpy()
            
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right
    
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
    
    def _cv_lane_detection(self, img_array):
        """Computer vision based lane detection fallback."""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Create masks for white and yellow lane markings
            white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 30, 255))
            yellow_mask = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
            combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            
            # Detect edges
            edges = cv2.Canny(blurred, 50, 150)
            
            # Define region of interest (bottom half of image)
            height = edges.shape[0]
            roi_vertices = np.array([[(0, height), (img_array.shape[1], height), 
                                     (img_array.shape[1], height//2), (0, height//2)]], dtype=np.int32)
            roi_mask = np.zeros_like(edges)
            cv2.fillPoly(roi_mask, roi_vertices, 255)
            edges_roi = cv2.bitwise_and(edges, roi_mask)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=50, 
                                   minLineLength=50, maxLineGap=50)
            
            if lines is not None and len(lines) > 0:
                # Convert to trajectory points
                trajectory_points = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Convert image coordinates to world coordinates using camera geometry
                    try:
                        world_start = self.cg.uv_to_roadXYZ_roadframe_iso8855(x1, y1)
                        world_end = self.cg.uv_to_roadXYZ_roadframe_iso8855(x2, y2)
                        trajectory_points.extend([world_start[:2], world_end[:2]])
                    except:
                        continue
                
                if trajectory_points:
                    # Sort by x coordinate and create smooth trajectory
                    trajectory_points = sorted(trajectory_points, key=lambda p: p[0])
                    return np.array(trajectory_points)
            
            return self._straight_trajectory()
            
        except Exception as e:
            print(f"CV lane detection failed: {e}")
            return self._straight_trajectory()
    
    def _straight_trajectory(self):
        """Generate a straight trajectory as fallback."""
        x = np.arange(0, 60, 1.0)
        y = np.zeros_like(x)
        trajectory = np.stack((x, y)).T
        return trajectory

    def __call__(self, img):
        """Make the class callable for compatibility."""
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = self.detect(img)
        if result.get('left_poly') and result.get('right_poly'):
            return result['left_poly'], result['right_poly'], result.get('left_mask'), result.get('right_mask')
        else:
            # Return dummy polynomials for compatibility
            dummy_poly = np.poly1d([0, 0, 0, 0])
            return dummy_poly, dummy_poly, None, None 