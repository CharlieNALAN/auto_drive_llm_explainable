#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .camera_geometry import CameraGeometry

class OpenVINOLaneDetector():
    """OpenVINO-based lane detector."""
    
    def __init__(self, cam_geom=None, model_path='./converted_model/lane_model.xml', device="CPU"):
        self.cg = cam_geom or CameraGeometry()
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        try:
            from openvino.runtime import Core
            ie = Core()
            model_ir = ie.read_model(model_path)
            self.compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
        except ImportError:
            print("Warning: OpenVINO not found. Using map-based navigation only.")
            self.compiled_model_ir = None

    def detect(self, img_array):
        """
        Detect lane lines in the image.
        
        Args:
            img_array: Input image array
            
        Returns:
            Tuple of (background, left_lane, right_lane) probabilities
        """
        if self.compiled_model_ir is None:
            raise Exception("OpenVINO model not available")
        img_array = np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
        output_layer_ir = next(iter(self.compiled_model_ir.outputs))
        model_output = self.compiled_model_ir([img_array])[output_layer_ir]
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right

    def fit_poly(self, probs):
        """
        Fit polynomial to lane detection probabilities.
        
        Args:
            probs: Lane detection probabilities
            
        Returns:
            Polynomial function
        """
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs)

    def detect_and_fit(self, img_array):
        """
        Detect lane lines and fit polynomials.
        
        Args:
            img_array: Input image array
            
        Returns:
            Tuple of (left_poly, right_poly, left_probs, right_probs)
        """
        _, left, right = self.detect(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def __call__(self, img):
        """
        Callable interface for the detector.
        
        Args:
            img: Input image
            
        Returns:
            Tuple of (left_poly, right_poly, left_probs, right_probs)
        """
        return self.detect_and_fit(img) 