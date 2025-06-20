#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration test script for the enhanced LLM-explainable autonomous driving system.
This script tests the integration of proven lane detection and control algorithms
from the working project into the LLM explainable framework.
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_camera_geometry():
    """Test camera geometry module."""
    print("Testing camera geometry...")
    
    from src.perception.camera_geometry import CameraGeometry
    
    cg = CameraGeometry()
    
    # Test coordinate transformation
    u, v = 512, 400  # Image center roughly
    try:
        world_coords = cg.uv_to_roadXYZ_roadframe_iso8855(u, v)
        print(f"✓ Coordinate transformation works: ({u}, {v}) -> {world_coords}")
    except Exception as e:
        print(f"✗ Coordinate transformation failed: {e}")
        return False
    
    # Test grid precomputation
    try:
        cut_v, grid = cg.precompute_grid()
        print(f"✓ Grid precomputation works: cut_v={cut_v}, grid shape={grid.shape}")
    except Exception as e:
        print(f"✗ Grid precomputation failed: {e}")
        return False
    
    return True

def test_lane_detector():
    """Test lane detector module."""
    print("\nTesting lane detector...")
    
    try:
        from src.perception.openvino_lane_detector import OpenVINOLaneDetector
        OPENVINO_AVAILABLE = True
    except ImportError:
        OPENVINO_AVAILABLE = False
    
    from src.perception.lane_detection import LaneDetector
    from src.perception.camera_geometry import CameraGeometry
    
    cg = CameraGeometry()
    
    # Try OpenVINO detector first, fallback to regular detector
    try:
        if OPENVINO_AVAILABLE:
            lane_detector = OpenVINOLaneDetector(cam_geom=cg)
            print("✓ Lane detector initialized successfully (OpenVINO)")
        else:
            raise ImportError("OpenVINO not available")
    except Exception:
        try:
            lane_detector = LaneDetector(cam_geom=cg)
            print("✓ Lane detector initialized successfully (fallback)")
        except Exception as e:
            print(f"✗ Lane detector initialization failed: {e}")
            return False
    
    # Test with dummy image
    try:
        dummy_image = np.zeros((512, 1024, 3), dtype=np.uint8)
        result = lane_detector.detect(dummy_image)
        print(f"✓ Lane detection works with dummy image: {list(result.keys())}")
    except Exception as e:
        print(f"✗ Lane detection failed: {e}")
        return False
    
    return True

def test_pure_pursuit_controller():
    """Test pure pursuit controller."""
    print("\nTesting pure pursuit controller...")
    
    from src.control.pure_pursuit import PurePursuitPlusPID
    from src.control.get_target_point import get_target_point, get_curvature
    
    try:
        controller = PurePursuitPlusPID()
        print("✓ Pure pursuit controller initialized successfully")
    except Exception as e:
        print(f"✗ Pure pursuit controller initialization failed: {e}")
        return False
    
    # Test with dummy trajectory
    try:
        trajectory = np.array([[i, 0] for i in range(20)], dtype=float)  # Straight line
        speed = 5.0
        desired_speed = 5.0
        dt = 0.033
        
        throttle, steer = controller.get_control(trajectory, speed, desired_speed, dt)
        print(f"✓ Control computation works: throttle={throttle:.3f}, steer={steer:.3f}")
    except Exception as e:
        print(f"✗ Control computation failed: {e}")
        return False
    
    # Test target point calculation
    try:
        target_point = get_target_point(10.0, trajectory)
        print(f"✓ Target point calculation works: {target_point}")
    except Exception as e:
        print(f"✗ Target point calculation failed: {e}")
        return False
    
    # Test curvature calculation
    try:
        curvature = get_curvature(trajectory)
        print(f"✓ Curvature calculation works: {curvature:.6f}")
    except Exception as e:
        print(f"✗ Curvature calculation failed: {e}")
        return False
    
    return True

def test_trajectory_utils():
    """Test trajectory utility functions."""
    print("\nTesting trajectory utils...")
    
    from src.utils.trajectory_utils import dist_point_linestring
    
    # Test distance calculation
    try:
        point = np.array([0, 0])
        line = np.array([[1, 0], [2, 0], [3, 0]])
        distance = dist_point_linestring(point, line)
        print(f"✓ Distance calculation works: {distance:.3f}")
    except Exception as e:
        print(f"✗ Distance calculation failed: {e}")
        return False
    
    return True

def test_llm_explainer():
    """Test LLM explainer module."""
    print("\nTesting LLM explainer...")
    
    from src.explainability.llm_explainer import LLMExplainer
    
    config = {
        "local": {"model_path": "nonexistent.gguf"},
        "openai": {"model": "gpt-3.5-turbo"},
        "anthropic": {"model": "claude-3-haiku"}
    }
    
    try:
        explainer = LLMExplainer(config, model_type='local')  # Will use mock mode
        print("✓ LLM explainer initialized successfully")
    except Exception as e:
        print(f"✗ LLM explainer initialization failed: {e}")
        return False
    
    # Test explanation generation
    try:
        # Create a mock control object with attributes instead of dict
        class MockControl:
            def __init__(self):
                self.throttle = 0.3
                self.steer = 0.0
                self.brake = 0.0
        
        dummy_data = {
            "vehicle_state": {
                "speed": 20.0,
                "target_speed": 20.0,
                "control": MockControl()
            },
            "perception": {
                "detected_objects": [],
                "lane_info": {"success": True}
            },
            "planning": {
                "behavior": "follow_lane",
                "reason": "Normal lane following"
            }
        }
        
        explanation = explainer.explain(dummy_data)
        print(f"✓ Explanation generation works: '{explanation}'")
    except Exception as e:
        print(f"✗ Explanation generation failed: {e}")
        return False
    
    return True

def main():
    """Run all integration tests."""
    print("=== Integration Test for Enhanced Autonomous Driving System ===\n")
    
    tests = [
        test_camera_geometry,
        test_lane_detector,
        test_pure_pursuit_controller,
        test_trajectory_utils,
        test_llm_explainer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("✓ All tests passed! The integration appears to be successful.")
        print("\nNext steps:")
        print("1. Copy the lane detection model from the working project:")
        print("   cp ../self-driving-carla-main/lane_detection/Deeplabv3+\\(MobilenetV2\\).pth ./")
        print("2. Install additional dependencies:")
        print("   pip install segmentation_models_pytorch albumentations")
        print("3. Run the main system:")
        print("   python main.py")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 