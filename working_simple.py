#!/usr/bin/env python
"""
Simple Working Version with YOLO Object Detection
Based on self-driving-carla-main/carla_sim.py with LLM explanations and YOLO detection
"""

import carla
import random
import numpy as np
import pygame
import cv2
import argparse
import logging
import queue
import re
import sys
import os

# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

# Import YOLO object detector and traffic light detector
from perception.object_detection import ObjectDetector
from perception.traffic_light_detector import TrafficLightDetector

# === CORE UTILITY FUNCTIONS ===
def carla_vec_to_np_array(vec):
    return np.array([vec.x, vec.y, vec.z])

def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5)
    return np.max(curvature)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return True
    return False

def send_control(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

# === YOLO CLASS NAMES ===
YOLO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def draw_detections(image, detections, traffic_light_detector=None):
    """Draw bounding boxes and labels on the image."""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Get class name
        class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else f"Class {class_id}"
        
        # Special handling for traffic lights
        if class_id == 9 and 'traffic_light_state' in detection:  # Traffic light
            state = detection['traffic_light_state']
            state_confidence = detection.get('state_confidence', 0.0)
            
            # Get color for traffic light state
            if traffic_light_detector:
                bbox_color = traffic_light_detector.get_traffic_light_color_for_visualization(state)
            else:
                bbox_color = (0, 255, 0)  # Default green
            
            # Draw bounding box with state color
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 3)
            
            # Enhanced label for traffic lights
            label = f"{class_name} ({state}): {confidence:.2f}"
        else:
            # Regular detection
            bbox_color = (0, 255, 0)  # Green for regular objects
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)
            label = f"{class_name}: {confidence:.2f}"
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), bbox_color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image

# === SYNCHRONOUS MODE ===
class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            no_rendering_mode=False,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

# === GEOMETRY FUNCTIONS ===
def linesegment_distances(p, a, b):
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)

def dist_point_linestring(p, line_string):
    a = line_string[:-1, :]
    b = line_string[1:, :]
    return np.min(linesegment_distances(p, a, b))

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True):
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:
        return []
    else:
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]
        if not full_line:
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        return intersections

def get_target_point(lookahead, polyline):
    intersections = []
    for j in range(len(polyline)-1):
        pt1 = polyline[j]
        pt2 = polyline[j+1]
        intersections += circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
    filtered = [p for p in intersections if p[0]>0]
    return filtered[0] if filtered else None

# === CONTROL CLASSES ===
class PurePursuit:
    def __init__(self, K_dd=0.4, wheel_base=2.65, waypoint_shift=1.4):
        self.K_dd = K_dd
        self.wheel_base = wheel_base
        self.waypoint_shift = waypoint_shift
    
    def get_control(self, waypoints, speed):
        waypoints[:,0] += self.waypoint_shift
        look_ahead_distance = np.clip(self.K_dd * speed, 3, 20)
        track_point = get_target_point(look_ahead_distance, waypoints)
        if track_point is None:
            waypoints[:,0] -= self.waypoint_shift
            return 0
        alpha = np.arctan2(track_point[1], track_point[0])
        steer = np.arctan((2 * self.wheel_base * np.sin(alpha)) / look_ahead_distance)
        waypoints[:,0] -= self.waypoint_shift
        return steer

class PIDController:
    def __init__(self, Kp, Ki, Kd, set_point):
        self.Kp, self.Ki, self.Kd, self.set_point = Kp, Ki, Kd, set_point
        self.int_term, self.derivative_term, self.last_error = 0, 0, None
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.int_term += error*self.Ki*dt
        if self.last_error is not None:
            self.derivative_term = (error-self.last_error)/dt*self.Kd
        self.last_error = error
        return self.Kp * error + self.int_term + self.derivative_term

class PurePursuitPlusPID:
    def __init__(self):
        self.pure_pursuit = PurePursuit()
        self.pid = PIDController(2, 0, 0, 0)

    def get_control(self, waypoints, speed, desired_speed, dt):
        self.pid.set_point = desired_speed
        a = self.pid.get_control(speed, dt)
        steer = self.pure_pursuit.get_control(waypoints, speed)
        return a, steer

# === TRAJECTORY FUNCTIONS ===
def get_trajectory_from_map(CARLA_map, vehicle):
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    trajectory = np.array([np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())
    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    return trajectory.T[:,:2]

# === SIMPLE LLM EXPLAINER ===
class SimpleLLMExplainer:
    def explain(self, speed, steer, curvature, detections=None):
        # Basic driving behavior
        if abs(steer) > 0.1:
            direction = "right" if steer > 0 else "left"
            base_explanation = f"Turning {direction} at {speed:.1f} km/h (curvature: {curvature:.4f})"
        else:
            base_explanation = f"Driving straight at {speed:.1f} km/h"
        
        # Add object detection information
        if detections and len(detections) > 0:
            # Get the most relevant objects (cars, persons, traffic lights, etc.)
            relevant_objects = []
            traffic_light_info = []
            
            for detection in detections:
                class_id = detection['class_id']
                if class_id < len(YOLO_CLASSES):
                    class_name = YOLO_CLASSES[class_id]
                    confidence = detection['confidence']
                    
                    # Special handling for traffic lights
                    if class_name == 'traffic light' and 'traffic_light_state' in detection:
                        state = detection['traffic_light_state']
                        if state != 'unknown':
                            traffic_light_info.append(f"{state} light")
                    # Focus on other traffic-relevant objects
                    elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'stop sign']:
                        relevant_objects.append(f"{class_name} ({confidence:.2f})")
            
            # Build explanation parts
            explanation_parts = [base_explanation]
            
            if traffic_light_info:
                explanation_parts.append(f"Traffic lights: {', '.join(traffic_light_info)}")
            
            if relevant_objects:
                objects_str = ", ".join(relevant_objects[:3])  # Limit to top 3 objects
                explanation_parts.append(f"Detected: {objects_str}")
            
            return ". ".join(explanation_parts)
        
        return base_explanation

# === MAIN FUNCTION ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO object detection')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    explainer = None if args.no_llm else SimpleLLMExplainer()
    
    # Initialize YOLO object detector
    object_detector = None
    traffic_light_detector = None
    if not args.no_yolo:
        try:
            yolo_config = {
                'model': 'yolov8n.pt',  # Use the model file in the project root
                'device': 'cuda',
                'confidence': 0.5,
                'classes': None  # Detect all classes
            }
            object_detector = ObjectDetector(yolo_config)
            traffic_light_detector = TrafficLightDetector()
            logger.info("YOLO object detector and traffic light detector initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO detector: {e}")
            object_detector = None
            traffic_light_detector = None
    
    pygame.init()
    
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Create vehicle  
    blueprint_library = world.get_blueprint_library()
    veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
    veh_bp.set_attribute('color','64,81,181')
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(veh_bp, spawn_point)
    
    # Create camera
    camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(blueprint_library.find('sensor.camera.rgb'), camera_transform, attach_to=vehicle)
    
    controller = PurePursuitPlusPID()
    startPoint = carla_vec_to_np_array(spawn_point.location)
    
    actor_list = [vehicle, camera]
    cross_track_list = []
    flag = True
    
    try:
        logger.info("Starting simulation...")
        with CarlaSyncMode(world, camera, fps=args.fps) as sync_mode:
            while True:
                if should_quit():
                    break
                    
                # Get sensor data
                snapshot, image = sync_mode.tick(timeout=2.0)
                
                # Convert image for processing
                img_array = carla_img_to_array(image)
                
                # Object detection
                detections = []
                if object_detector:
                    try:
                        detections = object_detector.detect(img_array)
                        
                        # Analyze traffic lights for color detection
                        if traffic_light_detector and detections:
                            traffic_lights = traffic_light_detector.analyze_traffic_lights(img_array, detections)
                            
                            # Update detections with traffic light state information
                            for i, detection in enumerate(detections):
                                if detection.get('class_id') == 9:  # Traffic light
                                    for tl in traffic_lights:
                                        if tl['id'] == detection['id']:
                                            detections[i].update({
                                                'traffic_light_state': tl['traffic_light_state'],
                                                'state_confidence': tl['state_confidence'],
                                                'color_scores': tl.get('color_scores', {})
                                            })
                                            break
                    except Exception as e:
                        logger.warning(f"YOLO detection failed: {e}")
                
                # Get trajectory from map
                trajectory = get_trajectory_from_map(world.get_map(), vehicle)
                
                # Speed control
                max_curvature = get_curvature(trajectory)
                move_speed = 5.56 if max_curvature <= 0.005 or flag else max(3.0, abs(5.56 - 20*max_curvature))
                
                # Get control
                speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./args.fps)
                send_control(vehicle, throttle, steer, 0)
                
                # Track position
                waypoint = world.get_map().get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)
                
                if np.linalg.norm(vehicle_loc-startPoint) > 20:
                    flag = False
                if np.linalg.norm(vehicle_loc-startPoint) < 20 and not flag:
                    logger.info('Route completed!')
                    break
                if speed < 1 and not flag:
                    logger.warning("Vehicle stopped!")
                    break
                
                # LLM explanation
                if explainer and len(cross_track_list) % 30 == 0:
                    explanation = explainer.explain(speed * 3.6, steer, max_curvature, detections)
                    logger.info(f"LLM: {explanation}")
                
                # Cross track error
                dist = dist_point_linestring(np.array([0,0]), trajectory)
                cross_track_list.append(int(dist))
                
                # Visualization with object detection
                img = cv2.resize(img_array, (800, 600))
                
                # Draw object detections
                if detections:
                    # Scale detection coordinates to resized image
                    scale_x = 800 / img_array.shape[1]
                    scale_y = 600 / img_array.shape[0]
                    
                    scaled_detections = []
                    for det in detections:
                        scaled_det = det.copy()
                        x1, y1, x2, y2 = det['bbox']
                        scaled_bbox = [
                            int(x1 * scale_x), int(y1 * scale_y),
                            int(x2 * scale_x), int(y2 * scale_y)
                        ]
                        scaled_det['bbox'] = scaled_bbox
                        scaled_detections.append(scaled_det)
                    
                    img = draw_detections(img, scaled_detections, traffic_light_detector)
                
                # Add driving information
                cv2.putText(img, f"Speed: {speed*3.6:.1f} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(img, f"Steer: {'R' if steer>0 else 'L'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(img, f"Objects: {len(detections)}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
                cv2.imshow('Autonomous Driving with YOLO', img)
                cv2.waitKey(1)
    
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        cv2.destroyAllWindows()
        pygame.quit()
        if cross_track_list:
            logger.info(f'Mean cross track error: {np.mean(cross_track_list):.2f}')

if __name__ == '__main__':
    main() 