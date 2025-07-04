# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).
# Modified to include LLM explanations

import carla
import random
from pathlib import Path
import numpy as np
import pygame
import argparse
import cv2
import json
import logging
import queue
import weakref
import time
import re

# Import from original project structure (copy all needed functions)
import sys
import os

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import object detection and traffic light detection
try:
    from src.perception.object_detection import ObjectDetector
    from src.perception.traffic_light_detector import TrafficLightDetector
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"YOLO not available: {e}")
    YOLO_AVAILABLE = False

# YOLO class names
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
    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    return np.max(curvature)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def create_carla_world(pygame_module, mapid):
    pygame_module.init()
    # Reduce pygame window size for better performance
    display = pygame_module.display.set_mode((640, 480), pygame_module.HWSURFACE | pygame_module.DOUBLEBUF)
    font = get_font()
    clock = pygame_module.time.Clock()
    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)
    # client.load_world('Town0' + mapid)
    world = client.get_world()
    
    # Set world to low quality for better performance
    settings = world.get_settings()
    settings.no_rendering_mode = False
    settings.synchronous_mode = False  # Will be set later in sync mode
    # Reduce quality settings
    settings.fixed_delta_seconds = None  # Variable timestep for better performance
    world.apply_settings(settings)
    
    return display, font, clock, world

def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def draw_detections_opencv(img, detections):
    """Draw detection boxes on OpenCV image"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection.get('class_id', 0)
        confidence = detection.get('confidence', 0.0)
        class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else f"Class {class_id}"
        
        # Choose color based on object type
        if class_id == 9 and 'traffic_light_state' in detection:  # Traffic light
            state = detection['traffic_light_state']
            if state == 'red':
                color = (0, 0, 255)  # Red in BGR
            elif state == 'green':
                color = (0, 255, 0)  # Green in BGR
            elif state == 'yellow':
                color = (0, 255, 255)  # Yellow in BGR
            else:
                color = (255, 255, 255)  # White
            label = f"Traffic Light ({state})"
        else:
            color = (255, 0, 0)  # Blue in BGR for other objects
            label = f"{class_name}: {confidence:.2f}"
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

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

# Geometry functions
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

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
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
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections

def get_target_point(lookahead, polyline):
    intersections = []
    for j in range(len(polyline)-1):
        pt1 = polyline[j]
        pt2 = polyline[j+1]
        intersections += circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
    filtered = [p for p in intersections if p[0]>0]
    if len(filtered)==0:
        return None
    return filtered[0]

# Control classes
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
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.int_term = 0
        self.derivative_term = 0
        self.last_error = None
    
    def get_control(self, measurement, dt):
        error = self.set_point - measurement
        self.int_term += error*self.Ki*dt
        if self.last_error is not None:
            self.derivative_term = (error-self.last_error)/dt*self.Kd
        self.last_error = error
        return self.Kp * error + self.int_term + self.derivative_term

class PurePursuitPlusPID:
    def __init__(self, pure_pursuit=None, pid=None):
        self.pure_pursuit = pure_pursuit or PurePursuit()
        self.pid = pid or PIDController(2, 0, 0, 0)

    def get_control(self, waypoints, speed, desired_speed, dt):
        self.pid.set_point = desired_speed
        a = self.pid.get_control(speed, dt)
        steer = self.pure_pursuit.get_control(waypoints, speed)
        return a, steer

# Camera geometry
def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

class CameraGeometry(object):
    def __init__(self, height=1.3, pitch_deg=5, image_width=1024, image_height=512, field_of_view_deg=45):
        self.height = height
        self.pitch_deg = pitch_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        self.intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, image_width, image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        pitch = pitch_deg * np.pi/180
        cpitch, spitch = np.cos(pitch), np.sin(pitch)
        self.rotation_cam_to_road = np.array([[1,0,0],[0,cpitch,spitch],[0,-spitch,cpitch]])
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])

    def uv_to_roadXYZ_camframe(self,u,v):
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def camframe_to_roadframe(self,vec_in_cam_frame):
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_roadframe(self,u,v):
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y])

    def precompute_grid(self,dist=60):
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v

# OpenVINO Lane Detector
class OpenVINOLaneDetector():
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
        if self.compiled_model_ir is None:
            raise Exception("OpenVINO model not available")
        
        # Preprocess image to reduce crack interference
        img_processed = self._preprocess_image(img_array)
        img_processed = np.expand_dims(np.transpose(img_processed, (2, 0, 1)), 0)
        
        output_layer_ir = next(iter(self.compiled_model_ir.outputs))
        model_output = self.compiled_model_ir([img_processed])[output_layer_ir]
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right
    
    def _preprocess_image(self, img_array):
        """Preprocess image to reduce noise from cracks and improve lane detection"""
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply mild Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)
        
        # Apply mild adaptive histogram equalization to improve contrast
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Convert back to 3-channel
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return processed

    def fit_poly(self, probs):
        # Apply gentle morphological operations to reduce noise from cracks
        probs_clean = cv2.morphologyEx(probs, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        probs_clean = cv2.morphologyEx(probs_clean, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        probs_flat = np.ravel(probs_clean[self.cut_v:, :])
        mask = probs_flat > 0.3  # Balanced threshold
        
        if np.sum(mask) < 30:  # More lenient minimum pixel requirement
            # Fallback to original if too aggressive filtering
            probs_flat = np.ravel(probs[self.cut_v:, :])
            mask = probs_flat > 0.3
        
        coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs)

    def detect_and_fit(self, img_array):
        _, left, right = self.detect(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        
        # Validate lane detection quality (temporarily disabled for debugging)
        # if self._validate_lane_detection(left_poly, right_poly):
        #     return left_poly, right_poly, left, right
        # else:
        #     # If validation fails, use fallback (could be previous detection or map-based)
        #     raise Exception("Lane detection quality too low, using map fallback")
        
        # Temporarily return without validation to show green lines
        return left_poly, right_poly, left, right
    
    def _validate_lane_detection(self, left_poly, right_poly):
        """Validate lane detection to filter out false positives from cracks"""
        try:
            # Check lane width at different distances
            distances = [10, 20, 30]  # Shorter distances for more reliable validation
            widths = []
            
            for dist in distances:
                left_y = left_poly(dist)
                right_y = right_poly(dist)
                width = abs(right_y - left_y)
                widths.append(width)
            
            # More relaxed lane width range (2.0 to 8 meters)
            valid_widths = [2.0 <= w <= 8.0 for w in widths]
            if sum(valid_widths) < len(distances) * 0.5:  # Relaxed to 50%
                return False
            
            # More relaxed width consistency (standard deviation < 2.5 meters)
            width_std = np.std(widths)
            if width_std > 2.5:  # More tolerant
                return False
            
            return True
            
        except:
            return True  # Default to True if calculation fails

    def __call__(self, img):
        return self.detect_and_fit(img)

# Core trajectory functions
def get_trajectory_from_lane_detector(lane_detector, image):
    image_arr = carla_img_to_array(image)
    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    
    # Get original image, convert RGB to BGR, and resize it (smaller for better performance)
    img = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (600, 400))
    
    # Get lane detection results and overlay them on the original image
    lane_img = img_left + img_right
    lane_img = cv2.normalize(lane_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    lane_img = lane_img.astype(np.uint8)
    lane_img_resized = cv2.resize(lane_img, (600, 400))
    
    # Convert lane detection to 3-channel for overlay
    lane_img_colored = cv2.cvtColor(lane_img_resized, cv2.COLOR_GRAY2BGR)
    
    # Create overlay: original image + lane lines
    # Make lane lines more visible by using green color
    lane_mask = lane_img_resized > 50  # Threshold for lane pixels
    img[lane_mask] = [0, 255, 0]  # Green color for lane lines (BGR format)
    
    x = np.arange(-2, 60, 1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    x += 0.5
    trajectory = np.stack((x,y)).T
    return trajectory, img

def get_trajectory_from_map(CARLA_map, vehicle):
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())
    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:,:2]
    return trajectory

def send_control(vehicle, throttle, steer, brake, hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

# Simple LLM Explainer
class LLMExplainer:
    def __init__(self, config=None, model_type="local"):
        self.config = config or {}
        self.model_type = model_type
        
    def explain(self, data):
        """Generate explanation based on vehicle and perception data"""
        vehicle_state = data.get('vehicle_state', {})
        perception = data.get('perception', {})
        detections = data.get('detections', [])
        
        speed_kmh = vehicle_state.get('speed', 0)
        throttle = vehicle_state.get('control', {}).get('throttle', 0)
        steer = vehicle_state.get('control', {}).get('steer', 0)
        
        # Basic driving description
        if speed_kmh < 1:
            action = "Stopped"
        elif throttle > 0.1:
            action = "Accelerating"
        elif abs(steer) > 0.1:
            if steer > 0:
                action = "Turning right"
            else:
                action = "Turning left"
        else:
            action = "Driving straight"
        
        explanation = f"{action} at {speed_kmh:.1f} km/h"
        
        # Process detections for traffic lights and other objects
        traffic_light_info = []
        relevant_objects = []
        
        for detection in detections:
            class_id = detection.get('class_id', 0)
            confidence = detection.get('confidence', 0.0)
            
            # Get class name
            class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else f"Class {class_id}"
            
            # Special handling for traffic lights
            if class_name == 'traffic light' and 'traffic_light_state' in detection:
                state = detection['traffic_light_state']
                if state != 'unknown':
                    traffic_light_info.append(f"{state} light")
            # Focus on traffic-relevant objects
            elif class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'stop sign']:
                relevant_objects.append(f"{class_name} ({confidence:.2f})")
        
        # Add traffic light information
        if traffic_light_info:
            explanation += f". Traffic lights: {', '.join(traffic_light_info)}"
        
        # Add object detection information
        if relevant_objects:
            objects_str = ", ".join(relevant_objects[:3])  # Limit to top 3 objects
            explanation += f". Detected: {objects_str}"
        
        return explanation

def main(fps_sim=20, mapid='1', weather_idx=0, showmap=False, model_type="openvino", enable_llm=True, enable_yolo=True):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize LLM explainer
    explainer = LLMExplainer() if enable_llm else None
    
    # Initialize YOLO detector and traffic light detector
    yolo_detector = None
    traffic_light_detector = None
    if enable_yolo and YOLO_AVAILABLE:
        try:
            yolo_config = {
                'model': 'yolov8n.pt',  # Use the model file in the project root
                'device': 'cuda',
                'confidence': 0.5,
                'classes': None  # Detect all classes
            }
            yolo_detector = ObjectDetector(yolo_config)
            traffic_light_detector = TrafficLightDetector()
            logger.info("YOLO detector initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize YOLO detector: {e}")
            yolo_detector = None
            traffic_light_detector = None
    elif enable_yolo:
        logger.warning("YOLO requested but not available")
    
    actor_list = []
    pygame.init()

    display, font, clock, world = create_carla_world(pygame, mapid)

    # weather_presets = find_weather_presets()
    # world.set_weather(weather_presets[weather_idx][0])

    controller = PurePursuitPlusPID()
    cross_track_list = []
    fps_list = []

    try:
        CARLA_map = world.get_map()

        # create a vehicle
        blueprint_library = world.get_blueprint_library()
        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        spawn_point = random.choice(CARLA_map.get_spawn_points())

        vehicle = world.spawn_actor(veh_bp, spawn_point)
        actor_list.append(vehicle)

        startPoint = carla_vec_to_np_array(spawn_point.location)

        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        
        # Lane Detector Model
        cg = CameraGeometry(pitch_deg=2.5)
        
        if model_type == "openvino":
            lane_detector = OpenVINOLaneDetector()
        else:
            lane_detector = None

        # Windshield cam - adjusted pitch to + degrees to better capture traffic lights
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        camera_windshield = world.spawn_actor(bp, cam_windshield_transform, attach_to=vehicle)
        actor_list.append(camera_windshield)
        sensors.append(camera_windshield)

        flag = True
        max_error = 0
        FPS = fps_sim

        logger.info("Starting simulation...")

        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                
                # Advance the simulation and wait for the data. 
                tick_response = sync_mode.tick(timeout=2.0)

                snapshot, image_rgb, image_windshield = tick_response
                try:
                    if lane_detector:
                        trajectory, img = get_trajectory_from_lane_detector(lane_detector, image_windshield)
                        use_map_fallback = False
                    else:
                        raise Exception("No lane detector")
                except:
                    trajectory = get_trajectory_from_map(CARLA_map, vehicle)
                    img_array = carla_img_to_array(image_windshield)
                    # Convert RGB to BGR for OpenCV display
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img_array, (600, 400))
                    use_map_fallback = True

                # Object detection
                detections = []
                if yolo_detector:
                    try:
                        img_array = carla_img_to_array(image_windshield)
                        detections = yolo_detector.detect(img_array)
                        
                        # Simple traffic light color detection
                        if traffic_light_detector and detections:
                            traffic_lights = traffic_light_detector.analyze_traffic_lights(img_array, detections)
                            for i, detection in enumerate(detections):
                                if detection.get('class_id') == 9:  # Traffic light
                                    for tl in traffic_lights:
                                        if tl.get('id') == detection.get('id'):
                                            detections[i]['traffic_light_state'] = tl.get('traffic_light_state', 'unknown')
                                            break
                    except Exception as e:
                        logger.warning(f"Object detection error: {e}")
                        detections = []

                max_curvature = get_curvature(np.array(trajectory))
                if max_curvature > 0.005 and flag == False:
                    move_speed = np.abs(5.56 - 20*max_curvature)
                    move_speed = max(move_speed, 3.0)
                else:
                    move_speed = 5.56

                speed = np.linalg.norm(carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./FPS)
                
                # Traffic light control - only consider front-facing traffic lights
                brake = 0
                img_width = carla_img_to_array(image_windshield).shape[1]
                for detection in detections:
                    if detection.get('class_id') == 9 and 'traffic_light_state' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        center_x = (x1 + x2) / 2
                        # Only consider traffic lights in the front center area (middle 40% of image width)
                        if 0.3 * img_width <= center_x <= 0.7 * img_width:
                            state = detection['traffic_light_state']
                            if state in ['red', 'yellow']:
                                throttle, brake = 0, 1
                                break
                
                send_control(vehicle, throttle, steer, brake)

                dist = dist_point_linestring(np.array([0,0]), trajectory)
                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                    
                waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)

                if np.linalg.norm(vehicle_loc-startPoint) > 20:
                    flag = False

                if np.linalg.norm(vehicle_loc-startPoint) < 20 and flag == False:
                    logger.info('Route completed!')
                    break
                
                if speed < 1 and flag == False:
                    logger.warning("Vehicle stopped!")
                    # break

                # LLM Explanation
                if explainer and len(cross_track_list) % 30 == 0:  # Every 30 frames
                    try:
                        explanation_input = {
                            "vehicle_state": {
                                "speed": speed * 3.6,
                                "control": {"throttle": max(0, throttle), "steer": steer, "brake": max(0, -throttle)}
                            },
                            "perception": {
                                "trajectory_curvature": max_curvature,
                                "cross_track_error": cross_track_error
                            },
                            "detections": detections
                        }
                        explanation = explainer.explain(explanation_input)
                        logger.info(f"LLM: {explanation}")
                    except Exception as e:
                        logger.error(f"LLM error: {e}")

                # Visualization
                fontText = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.75
                fontColor = (255,255,255)
                lineType = 2
                
                # Draw object detection results on the image
                if detections:
                    # Scale detection coordinates to OpenCV window size (600x400)
                    original_height, original_width = carla_img_to_array(image_windshield).shape[:2]
                    img_height, img_width = img.shape[:2]
                    scale_x = img_width / original_width
                    scale_y = img_height / original_height
                    
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
                    
                    img = draw_detections_opencv(img, scaled_detections)
                
                if dist < 0.75:
                    laneMessage = "Lane Tracking: Good"
                else:
                    laneMessage = "Lane Tracking: Bad"

                cv2.putText(img, laneMessage, (350,50), fontText, fontScale, fontColor, lineType)

                if steer > 0:
                    steerMessage = "Right"
                else:
                    steerMessage = "Left"

                cv2.putText(img, "Steering: {}".format(steerMessage), (400,90), fontText, fontScale, fontColor, lineType)
                cv2.putText(img, "X: {:.2f}, Y: {:.2f}".format(vehicle_loc[0], vehicle_loc[1]), (20,50), fontText, 0.5, fontColor, lineType)
                cv2.putText(img, f"Objects: {len(detections)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fontColor, lineType)

                cv2.imshow('Lane detect', img)
                cv2.waitKey(1)

                fps_list.append(clock.get_fps())

                # Draw the pygame display (clean, no detection boxes)
                draw_image(display, image_rgb)
                display.blit(font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)), (6, 4))
                
                pygame.display.flip()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info('Destroying actors...')
        for actor in actor_list:
            try:
                actor.destroy()
            except:
                pass
        
        if cross_track_list:
            logger.info(f'Mean cross track error: {np.mean(cross_track_list):.2f}')
        if fps_list:
            logger.info(f'Mean FPS: {np.mean(fps_list):.2f}')
            
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--map', default='1') 
    parser.add_argument('--weather', type=int, default=0)
    parser.add_argument('--show-map', action='store_true')
    parser.add_argument('--model', default='openvino')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO object detection')
    args = parser.parse_args()
    
    main(args.fps, args.map, args.weather, args.show_map, args.model, not args.no_llm, not args.no_yolo) 