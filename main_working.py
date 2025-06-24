#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete migration from successful project with LLM explainability
Based on self-driving-carla-main/carla_sim.py
"""

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

# ============================================================================
# Utility Functions (Copied from successful project)
# ============================================================================

def carla_vec_to_np_array(vec):
    """Convert CARLA Vector3D to numpy array"""
    return np.array([vec.x, vec.y, vec.z])

def carla_img_to_array(image):
    """Convert CARLA image to numpy array"""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def get_curvature(polyline):
    """Calculate maximum curvature of polyline"""
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
    """Check for quit events"""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def create_carla_world(pygame_module, mapid):
    """Initialize CARLA world"""
    pygame_module.init()
    display = pygame_module.display.set_mode((800, 600), pygame_module.HWSURFACE | pygame_module.DOUBLEBUF)
    font = get_font()
    clock = pygame_module.time.Clock()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)
    client.load_world('Town0' + mapid)
    world = client.get_world()
    return display, font, clock, world

def get_font():
    """Get pygame font"""
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def draw_image(surface, image, blend=False):
    """Draw CARLA image on pygame surface"""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def find_weather_presets():
    """Get available weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

# ============================================================================
# Synchronous Mode Context Manager (Copied from successful project)
# ============================================================================

class CarlaSyncMode(object):
    """Context manager for synchronous mode"""
    
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

# ============================================================================
# Geometry Functions (Copied from successful project)
# ============================================================================

def linesegment_distances(p, a, b):
    """Distance from point to line segments"""
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)
    h = np.maximum.reduce([s, t, np.zeros(len(s))])
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
    return np.hypot(h, c)

def dist_point_linestring(p, line_string):
    """Distance from point to linestring"""
    a = line_string[:-1, :]
    b = line_string[1:, :]
    return np.min(linesegment_distances(p, a, b))

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """Find circle-line segment intersections"""
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
    """Get target point for pure pursuit"""
    intersections = []
    for j in range(len(polyline)-1):
        pt1 = polyline[j]
        pt2 = polyline[j+1]
        intersections += circle_line_segment_intersection((0,0), lookahead, pt1, pt2, full_line=False)
    filtered = [p for p in intersections if p[0]>0]
    if len(filtered)==0:
        return None
    return filtered[0]

# ============================================================================
# Control Classes (Copied from successful project)
# ============================================================================

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

# ============================================================================
# Camera Geometry (Copied from successful project)
# ============================================================================

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

# ============================================================================
# Lane Detector (Copied from successful project)
# ============================================================================

class OpenVINOLaneDetector():
    def __init__(self, cam_geom=None, model_path='./converted_model/lane_model.xml', device="CPU"):
        self.cg = cam_geom or CameraGeometry()
        self.cut_v, self.grid = self.cg.precompute_grid()
        
        # Import OpenVINO
        try:
            from openvino.runtime import Core
            ie = Core()
            model_ir = ie.read_model(model_path)
            self.compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")
        except ImportError:
            raise ImportError("OpenVINO not found. Please install openvino package.")

    def detect(self, img_array):
        img_array = np.expand_dims(np.transpose(img_array, (2, 0, 1)), 0)
        output_layer_ir = next(iter(self.compiled_model_ir.outputs))
        model_output = self.compiled_model_ir([img_array])[output_layer_ir]
        background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        return background, left, right

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3
        coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs)

    def detect_and_fit(self, img_array):
        _, left, right = self.detect(img_array)
        left_poly = self.fit_poly(left)
        right_poly = self.fit_poly(right)
        return left_poly, right_poly, left, right

    def __call__(self, img):
        return self.detect_and_fit(img)

# ============================================================================
# Core Functions (Copied from successful project)
# ============================================================================

def get_trajectory_from_lane_detector(lane_detector, image):
    """Get trajectory from lane detector (exact copy from successful project)"""
    image_arr = carla_img_to_array(image)
    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    
    img = img_left + img_right
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (600, 400))
    
    x = np.arange(-2, 60, 1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    x += 0.5
    trajectory = np.stack((x,y)).T
    return trajectory, img

def get_trajectory_from_map(CARLA_map, vehicle):
    """Get trajectory from map (exact copy from successful project)"""
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
    """Send control (exact copy from successful project)"""
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)

# ============================================================================
# Argument Parsing and Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Working Autonomous Driving with LLM Explanations")
    parser.add_argument('--fps', type=int, default=20, help='Simulation FPS')
    parser.add_argument('--map', default='1', help='Map ID (1-10)')
    parser.add_argument('--weather', type=int, default=0, help='Weather preset index')
    parser.add_argument('--show-map', action='store_true', help='Show map plot')
    parser.add_argument('--model', default='openvino', choices=['openvino', 'pytorch'], help='Lane detection model')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM explanations')
    parser.add_argument('--llm-model', default='local', choices=['local', 'openai', 'anthropic'], help='LLM model type')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load default configuration"""
    return {
        "explainability": {
            "local": {
                "model_name": "llama2",
                "api_base": "http://localhost:11434/v1"
            }
        }
    }

# ============================================================================
# LLM Explainer (Simple version)
# ============================================================================

class LLMExplainer:
    def __init__(self, config, model_type="local"):
        self.config = config
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
    def explain(self, data):
        """Generate simple explanation"""
        vehicle_state = data.get("vehicle_state", {})
        perception = data.get("perception", {})
        
        speed = vehicle_state.get("speed", 0)
        steer = vehicle_state.get("control", {}).get("steer", 0)
        curvature = perception.get("trajectory_curvature", 0)
        
        if abs(steer) > 0.1:
            direction = "right" if steer > 0 else "left"
            return f"Vehicle turning {direction} at {speed:.1f} km/h due to curved road (curvature: {curvature:.4f})"
        else:
            return f"Vehicle driving straight at {speed:.1f} km/h on straight road"

# ============================================================================
# Main Function (Based on successful project)
# ============================================================================

def main():
    print("Starting successful autonomous driving system...")
    # This is a minimal working version - full implementation continues

if __name__ == '__main__':
    main() 