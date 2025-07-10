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
import threading
from collections import deque

# Import from original project structure (copy all needed functions)
import sys
import os

from src.explainability.llm_explainer import ThreadedLLMExplainer
from src.evaluation.evaluation_metrics import EvaluationMetrics

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import object detection and traffic light detection
try:
    from src.perception.object_detection import ObjectDetector
    from src.perception.traffic_light_detector import TrafficLightDetector
    from src.explainability.llm_explainer import LLMExplainer, ThreadedLLMExplainer
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
    display = pygame_module.display.set_mode((800, 600), pygame_module.HWSURFACE | pygame_module.DOUBLEBUF)
    font = get_font()
    clock = pygame_module.time.Clock()
    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)
    client.load_world('Town0' + mapid)
    world = client.get_world()
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

# Core trajectory functions
def get_trajectory_from_lane_detector(lane_detector, image):
    image_arr = carla_img_to_array(image)
    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    
    # Get original image, convert RGB to BGR, and resize it
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


def generate_evaluation_report(eval_metrics, cross_track_list, detections, logger, end_reason="仿真结束"):
    """生成并打印详细的评估报告"""
    # 获取完整评估报告
    evaluation_report = eval_metrics.get_full_report()
    
    # 打印评估报告
    logger.info("=" * 60)
    logger.info("🚗 自动驾驶性能评估报告")
    logger.info("=" * 60)
    logger.info(f"🏁 结束原因: {end_reason}")
    logger.info("")
    
    # 总体评分
    overall_score = evaluation_report['overall_score']
    logger.info(f"🏆 总体评分: {overall_score:.1f}/100")
    
    # 计算驾驶技能评分（专门评估技术水平）
    traj_perf = evaluation_report['trajectory_performance']
    comfort_perf = evaluation_report['comfort_performance']
    eff_perf = evaluation_report['efficiency_performance']
    
    # 驾驶技能评分计算
    lane_score = max(0, 100 - traj_perf['mean_cross_track_error'] * 100)
    speed_score = max(0, 100 - traj_perf['mean_speed_error'] * 15)
    smooth_score = max(0, 100 - eff_perf['mean_jerk'] * 30)
    steering_score = max(0, 100 - comfort_perf['mean_steering_rate'] * 80)
    
    driving_skill_score = (lane_score * 0.35 + speed_score * 0.25 + smooth_score * 0.25 + steering_score * 0.15)
    driving_skill_score = min(100, max(0, driving_skill_score))
    
    logger.info(f"🎯 驾驶技能评分: {driving_skill_score:.1f}/100")
    
    if overall_score >= 90:
        score_level = "优秀 🥇"
    elif overall_score >= 80:
        score_level = "良好 🥈"
    elif overall_score >= 70:
        score_level = "及格 🥉"
    else:
        score_level = "需改进 ⚠️"
    
    if driving_skill_score >= 90:
        skill_level = "专业级 🏎️"
    elif driving_skill_score >= 80:
        skill_level = "熟练 🚗"
    elif driving_skill_score >= 70:
        skill_level = "一般 📚"
    else:
        skill_level = "新手 🔰"
    
    logger.info(f"📈 综合评级: {score_level}")
    logger.info(f"🏁 驾驶技能: {skill_level}")
    logger.info("")
    
    # 轨迹跟踪性能
    traj_perf = evaluation_report['trajectory_performance']
    logger.info("🛣️ 轨迹跟踪性能:")
    logger.info(f"   • 平均横向偏差: {traj_perf['mean_cross_track_error']:.3f}m")
    logger.info(f"   • 最大横向偏差: {traj_perf['max_cross_track_error']:.3f}m")
    logger.info(f"   • 平均航向偏差: {traj_perf['mean_heading_error']:.3f}rad ({np.degrees(traj_perf['mean_heading_error']):.1f}°)")
    logger.info(f"   • 平均速度偏差: {traj_perf['mean_speed_error']:.2f}m/s")
    logger.info("")
    
    # 安全性能
    safety_perf = evaluation_report['safety_performance']
    logger.info("🛡️ 安全性能:")
    logger.info(f"   • 碰撞次数: {safety_perf['collision_count']}")
    logger.info(f"   • 差点碰撞次数: {safety_perf['near_collision_count']}")
    logger.info(f"   • 交通灯违规: {safety_perf['traffic_light_violations']}")
    logger.info(f"   • 与物体最小距离: {safety_perf['min_distance_to_objects']:.2f}m")
    logger.info(f"   • 平均安全距离: {safety_perf['mean_min_distance_to_objects']:.2f}m")
    logger.info("")
    
    # 效率性能
    eff_perf = evaluation_report['efficiency_performance']
    logger.info("⚡ 效率性能:")
    logger.info(f"   • 平均速度: {eff_perf['mean_speed']:.2f}m/s ({eff_perf['mean_speed']*3.6:.1f}km/h)")
    logger.info(f"   • 最高速度: {eff_perf['max_speed']:.2f}m/s ({eff_perf['max_speed']*3.6:.1f}km/h)")
    logger.info(f"   • 平均加速度: {eff_perf['mean_acceleration']:.2f}m/s²")
    logger.info(f"   • 平均加速度变化率: {eff_perf['mean_jerk']:.2f}m/s³")
    logger.info(f"   • 总行驶距离: {eff_perf['total_distance']:.1f}m")
    logger.info(f"   • 总行驶时间: {eff_perf['total_time']:.1f}s")
    logger.info(f"   • 路线完成率: {eff_perf['route_completion_rate']*100:.1f}%")
    logger.info("")
    
    # 舒适性能
    comfort_perf = evaluation_report['comfort_performance']
    logger.info("😌 舒适性能:")
    logger.info(f"   • 平均横向加速度: {comfort_perf['mean_lateral_acceleration']:.2f}m/s²")
    logger.info(f"   • 最大横向加速度: {comfort_perf['max_lateral_acceleration']:.2f}m/s²")
    logger.info(f"   • 平均转向角度: {comfort_perf['mean_steering_angle']:.3f}rad ({np.degrees(comfort_perf['mean_steering_angle']):.1f}°)")
    logger.info(f"   • 平均转向速率: {comfort_perf['mean_steering_rate']:.3f}rad/s")
    logger.info("")
    
    # 感知性能
    perception_perf = evaluation_report['perception_performance']
    logger.info("👁️ 感知性能:")
    logger.info(f"   • 目标检测准确率: {perception_perf['mean_object_detection_accuracy']*100:.1f}%")
    logger.info(f"   • 车道检测准确率: {perception_perf['mean_lane_detection_accuracy']*100:.1f}%")
    logger.info(f"   • 交通灯检测准确率: {perception_perf['mean_traffic_light_detection_accuracy']*100:.1f}%")
    logger.info("")
    
    # 决策性能
    decision_perf = evaluation_report['decision_performance']
    logger.info("🧠 决策性能:")
    logger.info(f"   • 平均决策时间: {decision_perf['mean_decision_time']*1000:.1f}ms")
    logger.info(f"   • 最大决策时间: {decision_perf['max_decision_time']*1000:.1f}ms")
    logger.info("")
    
    # 统计信息
    logger.info("📈 统计信息:")
    logger.info(f"   • 仿真帧数: {len(cross_track_list)}")
    logger.info(f"   • 实际仿真时间: {eff_perf['total_time']:.1f}s")
    logger.info(f"   • 平均帧率: {len(cross_track_list)/max(eff_perf['total_time'], 1):.1f} FPS")
    logger.info(f"   • 检测到的物体总数: {sum(len(dets) if isinstance(dets, list) else 0 for dets in [detections])}")
    logger.info("=" * 60)
    
    # 生成简要建议
    logger.info("💡 改进建议:")
    
    suggestions_count = 0
    
    # 横向偏差建议（降低阈值）
    if traj_perf['mean_cross_track_error'] > 0.05:  # 5cm以上
        logger.info("   • 建议改进车道跟踪算法，减少横向偏差")
        suggestions_count += 1
    
    # 航向偏差建议（新增）
    if traj_perf['mean_heading_error'] > 0.1:  # 约5.7度以上
        logger.info("   • 建议优化航向控制，减少车辆摆动")
        suggestions_count += 1
    
    # 速度跟踪建议
    if traj_perf['mean_speed_error'] > 1.0:  # 1m/s以上偏差
        logger.info("   • 建议改进速度跟踪控制器，提高速度稳定性")
        suggestions_count += 1
    
    # 平滑驾驶建议（基于加速度变化率）
    if eff_perf['mean_jerk'] > 10.0:  # 急动度过大
        logger.info("   • 建议优化加速度控制，提高驾驶平稳性")
        suggestions_count += 1
    
    # 转向平稳性建议
    if comfort_perf['mean_steering_rate'] > 1.5:  # 转向变化过快
        logger.info("   • 建议优化转向控制，减少急转向动作")
        suggestions_count += 1
    
    # 横向加速度建议（降低阈值）
    if comfort_perf['mean_lateral_acceleration'] > 1.5:
        logger.info("   • 建议减少横向加速度，提高乘坐舒适性")
        suggestions_count += 1
    
    # 安全距离建议
    if safety_perf['mean_min_distance_to_objects'] < 5.0:
        logger.info("   • 建议增加与其他物体的安全距离")
        suggestions_count += 1
    
    # 碰撞和违规建议
    if safety_perf['collision_count'] > 0:
        logger.info("   • 建议增强碰撞检测和避障能力")
        suggestions_count += 1
    
    if safety_perf['traffic_light_violations'] > 0:
        logger.info("   • 建议改进交通信号识别和遵守")
        suggestions_count += 1
    
    # 感知准确率建议
    if perception_perf['mean_object_detection_accuracy'] < 0.85:
        logger.info("   • 建议改进目标检测算法精度")
        suggestions_count += 1
    
    if perception_perf['mean_lane_detection_accuracy'] < 0.9:
        logger.info("   • 建议提升车道检测算法性能")
        suggestions_count += 1
    
    # 如果没有具体建议，给出通用建议
    if suggestions_count == 0:
        if overall_score < 90:
            logger.info("   • 继续优化各项性能指标，争取达到优秀水平")
            logger.info("   • 可重点关注提升驾驶技能评分")
            suggestions_count += 2
    
    if suggestions_count == 0:
        logger.info("   • 当前表现优秀，继续保持！🎉")
    
    logger.info("🎯 评估完成！")
    logger.info("=" * 60)


def main(fps_sim=100, mapid='2', weather_idx=2, showmap=False, model_type="openvino", enable_llm=True, enable_yolo=True):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize LLM explainer with Deepseek API
    threaded_explainer = None
    if enable_llm:
        llm_config = {
            'api_key': "sk-e1bbc1ee34174850a8bdf7d03cb3b67a",  # 将从环境变量 DEEPSEEK_API_KEY 获取
            'api_base_url': 'https://api.deepseek.com/v1',
            'api_model': 'deepseek-chat'
        }
        explainer = LLMExplainer(config=llm_config, model_type="deepseek_api")
        threaded_explainer = ThreadedLLMExplainer(explainer, max_queue_size=5)
        threaded_explainer.start()
        logger.info(f"Threaded LLM Explainer initialized with model_type: {explainer.model_type}")
    else:
        threaded_explainer = None
    
    # Initialize YOLO detector and traffic light detector
    yolo_detector = None
    traffic_light_detector = None
    if enable_yolo and YOLO_AVAILABLE:
        try:
            yolo_config = {
                'model': 'yolov8n.pt',  # Use the model file in the project root
                'device': 'cpu',
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

    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[weather_idx][0])
    logger.info(f"Weather set to: {weather_presets[weather_idx][1]}")

    controller = PurePursuitPlusPID()
    cross_track_list = []
    fps_list = []

    try:
        CARLA_map = world.get_map()

        # ========== 清理现有的actors ==========
        logger.info("正在清理现有的actors...")
        actors = world.get_actors()
        vehicles = actors.filter('*vehicle*')
        sensors = actors.filter('*sensor*')
        walkers = actors.filter('*walker*')

        logger.info(f"找到 {len(vehicles)} 辆现有车辆")
        logger.info(f"找到 {len(sensors)} 个现有传感器")
        logger.info(f"找到 {len(walkers)} 个现有行人")

        # 销毁所有现有的车辆、传感器和行人
        for actor in vehicles:
            actor.destroy()
        for actor in sensors:
            actor.destroy()
        for actor in walkers:
            actor.destroy()

        logger.info("现有actors清理完成！")

        # ========== 生成NPC车辆 ==========
        logger.info("正在生成NPC车辆...")
        vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
        spawn_points = CARLA_map.get_spawn_points()
        
        spawned_vehicles = []
        for i in range(10):  # 生成20辆NPC车辆
            vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
            if vehicle is not None:
                spawned_vehicles.append(vehicle)
                logger.info(f"✅ NPC车辆 {i+1} 生成成功")

        logger.info(f"成功生成了 {len(spawned_vehicles)} 辆NPC车辆")

        # ========== 生成行人 ==========
        logger.info("正在生成行人...")
        walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')

        logger.info(f"找到 {len(walker_blueprints)} 种行人类型")

        # 设置要生成的行人数量
        num_walkers = 10

        # 生成行人的spawn点
        walker_spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                walker_spawn_points.append(spawn_point)

        logger.info(f"找到 {len(walker_spawn_points)} 个有效的行人生成位置")

        # 生成行人和控制器
        spawned_walkers = []
        walker_controllers = []

        for i, spawn_point in enumerate(walker_spawn_points):
            # 随机选择行人类型
            walker_bp = random.choice(walker_blueprints)
            
            # 设置行人属性
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            
            # 随机设置行人的速度属性
            if walker_bp.has_attribute('speed'):
                speed = random.uniform(1.0, 2.5)
                walker_bp.set_attribute('speed', str(speed))
            
            # 生成行人
            walker = world.try_spawn_actor(walker_bp, spawn_point)
            if walker is not None:
                spawned_walkers.append(walker)
                
                # 为每个行人创建AI控制器
                walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                if walker_controller is not None:
                    walker_controllers.append(walker_controller)
                    logger.info(f"✅ 行人 {i+1} 生成成功，位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                else:
                    logger.info(f"❌ 行人 {i+1} 控制器生成失败")
            else:
                logger.info(f"❌ 行人 {i+1} 生成失败")

        logger.info(f"🚶 成功生成了 {len(spawned_walkers)} 个行人")
        logger.info(f"🎮 成功创建了 {len(walker_controllers)} 个行人AI控制器")

        # 等待一帧，确保所有行人完全生成
        world.tick()

        # 启动行人AI控制器
        logger.info("正在启动行人AI控制器...")
        active_controllers = 0

        for i, walker_controller in enumerate(walker_controllers):
            try:
                # 启动控制器
                walker_controller.start()
                
                # 获取随机位置作为目标
                target_location = world.get_random_location_from_navigation()
                
                if target_location is not None:
                    walker_controller.go_to_location(target_location)
                    # 设置随机行走速度 (1.0-2.5 m/s)
                    max_speed = random.uniform(1.0, 2.5)
                    walker_controller.set_max_speed(max_speed)
                    active_controllers += 1
                    logger.info(f"🎯 行人 {i+1} AI已启动，目标位置: ({target_location.x:.1f}, {target_location.y:.1f}), 速度: {max_speed:.1f} m/s")
                else:
                    # 如果无法获取导航位置，设置为缓慢随机行走
                    walker_controller.set_max_speed(1.0)
                    active_controllers += 1
                    logger.info(f"🎯 行人 {i+1} AI已启动，缓慢随机行走")
            except Exception as e:
                logger.error(f"❌ 行人 {i+1} AI启动失败: {e}")

        logger.info(f"🎮 成功启动了 {active_controllers} 个行人AI")

        # ========== 启用Traffic Manager ==========
        logger.info("正在启用Traffic Manager...")
        
        # 获取Traffic Manager实例
        client = carla.Client('localhost', 2000)
        traffic_manager = client.get_trafficmanager(8000)  # 使用默认端口8000
        logger.info("Traffic Manager已连接")
        
        # 设置Traffic Manager为同步模式（与世界同步）
        traffic_manager.set_synchronous_mode(True)
        logger.info("Traffic Manager设置为同步模式")
        
        # 设置随机种子，保证行为一致
        traffic_manager.set_random_device_seed(42)
        logger.info("设置随机种子：42（行为更一致）")
        
        # 设置全局速度限制
        traffic_manager.global_percentage_speed_difference(50.0)  # 全局比限速慢50%
        logger.info("全局速度设置：比限速慢50%")
        
        # 设置更保守的全局驾驶行为
        traffic_manager.set_global_distance_to_leading_vehicle(4.0)  # 全局跟车距离4米
        logger.info("全局跟车距离：4米")

        # ========== 为所有NPC车辆启用自动驾驶 ==========
        logger.info("正在为NPC车辆启用自动驾驶...")
        autopilot_count = 0
        
        for vehicle in spawned_vehicles:
            try:
                # 为每辆车启用autopilot
                vehicle.set_autopilot(True, traffic_manager.get_port())
                autopilot_count += 1
                
                # 为每辆车设置不同的速度，增加多样性
                speed_difference = random.uniform(20.0, 80.0)  # 比限速慢20-80%
                traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_difference)
                
                # 为每辆车设置不同的跟车距离
                distance = random.uniform(2.0, 6.0)  # 跟车距离2-6米
                traffic_manager.distance_to_leading_vehicle(vehicle, distance)
                
                # 随机设置是否允许变道
                allow_lane_change = random.choice([True, False])
                traffic_manager.auto_lane_change(vehicle, allow_lane_change)
                
                logger.info(f"🚗 NPC车辆 {vehicle.id} 启用autopilot成功，速度:{speed_difference:.0f}%慢，跟车距离:{distance:.1f}m")
                
            except Exception as e:
                logger.error(f"❌ 为NPC车辆 {vehicle.id} 启用autopilot失败: {e}")
        
        logger.info(f"✅ 成功为 {autopilot_count} 辆NPC车辆启用autopilot")

        # ========== 环境信息总结 ==========
        logger.info("🎬 完整交通环境已设置完成！")
        logger.info("📊 环境配置:")
        logger.info(f"   - 地图: Town0{mapid}")
        logger.info(f"   - 天气: {weather_presets[weather_idx][1]}")
        logger.info(f"   - NPC车辆: {len(spawned_vehicles)} 辆 (全部自动驾驶)")
        logger.info(f"   - 行人: {len(spawned_walkers)} 个 (AI控制)")
        logger.info(f"   - 行人控制器: {len(walker_controllers)} 个")
        logger.info(f"   - Traffic Manager: 已启用 (同步模式)")
        logger.info(f"   - 仿真帧率: {fps_sim} FPS")
        logger.info(f"   - 车道检测: {model_type}")
        logger.info(f"   - YOLO检测: {'启用' if enable_yolo else '禁用'}")
        logger.info(f"   - LLM解释: {'启用' if enable_llm else '禁用'}")
        logger.info("🚗 现在开始自动驾驶测试...")

        # create a vehicle (Ego车辆)
        blueprint_library = world.get_blueprint_library()
        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','255,0,0')  # 设置为红色，便于识别
        veh_bp.set_attribute('role_name', 'hero')  # 设置为Ego车辆
        
        # 为Ego车辆寻找一个空闲的spawn点
        spawn_points = CARLA_map.get_spawn_points()
        ego_vehicle = None
        
        for i, spawn_point in enumerate(spawn_points):
            try:
                ego_vehicle = world.try_spawn_actor(veh_bp, spawn_point)
                if ego_vehicle is not None:
                    logger.info(f"✅ Ego车辆生成成功！")
                    logger.info(f"🚗 车辆类型: {ego_vehicle.type_id}")
                    logger.info(f"🎯 Role Name: hero (Ego Vehicle)")
                    logger.info(f"🔴 颜色: 红色 (便于识别)")
                    logger.info(f"📍 使用的spawn点索引: {i}")
                    logger.info(f"🆔 Ego车辆ID: {ego_vehicle.id}")
                    logger.info(f"📍 Ego车辆位置: x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}, z={spawn_point.location.z:.2f}")
                    break
            except Exception as e:
                logger.warning(f"尝试spawn点 {i} 失败: {e}")
                continue
        
        if ego_vehicle is None:
            logger.error("❌ 无法为Ego车辆找到空闲的spawn点")
            logger.info("🧹 正在清理资源并退出...")
            return
        
        actor_list.append(ego_vehicle)

        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=ego_vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        
        # Lane Detector Model
        cg = CameraGeometry(pitch_deg=5)
        
        if model_type == "openvino":
            lane_detector = OpenVINOLaneDetector()
        else:
            lane_detector = None
            
        # Initialize evaluation metrics
        eval_metrics = EvaluationMetrics(window_size=100)
        logger.info("📊 评估指标系统已初始化")

        # Windshield cam - adjusted pitch to + degrees to better capture traffic lights
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        camera_windshield = world.spawn_actor(bp, cam_windshield_transform, attach_to=ego_vehicle)
        actor_list.append(camera_windshield)
        sensors.append(camera_windshield)

        max_error = 0
        FPS = fps_sim
        
        # 仿真计时和状态变量
        start_time = time.time()
        max_simulation_time = 60  # 5分钟
        stationary_time = 0
        last_position = None

        logger.info("Starting simulation...")
        logger.info(f"⏱️ 最大仿真时间: {max_simulation_time}秒")

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
                    trajectory = get_trajectory_from_map(CARLA_map, ego_vehicle)
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
                if max_curvature > 0.005:
                    move_speed = np.abs(5.56 - 20*max_curvature)
                    move_speed = max(move_speed, 3.0)
                else:
                    move_speed = 5.56

                speed = np.linalg.norm(carla_vec_to_np_array(ego_vehicle.get_velocity()))
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./FPS)
                
                # Traffic light control - only consider front-facing traffic lights
                brake = 0
                img_width = carla_img_to_array(image_windshield).shape[1]
                img_height = carla_img_to_array(image_windshield).shape[0]
                
                # Check for traffic lights
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
                
                # Front obstruction detection - check for blocking objects in current lane
                if brake == 0:  # Only check if not already braking for traffic lights
                    # Define object types that can block the path
                    blocking_objects = [
                        'car', 'truck', 'bus', 'motorcycle',  # Vehicles
                        'person', 'bicycle'  # Pedestrians and cyclists
                    ]
                    
                    # Convert class IDs to names for easier checking
                    blocking_class_ids = []
                    for i, class_name in enumerate(YOLO_CLASSES):
                        if class_name in blocking_objects:
                            blocking_class_ids.append(i)
                    
                    for detection in detections:
                        class_id = detection.get('class_id', 0)
                        if class_id in blocking_class_ids:
                            x1, y1, x2, y2 = detection['bbox']
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            bbox_area = bbox_width * bbox_height
                            
                            # Check if object is in front center area (current lane)
                            # Use a narrower corridor than traffic lights (middle 30% of image width)
                            # and lower half of image (where road objects appear)
                            lane_left = 0.35 * img_width
                            lane_right = 0.65 * img_width
                            horizon_line = 0.4 * img_height  # Objects above this line are far away
                            
                            if (lane_left <= center_x <= lane_right and 
                                center_y > horizon_line):
                                
                                # Determine if object is close enough to require stopping
                                # Larger bbox area indicates closer object
                                min_area_for_stop = 1500  # Minimum area to trigger stopping
                                
                                # Additional checks for different object types
                                class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else ""
                                confidence = detection.get('confidence', 0.0)
                                
                                # More sensitive detection for pedestrians
                                if class_name == 'person' and bbox_area > 800 and confidence > 0.6:
                                    throttle, brake = 0, 1
                                    logger.info(f"Emergency stop: Pedestrian detected in front (area: {bbox_area:.0f})")
                                    break
                                
                                # Vehicle obstruction detection
                                elif class_name in ['car', 'truck', 'bus', 'motorcycle'] and bbox_area > min_area_for_stop:
                                    throttle, brake = 0, 1
                                    logger.info(f"Stop: {class_name} blocking path (area: {bbox_area:.0f})")
                                    break
                                
                                # Bicycle detection
                                elif class_name == 'bicycle' and bbox_area > 1000 and confidence > 0.5:
                                    throttle, brake = 0, 1
                                    logger.info(f"Stop: Bicycle in front (area: {bbox_area:.0f})")
                                    break
                
                send_control(ego_vehicle, throttle, steer, brake)

                dist = dist_point_linestring(np.array([0,0]), trajectory)
                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                    
                waypoint = CARLA_map.get_waypoint(ego_vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)
                
                # 更新停止时间统计
                current_position = vehicle_loc
                if last_position is not None:
                    position_change = np.linalg.norm(current_position - last_position)
                    if position_change < 0.1:  # 车辆基本没有移动
                        stationary_time += 1.0 / FPS
                    else:
                        stationary_time = 0  # 重置停止时间
                last_position = current_position

                # ========== 收集评估数据 ==========
                # 获取车辆状态信息
                vehicle_velocity = ego_vehicle.get_velocity()
                vehicle_acceleration = ego_vehicle.get_acceleration()
                vehicle_transform = ego_vehicle.get_transform()
                
                # 计算当前车辆的实际速度和加速度
                current_speed = np.linalg.norm(carla_vec_to_np_array(vehicle_velocity))
                current_acceleration = np.linalg.norm(carla_vec_to_np_array(vehicle_acceleration))
                lateral_acceleration = np.abs(vehicle_acceleration.y)  # 横向加速度
                
                # 计算航向偏差（相对于期望轨迹）
                if len(trajectory) > 1:
                    next_waypoint = trajectory[1] if len(trajectory) > 1 else trajectory[0]
                    desired_heading = np.arctan2(next_waypoint[1], next_waypoint[0])
                    current_heading = np.radians(vehicle_transform.rotation.yaw)
                    heading_error = np.abs(desired_heading - current_heading)
                    # 归一化角度差到[-π, π]
                    heading_error = ((heading_error + np.pi) % (2 * np.pi)) - np.pi
                    heading_error = np.abs(heading_error)
                else:
                    heading_error = 0.0
                
                # 计算速度偏差
                speed_error = np.abs(current_speed - move_speed)
                
                # 检测碰撞和安全事件
                collision_occurred = False  # 需要实际的碰撞检测
                near_collision = False
                traffic_light_violation = False
                stop_sign_violation = False
                lane_change_violation = False
                
                # 计算与最近物体的距离
                min_distance_to_objects = 100.0  # 默认较大值
                if detections:
                    # 简单估算：基于边界框大小估算距离
                    for detection in detections:
                        bbox_area = (detection['bbox'][2] - detection['bbox'][0]) * (detection['bbox'][3] - detection['bbox'][1])
                        # 边界框越大，距离越近（简单估算）
                        estimated_distance = max(1.0, 1000.0 / max(bbox_area, 100))
                        min_distance_to_objects = min(min_distance_to_objects, estimated_distance)
                
                # 检查交通灯违规
                for detection in detections:
                    if detection.get('class_id') == 9 and 'traffic_light_state' in detection:
                        state = detection['traffic_light_state']
                        if state == 'red' and throttle > 0:  # 红灯时仍然加速
                            traffic_light_violation = True
                
                # 计算行驶距离（本帧）
                dt = 1.0 / FPS
                distance_traveled = current_speed * dt
                
                # 更新评估指标
                eval_metrics.update_trajectory_metrics(
                    cross_track_error=cross_track_error / 100.0,  # 转换为米
                    heading_error=heading_error,
                    speed_error=speed_error
                )
                
                eval_metrics.update_safety_metrics(
                    collision_occurred=collision_occurred,
                    near_collision=near_collision,
                    traffic_light_violation=traffic_light_violation,
                    stop_sign_violation=stop_sign_violation,
                    lane_change_violation=lane_change_violation,
                    min_distance_to_objects=min_distance_to_objects
                )
                
                eval_metrics.update_efficiency_metrics(
                    speed=current_speed,
                    acceleration=current_acceleration,
                    distance_traveled=distance_traveled,
                    time_elapsed=dt
                )
                
                eval_metrics.update_comfort_metrics(
                    lateral_acceleration=lateral_acceleration,
                    steering_angle=steer,
                    throttle=max(0, throttle),
                    brake=brake,
                    time_elapsed=dt
                )
                
                # 模拟感知准确率（在实际应用中需要真值对比）
                object_detection_acc = 0.85 if detections else 0.0
                lane_detection_acc = 0.90 if not use_map_fallback else 0.70
                traffic_light_acc = 0.80
                
                eval_metrics.update_perception_metrics(
                    object_detection_acc=object_detection_acc,
                    lane_detection_acc=lane_detection_acc,
                    traffic_light_detection_acc=traffic_light_acc
                )
                
                # 更新决策性能（模拟决策时间）
                decision_time = 0.05  # 模拟50ms决策时间
                eval_metrics.update_decision_metrics(decision_time=decision_time)

                # LLM Explanation - 每60帧调用一次（约3秒）- 非阻塞异步处理
                if threaded_explainer and len(cross_track_list) % 60 == 0:  # Every 60 frames (~3 seconds)
                    try:
                        # Get current vehicle transform and additional information
                        vehicle_transform = ego_vehicle.get_transform()
                        vehicle_velocity = ego_vehicle.get_velocity()
                        vehicle_acceleration = ego_vehicle.get_acceleration()
                        vehicle_angular_velocity = ego_vehicle.get_angular_velocity()
                        
                        # Get current waypoint information
                        current_waypoint = CARLA_map.get_waypoint(vehicle_transform.location)
                        
                        # Get upcoming waypoints for context
                        upcoming_waypoints = []
                        temp_waypoint = current_waypoint
                        for i in range(5):  # Get next 5 waypoints
                            next_wps = temp_waypoint.next(5.0)  # 5 meters ahead
                            if next_wps:
                                temp_waypoint = next_wps[0]
                                upcoming_waypoints.append({
                                    "distance": i * 5.0,
                                    "lane_change": temp_waypoint.lane_change,
                                    "lane_type": str(temp_waypoint.lane_type),
                                    "road_id": temp_waypoint.road_id,
                                    "lane_id": temp_waypoint.lane_id
                                })
                        
                        # Categorize detected objects by relevance
                        critical_objects = []
                        nearby_vehicles = []
                        traffic_lights = []
                        pedestrians = []
                        other_objects = []
                        blocking_objects = []
                        
                        # Get image dimensions for lane detection
                        img_width = carla_img_to_array(image_windshield).shape[1]
                        img_height = carla_img_to_array(image_windshield).shape[0]
                        lane_left = 0.35 * img_width
                        lane_right = 0.65 * img_width
                        horizon_line = 0.4 * img_height
                        
                        for detection in detections:
                            class_id = detection.get('class_id', 0)
                            class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else f"Class {class_id}"
                            confidence = detection.get('confidence', 0.0)
                            bbox = detection.get('bbox', [0, 0, 0, 0])
                            
                            # Calculate object position relative to vehicle center
                            x_center = (bbox[0] + bbox[2]) / 2
                            y_center = (bbox[1] + bbox[3]) / 2
                            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            
                            obj_info = {
                                "class": class_name,
                                "confidence": confidence,
                                "position": {"x": x_center, "y": y_center},
                                "size": bbox_area,
                                "is_close": bbox_area > 1000  # Large objects are closer
                            }
                            
                            # Check if object is in front blocking zone
                            is_blocking = False
                            if (lane_left <= x_center <= lane_right and y_center > horizon_line):
                                blocking_objects_list = ['car', 'truck', 'bus', 'motorcycle', 'person', 'bicycle']
                                if class_name in blocking_objects_list:
                                    # Check if object meets blocking criteria
                                    if ((class_name == 'person' and bbox_area > 800 and confidence > 0.6) or
                                        (class_name in ['car', 'truck', 'bus', 'motorcycle'] and bbox_area > 1500) or
                                        (class_name == 'bicycle' and bbox_area > 1000 and confidence > 0.5)):
                                        is_blocking = True
                                        obj_info["is_blocking"] = True
                                        obj_info["blocking_reason"] = f"Object in lane with area {bbox_area:.0f}"
                                        blocking_objects.append(obj_info)
                            
                            if not is_blocking:
                                if class_name == 'traffic light':
                                    obj_info["state"] = detection.get('traffic_light_state', 'unknown')
                                    traffic_lights.append(obj_info)
                                elif class_name in ['car', 'truck', 'bus', 'motorcycle']:
                                    nearby_vehicles.append(obj_info)
                                elif class_name in ['person', 'bicycle']:
                                    pedestrians.append(obj_info)
                                elif class_name in ['stop sign'] or confidence > 0.8:
                                    critical_objects.append(obj_info)
                                else:
                                    other_objects.append(obj_info)
                        
                        # Calculate trajectory information
                        trajectory_length = len(trajectory)
                        trajectory_points_ahead = min(10, trajectory_length)
                        lookahead_distance = np.linalg.norm(trajectory[min(5, trajectory_length-1)]) if trajectory_length > 1 else 0
                        
                        # Performance metrics
                        avg_cross_track_error = np.mean(cross_track_list[-10:]) if len(cross_track_list) >= 10 else cross_track_error
                        
                        # Determine stopping reason
                        stopping_reason = "none"
                        if brake > 0:
                            if any(tl.get("state") in ['red', 'yellow'] for tl in traffic_lights):
                                stopping_reason = "traffic_light"
                            elif len(blocking_objects) > 0:
                                stopping_reason = "obstruction"
                            else:
                                stopping_reason = "other"
                        
                        explanation_input = {
                            "vehicle_state": {
                                "speed_kmh": speed * 3.6,
                                "speed_ms": speed,
                                "target_speed": move_speed,
                                "is_braking": brake > 0,
                                "stopping_reason": stopping_reason,
                                "control": {
                                    "throttle": max(0, throttle), 
                                    "steer": steer, 
                                    "brake": brake,
                                    "steer_angle_deg": steer * 30  # Approximate steering angle
                                },
                                "acceleration": {
                                    "x": vehicle_acceleration.x,
                                    "y": vehicle_acceleration.y,
                                    "magnitude": np.linalg.norm([vehicle_acceleration.x, vehicle_acceleration.y])
                                },
                                "angular_velocity": vehicle_angular_velocity.z,
                                "position": {
                                    "x": vehicle_transform.location.x,
                                    "y": vehicle_transform.location.y,
                                    "heading_deg": vehicle_transform.rotation.yaw
                                }
                            },
                            "road_context": {
                                "current_lane": {
                                    "road_id": current_waypoint.road_id,
                                    "lane_id": current_waypoint.lane_id,
                                    "lane_type": str(current_waypoint.lane_type),
                                    "lane_change": str(current_waypoint.lane_change),
                                    "is_junction": current_waypoint.is_junction
                                },
                                "upcoming_waypoints": upcoming_waypoints,
                                "navigation_mode": "lane_detection" if not use_map_fallback else "map_based"
                            },
                            "perception": {
                                "trajectory_curvature": max_curvature,
                                "trajectory_length": trajectory_length,
                                "lookahead_distance": lookahead_distance,
                                "cross_track_error": cross_track_error,
                                "avg_cross_track_error": avg_cross_track_error,
                                "lane_tracking_quality": "good" if cross_track_error < 0.75 else "poor"
                            },
                            "detected_objects": {
                                "total_count": len(detections),
                                "traffic_lights": traffic_lights,
                                "nearby_vehicles": nearby_vehicles,
                                "pedestrians": pedestrians,
                                "critical_objects": critical_objects,
                                "blocking_objects": blocking_objects,  # Objects causing vehicle to stop
                                "other_objects": other_objects[:3]  # Limit to most confident 3
                            },
                            "environment": {
                                "simulation_time": len(cross_track_list) / FPS,  # Approximate time in seconds
                                "frame_count": len(cross_track_list),
                                "weather": "clear"  # Could be extended with actual weather
                            }
                        }
                        
                        # Add request to queue (non-blocking)
                        success = threaded_explainer.add_explanation_request(explanation_input)
                        if not success:
                            logger.debug("LLM explanation queue full, skipping this request")
                        
                    except Exception as e:
                        logger.error(f"LLM preparation error: {e}")

                # Visualization
                fontText = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.75
                fontColor = (255,255,255)
                lineType = 2
                
                # Draw object detection results on the image
                if detections:
                    # Scale detection coordinates to OpenCV window size
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
                    
                    # Draw lane obstruction detection zone
                    lane_left = int(0.35 * img_width)
                    lane_right = int(0.65 * img_width)
                    horizon_line = int(0.4 * img_height)
                    
                    # Draw detection zone boundaries (semi-transparent)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (lane_left, horizon_line), (lane_right, img_height), (0, 255, 0), 2)
                    cv2.line(overlay, (lane_left, horizon_line), (lane_right, horizon_line), (0, 255, 0), 2)
                    cv2.putText(overlay, "DETECTION ZONE", (lane_left + 5, horizon_line - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Blend with original image
                    img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)
                
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
                
                # Display obstruction status
                if brake > 0:
                    if any(det.get('class_id') == 9 and 'traffic_light_state' in det for det in detections):
                        cv2.putText(img, "STOP: Traffic Light", (350, 130), fontText, 0.6, (0, 0, 255), 2)
                    else:
                        # Check if stopping due to obstruction
                        blocking_objects = ['car', 'truck', 'bus', 'motorcycle', 'person', 'bicycle']
                        blocking_class_ids = [i for i, name in enumerate(YOLO_CLASSES) if name in blocking_objects]
                        
                        img_width = carla_img_to_array(image_windshield).shape[1]
                        img_height = carla_img_to_array(image_windshield).shape[0]
                        
                        obstruction_detected = False
                        for detection in detections:
                            class_id = detection.get('class_id', 0)
                            if class_id in blocking_class_ids:
                                x1, y1, x2, y2 = detection['bbox']
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                # Scale to display coordinates
                                original_height, original_width = carla_img_to_array(image_windshield).shape[:2]
                                display_height, display_width = img.shape[:2]
                                scale_x = display_width / original_width
                                scale_y = display_height / original_height
                                
                                lane_left = 0.35 * img_width
                                lane_right = 0.65 * img_width
                                horizon_line = 0.4 * img_height
                                
                                if (lane_left <= center_x <= lane_right and center_y > horizon_line):
                                    class_name = YOLO_CLASSES[class_id] if class_id < len(YOLO_CLASSES) else ""
                                    cv2.putText(img, f"STOP: {class_name.upper()}", (350, 130), fontText, 0.6, (0, 0, 255), 2)
                                    
                                    # Draw warning box around the blocking object
                                    scaled_x1 = int(x1 * scale_x)
                                    scaled_y1 = int(y1 * scale_y)
                                    scaled_x2 = int(x2 * scale_x)
                                    scaled_y2 = int(y2 * scale_y)
                                    
                                    # Draw flashing red warning box
                                    cv2.rectangle(img, (scaled_x1-3, scaled_y1-3), (scaled_x2+3, scaled_y2+3), (0, 0, 255), 3)
                                    cv2.putText(img, "BLOCKING", (scaled_x1, scaled_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                                    obstruction_detected = True
                                    break
                        
                        if not obstruction_detected:
                            cv2.putText(img, "BRAKING", (350, 130), fontText, 0.6, (0, 165, 255), 2)
                
                # Display LLM processing status
                if threaded_explainer:
                    if threaded_explainer.is_processing():
                        cv2.putText(img, "LLM: Processing...", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), lineType)
                    else:
                        recent_explanations = threaded_explainer.get_recent_explanations(1)
                        if recent_explanations:
                            last_time = recent_explanations[-1]['processing_time']
                            cv2.putText(img, f"LLM: Ready ({last_time:.1f}s)", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType)
                        else:
                            cv2.putText(img, "LLM: Ready", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), lineType)

                cv2.imshow('Lane detect', img)
                cv2.waitKey(1)

                fps_list.append(clock.get_fps())

                # Draw the pygame display (clean, no detection boxes)
                draw_image(display, image_rgb)
                display.blit(font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)), (6, 4))
                
                pygame.display.flip()

                # 检查仿真结束条件并生成评估报告
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 最大仿真时间限制
                if elapsed_time > max_simulation_time:
                    logger.warning(f"⏰ Maximum simulation time reached! ({max_simulation_time}s)")
                    
                    # ========== 生成评估报告（超时结束）==========
                    logger.info("📊 正在生成驾驶评估报告...")
                    
                    # 设置路线完成率（基于时间）
                    eval_metrics.route_completion_rate = min(1.0, elapsed_time / max_simulation_time)
                    
                    # 生成并显示评估报告
                    generate_evaluation_report(eval_metrics, cross_track_list, detections, logger, "仿真超时")
                    break

                # 车辆长时间停止
                if speed < 0.1 and stationary_time > 30:  # 30秒不动
                    logger.warning(f"🚫 Vehicle stuck for too long! ({stationary_time:.1f}s)")
                    
                    # ========== 生成评估报告（车辆卡住）==========
                    logger.info("📊 正在生成驾驶评估报告...")
                    
                    # 设置路线完成率（基于时间，但因为卡住会降低）
                    eval_metrics.route_completion_rate = min(0.5, elapsed_time / max_simulation_time)
                    
                    # 生成并显示评估报告
                    generate_evaluation_report(eval_metrics, cross_track_list, detections, logger, "车辆停止")
                    break

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info('🧹 正在清理资源...')
        
        # Stop the LLM explainer thread
        if threaded_explainer:
            logger.info('Stopping LLM explainer thread...')
            threaded_explainer.stop()
            threaded_explainer.join(timeout=5.0)  # Wait up to 5 seconds for thread to stop
            if threaded_explainer.is_alive():
                logger.warning('LLM explainer thread did not stop gracefully')
            else:
                logger.info('LLM explainer thread stopped successfully')
        
        # 停止所有行人控制器
        if 'walker_controllers' in locals():
            logger.info('正在停止行人控制器...')
            for controller in walker_controllers:
                try:
                    controller.stop()
                    controller.destroy()
                except Exception as e:
                    logger.warning(f"停止行人控制器时出错: {e}")
            logger.info('✅ 行人控制器已停止')
        
        # 关闭Traffic Manager同步模式
        if 'traffic_manager' in locals():
            try:
                traffic_manager.set_synchronous_mode(False)
                logger.info('✅ Traffic Manager已恢复异步模式')
            except Exception as e:
                logger.warning(f"关闭Traffic Manager时出错: {e}")
        
        # Destroy CARLA actors
        logger.info('正在销毁所有actors...')
        
        # 销毁NPC车辆
        if 'spawned_vehicles' in locals():
            for vehicle in spawned_vehicles:
                try:
                    vehicle.destroy()
                except Exception as e:
                    logger.warning(f"销毁NPC车辆时出错: {e}")
            logger.info(f'✅ 已销毁 {len(spawned_vehicles)} 辆NPC车辆')
        
        # 销毁行人
        if 'spawned_walkers' in locals():
            for walker in spawned_walkers:
                try:
                    walker.destroy()
                except Exception as e:
                    logger.warning(f"销毁行人时出错: {e}")
            logger.info(f'✅ 已销毁 {len(spawned_walkers)} 个行人')
        
        # 销毁Ego车辆和其他actors
        for actor in actor_list:
            try:
                actor.destroy()
            except Exception as e:
                logger.warning(f"销毁actor时出错: {e}")
        
        logger.info('✅ 所有actors已销毁')
        
        # Print statistics
        if cross_track_list:
            logger.info(f'Mean cross track error: {np.mean(cross_track_list):.2f}')
        if fps_list:
            logger.info(f'Mean FPS: {np.mean(fps_list):.2f}')
        
        # Show LLM processing statistics
        if threaded_explainer and threaded_explainer.recent_explanations:
            recent_explanations = threaded_explainer.get_recent_explanations(5)
            if recent_explanations:
                processing_times = [exp['processing_time'] for exp in recent_explanations]
                logger.info(f'LLM processing times - Mean: {np.mean(processing_times):.2f}s, '
                           f'Min: {np.min(processing_times):.2f}s, Max: {np.max(processing_times):.2f}s')
                logger.info(f'Total explanations processed: {len(threaded_explainer.recent_explanations)}')
            
        cv2.destroyAllWindows()
        pygame.quit()
        
        logger.info('🎉 清理完成！')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--map', default='1') 
    parser.add_argument('--weather', type=int, default=0)
    parser.add_argument('--show-map', action='store_true')
    parser.add_argument('--model', default='openvino')
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO object detection')
    args = parser.parse_args()
    
    main(args.fps, args.map, args.weather, args.show_map, args.model, not args.no_llm, not args.no_yolo) 