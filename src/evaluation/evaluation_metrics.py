"""
自动驾驶算法评估指标模块
提供多维度的评估指标计算功能
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

class EvaluationMetrics:
    """自动驾驶算法评估指标类"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化评估指标
        
        Args:
            window_size: 滑动窗口大小，用于计算平均值
        """
        self.window_size = window_size
        self.reset()
        
    def reset(self):
        """重置所有指标"""
        # 轨迹跟踪指标
        self.cross_track_errors = deque(maxlen=self.window_size)
        self.heading_errors = deque(maxlen=self.window_size)
        self.speed_errors = deque(maxlen=self.window_size)
        
        # 安全性指标
        self.collision_count = 0
        self.near_collision_count = 0
        self.traffic_light_violations = 0
        self.stop_sign_violations = 0
        self.lane_change_violations = 0
        self.min_distances_to_objects = deque(maxlen=self.window_size)
        
        # 效率性指标
        self.speeds = deque(maxlen=self.window_size)
        self.accelerations = deque(maxlen=self.window_size)
        self.jerks = deque(maxlen=self.window_size)
        self.route_completion_rate = 0.0
        self.total_distance = 0.0
        self.total_time = 0.0
        
        # 舒适性指标
        self.lateral_accelerations = deque(maxlen=self.window_size)
        self.steering_angles = deque(maxlen=self.window_size)
        self.steering_rates = deque(maxlen=self.window_size)
        self.throttle_changes = deque(maxlen=self.window_size)
        self.brake_changes = deque(maxlen=self.window_size)
        
        # 感知性能指标
        self.object_detection_accuracy = deque(maxlen=self.window_size)
        self.lane_detection_accuracy = deque(maxlen=self.window_size)
        self.traffic_light_detection_accuracy = deque(maxlen=self.window_size)
        
        # 决策性能指标
        self.decision_time = deque(maxlen=self.window_size)
        self.overtaking_success_rate = 0.0
        self.lane_change_success_rate = 0.0
        
        # 上一帧的数据（用于计算导数）
        self.prev_speed = 0.0
        self.prev_acceleration = 0.0
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        
    def update_trajectory_metrics(self, cross_track_error: float, heading_error: float, 
                                speed_error: float):
        """
        更新轨迹跟踪指标
        
        Args:
            cross_track_error: 横向偏差 (m)
            heading_error: 航向偏差 (rad)
            speed_error: 速度偏差 (m/s)
        """
        self.cross_track_errors.append(abs(cross_track_error))
        self.heading_errors.append(abs(heading_error))
        self.speed_errors.append(abs(speed_error))
        
    def update_safety_metrics(self, collision_occurred: bool, near_collision: bool,
                            traffic_light_violation: bool, stop_sign_violation: bool,
                            lane_change_violation: bool, min_distance_to_objects: float):
        """
        更新安全性指标
        
        Args:
            collision_occurred: 是否发生碰撞
            near_collision: 是否差点碰撞
            traffic_light_violation: 是否违反交通灯
            stop_sign_violation: 是否违反停车标志
            lane_change_violation: 是否违规变道
            min_distance_to_objects: 与最近物体的距离
        """
        if collision_occurred:
            self.collision_count += 1
        if near_collision:
            self.near_collision_count += 1
        if traffic_light_violation:
            self.traffic_light_violations += 1
        if stop_sign_violation:
            self.stop_sign_violations += 1
        if lane_change_violation:
            self.lane_change_violations += 1
            
        self.min_distances_to_objects.append(min_distance_to_objects)
        
    def update_efficiency_metrics(self, speed: float, acceleration: float, 
                                distance_traveled: float, time_elapsed: float):
        """
        更新效率性指标
        
        Args:
            speed: 当前速度 (m/s)
            acceleration: 当前加速度 (m/s²)
            distance_traveled: 本帧行驶距离 (m)
            time_elapsed: 本帧耗时 (s)
        """
        self.speeds.append(speed)
        self.accelerations.append(acceleration)
        
        # 计算加速度变化率（jerk）
        if len(self.accelerations) > 1:
            jerk = (acceleration - self.prev_acceleration) / time_elapsed
            self.jerks.append(abs(jerk))
            
        self.total_distance += distance_traveled
        self.total_time += time_elapsed
        
        self.prev_acceleration = acceleration
        
    def update_comfort_metrics(self, lateral_acceleration: float, steering_angle: float,
                             throttle: float, brake: float, time_elapsed: float):
        """
        更新舒适性指标
        
        Args:
            lateral_acceleration: 横向加速度 (m/s²)
            steering_angle: 转向角度 (rad)
            throttle: 油门踏板 (0-1)
            brake: 刹车踏板 (0-1)
            time_elapsed: 本帧耗时 (s)
        """
        self.lateral_accelerations.append(abs(lateral_acceleration))
        self.steering_angles.append(abs(steering_angle))
        
        # 计算转向角速度
        if len(self.steering_angles) > 1:
            steering_rate = (steering_angle - self.prev_steering) / time_elapsed
            self.steering_rates.append(abs(steering_rate))
            
        # 计算油门刹车变化率
        throttle_change = abs(throttle - self.prev_throttle) / time_elapsed
        brake_change = abs(brake - self.prev_brake) / time_elapsed
        
        self.throttle_changes.append(throttle_change)
        self.brake_changes.append(brake_change)
        
        self.prev_steering = steering_angle
        self.prev_throttle = throttle
        self.prev_brake = brake
        
    def update_perception_metrics(self, object_detection_acc: float, 
                                lane_detection_acc: float, 
                                traffic_light_detection_acc: float):
        """
        更新感知性能指标
        
        Args:
            object_detection_acc: 目标检测准确率
            lane_detection_acc: 车道检测准确率
            traffic_light_detection_acc: 交通灯检测准确率
        """
        self.object_detection_accuracy.append(object_detection_acc)
        self.lane_detection_accuracy.append(lane_detection_acc)
        self.traffic_light_detection_accuracy.append(traffic_light_detection_acc)
        
    def update_decision_metrics(self, decision_time: float):
        """
        更新决策性能指标
        
        Args:
            decision_time: 决策耗时 (s)
        """
        self.decision_time.append(decision_time)
        
    def get_trajectory_performance(self) -> Dict[str, float]:
        """获取轨迹跟踪性能指标"""
        return {
            'mean_cross_track_error': np.mean(self.cross_track_errors) if self.cross_track_errors else 0.0,
            'max_cross_track_error': np.max(self.cross_track_errors) if self.cross_track_errors else 0.0,
            'std_cross_track_error': np.std(self.cross_track_errors) if self.cross_track_errors else 0.0,
            'mean_heading_error': np.mean(self.heading_errors) if self.heading_errors else 0.0,
            'max_heading_error': np.max(self.heading_errors) if self.heading_errors else 0.0,
            'mean_speed_error': np.mean(self.speed_errors) if self.speed_errors else 0.0,
            'max_speed_error': np.max(self.speed_errors) if self.speed_errors else 0.0,
        }
        
    def get_safety_performance(self) -> Dict[str, float]:
        """获取安全性能指标"""
        total_frames = len(self.cross_track_errors)
        return {
            'collision_count': self.collision_count,
            'collision_rate': self.collision_count / max(total_frames, 1) * 1000,  # per 1000 frames
            'near_collision_count': self.near_collision_count,
            'near_collision_rate': self.near_collision_count / max(total_frames, 1) * 1000,
            'traffic_light_violations': self.traffic_light_violations,
            'stop_sign_violations': self.stop_sign_violations,
            'lane_change_violations': self.lane_change_violations,
            'mean_min_distance_to_objects': np.mean(self.min_distances_to_objects) if self.min_distances_to_objects else 0.0,
            'min_distance_to_objects': np.min(self.min_distances_to_objects) if self.min_distances_to_objects else 0.0,
        }
        
    def get_efficiency_performance(self) -> Dict[str, float]:
        """获取效率性能指标"""
        return {
            'mean_speed': np.mean(self.speeds) if self.speeds else 0.0,
            'max_speed': np.max(self.speeds) if self.speeds else 0.0,
            'mean_acceleration': np.mean(self.accelerations) if self.accelerations else 0.0,
            'max_acceleration': np.max(self.accelerations) if self.accelerations else 0.0,
            'mean_jerk': np.mean(self.jerks) if self.jerks else 0.0,
            'max_jerk': np.max(self.jerks) if self.jerks else 0.0,
            'average_speed': self.total_distance / max(self.total_time, 1e-6),  # m/s
            'route_completion_rate': self.route_completion_rate,
            'total_distance': self.total_distance,
            'total_time': self.total_time,
        }
        
    def get_comfort_performance(self) -> Dict[str, float]:
        """获取舒适性能指标"""
        return {
            'mean_lateral_acceleration': np.mean(self.lateral_accelerations) if self.lateral_accelerations else 0.0,
            'max_lateral_acceleration': np.max(self.lateral_accelerations) if self.lateral_accelerations else 0.0,
            'mean_steering_angle': np.mean(self.steering_angles) if self.steering_angles else 0.0,
            'max_steering_angle': np.max(self.steering_angles) if self.steering_angles else 0.0,
            'mean_steering_rate': np.mean(self.steering_rates) if self.steering_rates else 0.0,
            'max_steering_rate': np.max(self.steering_rates) if self.steering_rates else 0.0,
            'mean_throttle_change': np.mean(self.throttle_changes) if self.throttle_changes else 0.0,
            'max_throttle_change': np.max(self.throttle_changes) if self.throttle_changes else 0.0,
            'mean_brake_change': np.mean(self.brake_changes) if self.brake_changes else 0.0,
            'max_brake_change': np.max(self.brake_changes) if self.brake_changes else 0.0,
        }
        
    def get_perception_performance(self) -> Dict[str, float]:
        """获取感知性能指标"""
        return {
            'mean_object_detection_accuracy': np.mean(self.object_detection_accuracy) if self.object_detection_accuracy else 0.0,
            'mean_lane_detection_accuracy': np.mean(self.lane_detection_accuracy) if self.lane_detection_accuracy else 0.0,
            'mean_traffic_light_detection_accuracy': np.mean(self.traffic_light_detection_accuracy) if self.traffic_light_detection_accuracy else 0.0,
        }
        
    def get_decision_performance(self) -> Dict[str, float]:
        """获取决策性能指标"""
        return {
            'mean_decision_time': np.mean(self.decision_time) if self.decision_time else 0.0,
            'max_decision_time': np.max(self.decision_time) if self.decision_time else 0.0,
            'overtaking_success_rate': self.overtaking_success_rate,
            'lane_change_success_rate': self.lane_change_success_rate,
        }
        
    def get_overall_score(self) -> float:
        """
        计算综合评分 (0-100)
        
        Returns:
            综合评分，100分为满分
        """
        # 权重设置
        weights = {
            'safety': 0.4,      # 安全性权重最高
            'trajectory': 0.2,   # 轨迹跟踪
            'efficiency': 0.2,   # 效率性
            'comfort': 0.1,     # 舒适性
            'perception': 0.05,  # 感知性能
            'decision': 0.05,   # 决策性能
        }
        
        scores = {}
        
        # 安全性评分 (碰撞和违规会大幅降分)
        safety_perf = self.get_safety_performance()
        safety_score = 100.0
        safety_score -= safety_perf['collision_count'] * 20  # 每次碰撞扣20分
        safety_score -= safety_perf['near_collision_count'] * 5  # 每次差点碰撞扣5分
        safety_score -= safety_perf['traffic_light_violations'] * 10  # 每次违反交通灯扣10分
        safety_score -= safety_perf['stop_sign_violations'] * 10
        safety_score -= safety_perf['lane_change_violations'] * 5
        safety_score = max(0, safety_score)
        scores['safety'] = safety_score
        
        # 轨迹跟踪评分
        traj_perf = self.get_trajectory_performance()
        cross_track_score = max(0, 100 - traj_perf['mean_cross_track_error'] * 50)  # 横向偏差越大分数越低
        heading_score = max(0, 100 - traj_perf['mean_heading_error'] * 180 / np.pi * 2)  # 航向偏差
        speed_score = max(0, 100 - traj_perf['mean_speed_error'] * 10)  # 速度偏差
        scores['trajectory'] = (cross_track_score + heading_score + speed_score) / 3
        
        # 效率性评分
        eff_perf = self.get_efficiency_performance()
        speed_score = min(100, eff_perf['mean_speed'] * 10)  # 速度越快分数越高，但有上限
        jerk_score = max(0, 100 - eff_perf['mean_jerk'] * 20)  # 加速度变化越大分数越低
        scores['efficiency'] = (speed_score + jerk_score) / 2
        
        # 舒适性评分
        comfort_perf = self.get_comfort_performance()
        lateral_acc_score = max(0, 100 - comfort_perf['mean_lateral_acceleration'] * 20)
        steering_score = max(0, 100 - comfort_perf['mean_steering_rate'] * 50)
        scores['comfort'] = (lateral_acc_score + steering_score) / 2
        
        # 感知性能评分
        perception_perf = self.get_perception_performance()
        scores['perception'] = (perception_perf['mean_object_detection_accuracy'] + 
                              perception_perf['mean_lane_detection_accuracy'] + 
                              perception_perf['mean_traffic_light_detection_accuracy']) / 3 * 100
        
        # 决策性能评分
        decision_perf = self.get_decision_performance()
        decision_time_score = max(0, 100 - decision_perf['mean_decision_time'] * 1000)  # 决策时间越短分数越高
        scores['decision'] = decision_time_score
        
        # 计算加权总分
        total_score = sum(scores[key] * weights[key] for key in scores)
        
        return min(100, max(0, total_score))
    
    def get_full_report(self) -> Dict[str, Dict[str, float]]:
        """获取完整评估报告"""
        return {
            'trajectory_performance': self.get_trajectory_performance(),
            'safety_performance': self.get_safety_performance(),
            'efficiency_performance': self.get_efficiency_performance(),
            'comfort_performance': self.get_comfort_performance(),
            'perception_performance': self.get_perception_performance(),
            'decision_performance': self.get_decision_performance(),
            'overall_score': self.get_overall_score(),
        } 