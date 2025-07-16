#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
import json
import os
import threading
import queue
import time
from collections import deque

# transformers no longer needed since we use API approach
TRANSFORMERS_AVAILABLE = False


class LLMExplainer:
    def __init__(self, config=None, model_type="deepseek_api"):
        self.config = config or {}
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Initialize API related variables
        self.api_key = None
        self.api_base_url = None
        self.api_model = None
        
        if model_type == "deepseek_api":
            try:
                self._initialize_deepseek_api()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Deepseek API: {e}")
                self.logger.info("Falling back to rule-based explanation")
                self.model_type = "fallback"
        else:
            self.model_type = "fallback"
    
    def _initialize_deepseek_api(self):
        """Initialize Deepseek API configuration"""
        # Get API key and base URL from configuration
        self.api_key = self.config.get('api_key')
        self.api_base_url = self.config.get('api_base_url', 'https://api.deepseek.com/v1')
        self.api_model = self.config.get('api_model', 'deepseek-chat')
        
        if not self.api_key:
            # Try to get from environment variable
            self.api_key = os.getenv('DEEPSEEK_API_KEY')
            
        if not self.api_key:
            raise Exception("No Deepseek API key provided. Please set DEEPSEEK_API_KEY environment variable or provide api_key in config.")
        
        self.logger.info(f"Deepseek API initialized with model: {self.api_model}")
        
        # Test API connection
        try:
            self._call_deepseek_api("Test connection", max_tokens=10)
            self.logger.info("Deepseek API connection test successful")
        except Exception as e:
            self.logger.warning(f"Deepseek API test failed: {e}")
            raise
    
    def _call_deepseek_api(self, prompt, max_tokens=150, temperature=0.7):
        """Call Deepseek API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Set system prompt to ensure concise and accurate English driving explanations
        system_prompt = """You are an AI explainer for autonomous vehicles. Please generate concise, accurate, and direct English explanations based on the given driving data. Keep it to one sentence."""
        
        data = {
            'model': self.api_model,
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False
        }
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                self.logger.error(f"API request failed with status {response.status_code}: {response.text}")
                raise Exception(f"API request failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            raise
        
    def explain(self, data):
        """Generate explanation based on comprehensive vehicle and perception data"""
        if self.model_type == "deepseek_api":
            return self._explain_with_deepseek_api(data)
        else:
            return self._explain_fallback(data)
    
    def _explain_with_deepseek_api(self, data):
        """Generate driving explanation using Deepseek API"""
        try:
            prompt = self._build_prompt(data)
            explanation = self._call_deepseek_api(prompt, max_tokens=100, temperature=0.7)
            
            # If generated explanation is empty or too short, use fallback method
            if not explanation or len(explanation) < 5:
                self.logger.warning("API generated explanation too short, using fallback")
                return self._explain_fallback(data)
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Deepseek API generation failed: {e}")
            return self._explain_fallback(data)
    
    def _build_prompt(self, data):
        """Build prompt for LLM"""
        vehicle_state = data.get('vehicle_state', {})
        road_context = data.get('road_context', {})
        perception = data.get('perception', {})
        detected_objects = data.get('detected_objects', {})
        
        # Extract key information
        speed_kmh = vehicle_state.get('speed_kmh', 0)
        target_speed = vehicle_state.get('target_speed', 0)
        control = vehicle_state.get('control', {})
        throttle = control.get('throttle', 0)
        steer = control.get('steer', 0)
        brake = control.get('brake', 0)
        
        current_lane = road_context.get('current_lane', {})
        is_junction = current_lane.get('is_junction', False)
        navigation_mode = road_context.get('navigation_mode', 'unknown')
        
        trajectory_curvature = perception.get('trajectory_curvature', 0)
        cross_track_error = perception.get('cross_track_error', 0)
        lane_tracking_quality = perception.get('lane_tracking_quality', 'unknown')
        
        # Detected objects information
        traffic_lights = detected_objects.get('traffic_lights', [])
        nearby_vehicles = detected_objects.get('nearby_vehicles', [])
        pedestrians = detected_objects.get('pedestrians', [])
        critical_objects = detected_objects.get('critical_objects', [])
        
        prompt = f"""You are an autonomous vehicle explanation system. Please generate a concise English driving behavior explanation based on the following information:

Vehicle State:
- Current speed: {speed_kmh:.1f} km/h
- Target speed: {target_speed:.1f} km/h
- Throttle: {throttle:.2f}, Steering: {steer:.2f}, Brake: {brake:.2f}

Road Environment:
- Lane tracking quality: {lane_tracking_quality}
- Lateral deviation: {cross_track_error:.2f}m
- Trajectory curvature: {trajectory_curvature:.4f}
- At intersection: {"Yes" if is_junction else "No"}
- Navigation mode: {navigation_mode}

Detected Objects:
- Traffic lights: {len(traffic_lights)} detected
- Nearby vehicles: {len(nearby_vehicles)} detected
- Pedestrians: {len(pedestrians)} detected
- Critical objects: {len(critical_objects)} detected

Please generate a single sentence explanation describing what the vehicle is currently doing and why. Keep it concise, accurate, and natural in English."""

        return prompt
    
    def _explain_fallback(self, data):
        """Fallback rule-based explanation method"""
        vehicle_state = data.get('vehicle_state', {})
        road_context = data.get('road_context', {})
        perception = data.get('perception', {})
        detected_objects = data.get('detected_objects', {})
        
        # Extract vehicle information
        speed_kmh = vehicle_state.get('speed_kmh', 0)
        target_speed = vehicle_state.get('target_speed', 0)
        control = vehicle_state.get('control', {})
        throttle = control.get('throttle', 0)
        steer = control.get('steer', 0)
        brake = control.get('brake', 0)
        
        # Extract road context
        current_lane = road_context.get('current_lane', {})
        navigation_mode = road_context.get('navigation_mode', 'unknown')
        upcoming_waypoints = road_context.get('upcoming_waypoints', [])
        
        # Extract perception information
        trajectory_curvature = perception.get('trajectory_curvature', 0)
        cross_track_error = perception.get('cross_track_error', 0)
        lane_tracking_quality = perception.get('lane_tracking_quality', 'unknown')
        
        # Basic driving action description
        if speed_kmh < 1:
            if brake > 0.1:
                action = "braking to stop"
            else:
                action = "stationary"
        elif brake > 0.1:
            action = "braking"
        elif throttle > 0.1:
            if speed_kmh < target_speed - 1:
                action = "accelerating"
            else:
                action = "maintaining speed"
        elif abs(steer) > 0.1:
            if trajectory_curvature > 0.01:
                direction = "right" if steer > 0 else "left"
                action = f"turning {direction}"
            else:
                direction = "right" if steer > 0 else "left"
                action = f"steering slightly {direction}"
        else:
            action = "driving straight"
        
        # Speed information
        if abs(speed_kmh - target_speed) > 2:
            speed_status = f"at {speed_kmh:.1f}km/h, target {target_speed:.1f}km/h"
        else:
            speed_status = f"at {speed_kmh:.1f}km/h"
        
        explanation = f"Vehicle is {action} {speed_status}"
        
        # Lane tracking quality
        if lane_tracking_quality == "good":
            explanation += " with good lane keeping"
        elif lane_tracking_quality == "poor":
            explanation += f" with {cross_track_error:.1f}m lane deviation"
        
        # Road context information
        if current_lane.get('is_junction'):
            explanation += " while passing through intersection"
        elif trajectory_curvature > 0.01:
            explanation += " while navigating curve"
        
        # Navigation mode
        if navigation_mode == "map_based":
            explanation += " (map navigation)"
        
        # Process detected objects by category
        traffic_light_info = []
        vehicle_info = []
        pedestrian_info = []
        critical_info = []
        
        # Traffic lights
        for tl in detected_objects.get('traffic_lights', []):
            state = tl.get('state', 'unknown')
            if state != 'unknown':
                if tl.get('is_close', False):
                    traffic_light_info.append(f"{state} light ahead")
                else:
                    traffic_light_info.append(f"{state} light distant")
        
        # Nearby vehicles
        close_vehicles = [v for v in detected_objects.get('nearby_vehicles', []) if v.get('is_close', False)]
        if close_vehicles:
            vehicle_types = {}
            for v in close_vehicles:
                vtype = v.get('class', 'vehicle')
                if vtype == 'car':
                    vtype = 'car'
                elif vtype == 'truck':
                    vtype = 'truck'
                elif vtype == 'bus':
                    vtype = 'bus'
                elif vtype == 'motorcycle':
                    vtype = 'motorcycle'
                vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1
            
            for vtype, count in vehicle_types.items():
                if count == 1:
                    vehicle_info.append(f"{vtype} ahead")
                else:
                    vehicle_info.append(f"{count} {vtype}s ahead")
        
        # Pedestrians
        close_pedestrians = [p for p in detected_objects.get('pedestrians', []) if p.get('is_close', False)]
        if close_pedestrians:
            pedestrian_count = len(close_pedestrians)
            if pedestrian_count == 1:
                pedestrian_info.append("pedestrian ahead")
            else:
                pedestrian_info.append(f"{pedestrian_count} pedestrians ahead")
        
        # Critical objects
        for obj in detected_objects.get('critical_objects', []):
            obj_class = obj.get('class', 'object')
            if obj_class == 'stop sign':
                critical_info.append("stop sign")
            else:
                critical_info.append(obj_class)
        
        # Combine object detection information
        all_objects = []
        if traffic_light_info:
            all_objects.extend(traffic_light_info)
        if critical_info:
            all_objects.extend(critical_info)
        if vehicle_info:
            all_objects.extend(vehicle_info)
        if pedestrian_info:
            all_objects.extend(pedestrian_info)
        
        if all_objects:
            explanation += f". Detected: {', '.join(all_objects)}"
        
        # Upcoming road information
        if upcoming_waypoints:
            next_wp = upcoming_waypoints[0]
            if next_wp.get('lane_change') != 'NONE':
                lane_change = next_wp.get('lane_change', '')
                if 'Left' in str(lane_change):
                    explanation += ", preparing to turn left"
                elif 'Right' in str(lane_change):
                    explanation += ", preparing to turn right"
        
        return explanation
    
class ThreadedLLMExplainer(threading.Thread):
    """
    A threaded LLM explainer that runs in the background to avoid blocking the main simulation loop.
    """
    def __init__(self, llm_explainer, max_queue_size=10, display_callback=None):
        super().__init__(daemon=True)
        self.llm_explainer = llm_explainer
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__ + '.ThreadedLLMExplainer')
        self.display_callback = display_callback  # Callback function for pygame display
        
        # Keep track of recent explanations
        self.recent_explanations = deque(maxlen=5)
        
    def run(self):
        """Main thread loop that processes LLM explanation requests."""
        self.logger.info("LLM explainer thread started")
        
        while not self.stop_event.is_set():
            try:
                # Wait for input with timeout to allow checking stop_event
                explanation_input = self.input_queue.get(timeout=0.5)
                
                # Process the explanation
                start_time = time.time()
                explanation = self.llm_explainer.explain(explanation_input)
                processing_time = time.time() - start_time
                
                # Store the explanation
                self.recent_explanations.append({
                    'explanation': explanation,
                    'timestamp': time.time(),
                    'processing_time': processing_time
                })
                
                self.logger.info(f"LLM: {explanation} (processed in {processing_time:.2f}s)")
                
                # Call display callback if provided
                if self.display_callback:
                    self.display_callback(explanation)
                
                # Mark the task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                # No new data, continue loop
                continue
            except Exception as e:
                self.logger.error(f"LLM explainer error: {e}")
                # Mark the task as done even if there was an error
                try:
                    self.input_queue.task_done()
                except:
                    pass
        
        self.logger.info("LLM explainer thread stopped")
    
    def add_explanation_request(self, explanation_input):
        """Add a new explanation request to the queue (non-blocking)."""
        try:
            # If queue is full, remove the oldest item to make room
            if self.input_queue.full():
                try:
                    self.input_queue.get_nowait()
                    self.input_queue.task_done()
                except queue.Empty:
                    pass
            
            self.input_queue.put_nowait(explanation_input)
            return True
        except queue.Full:
            self.logger.warning("LLM explanation queue is full, skipping request")
            return False
    
    def stop(self):
        """Stop the thread gracefully."""
        self.stop_event.set()
        
        # Clear the queue
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
                self.input_queue.task_done()
            except queue.Empty:
                break
    
    def get_recent_explanations(self, count=1):
        """Get the most recent explanations."""
        return list(self.recent_explanations)[-count:]
    
    def is_processing(self):
        """Check if the explainer is currently processing requests."""
        return not self.input_queue.empty()
