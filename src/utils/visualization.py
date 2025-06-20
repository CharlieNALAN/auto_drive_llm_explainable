#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pygame
import cv2
import carla

class Visualization:
    """Class to handle visualization of autonomous driving system."""
    
    def __init__(self, width, height):
        """Initialize visualization."""
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("LLM-Based Explainable Autonomous Driving")
        
        # Fonts
        self.font_small = pygame.font.SysFont('Arial', 14)
        self.font_medium = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_large = pygame.font.SysFont('Arial', 30, bold=True)
        
        # Colors
        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'background': (47, 47, 47)
        }
        
        # Object colors by class
        self.class_colors = {
            'car': self.colors['blue'],
            'pedestrian': self.colors['red'],
            'traffic_light': self.colors['yellow'],
            'stop_sign': self.colors['red'],
            'truck': self.colors['cyan'],
            'motorcycle': self.colors['magenta'],
            'bicycle': self.colors['orange'],
            'bus': self.colors['purple'],
            'other': self.colors['gray']
        }
        
        # Class name mapping (from COCO dataset indices)
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic_light',
            11: 'stop_sign',
            # Add more as needed
        }
    
    def display(self, camera_img, detected_objects, lane_info, 
                predicted_trajectories, planned_trajectory, explanation):
        """Display everything on screen."""
        # Convert camera image to pygame surface
        camera_surface = self._process_camera_image(camera_img)
        
        # Draw objects on the camera view
        self._draw_detected_objects(camera_surface, detected_objects)
        
        # Draw lane lines
        self._draw_lane_lines(camera_surface, lane_info)
        
        # Draw trajectories
        self._draw_trajectories(camera_surface, predicted_trajectories, planned_trajectory)
        
        # Create the explanation box
        self._draw_explanation_box(camera_surface, explanation)
        
        # Draw vehicle control info on the left side
        if planned_trajectory and 'behavior' in planned_trajectory:
            control_data = planned_trajectory.get('control_data', {})
            self._draw_vehicle_info(camera_surface, control_data, planned_trajectory.get('behavior'))
        
        # Display the final surface
        pygame.display.flip()
    
    def _process_camera_image(self, camera_img):
        """Convert camera image to pygame surface."""
        # Convert from RGB to BGR (pygame format)
        camera_img = camera_img[:, :, ::-1]
        
        # Create surface from numpy array
        camera_surface = pygame.surfarray.make_surface(camera_img.swapaxes(0, 1))
        
        # Display the surface
        self.screen.blit(camera_surface, (0, 0))
        
        return self.screen
    
    def _draw_detected_objects(self, surface, detected_objects):
        """Draw bounding boxes around detected objects."""
        for obj in detected_objects:
            # Get object information
            cls_id = obj['class_id']
            cls_name = self.class_names.get(cls_id, 'unknown')
            box = obj['bbox']  # [x1, y1, x2, y2]
            confidence = obj['confidence']
            
            # Get color for this class
            color = self.class_colors.get(cls_name, self.colors['gray'])
            
            # Draw bounding box
            pygame.draw.rect(surface, color, 
                             (int(box[0]), int(box[1]), 
                              int(box[2] - box[0]), int(box[3] - box[1])), 2)
            
            # Draw label
            label = f"{cls_name}: {confidence:.2f}"
            text = self.font_small.render(label, True, self.colors['white'], color)
            surface.blit(text, (int(box[0]), int(box[1]) - 20))
    
    def _draw_lane_lines(self, surface, lane_info):
        """Draw lane lines on the camera view."""
        if lane_info and 'lanes' in lane_info:
            lanes_drawn = 0
            for lane in lane_info['lanes']:
                # Check if lane has points
                if isinstance(lane, dict) and 'points' in lane:
                    points = lane['points']
                elif isinstance(lane, (list, np.ndarray)):
                    points = lane
                else:
                    continue
                
                # Skip if points is empty
                if len(points) < 2:
                    continue
                
                # Debug: Print first few world coordinates
                if lanes_drawn == 0:  # Only for first lane to avoid spam
                    world_sample = points[:3] if len(points) >= 3 else points
                    print(f"Lane world coords (first 3): {world_sample}")
                
                # Convert world coordinates to screen coordinates
                screen_points = self._world_to_screen_coords(points)
                
                # Debug: Print converted screen coordinates
                if lanes_drawn == 0 and len(screen_points) > 0:
                    screen_sample = screen_points[:3] if len(screen_points) >= 3 else screen_points
                    print(f"Lane screen coords (first 3): {screen_sample}")
                
                # Draw lane markings
                lines_drawn = 0
                for i in range(len(screen_points) - 1):
                    try:
                        # Only draw if points are on screen
                        p1 = screen_points[i]
                        p2 = screen_points[i+1]
                        # More permissive bounds check for debugging
                        if (-50 <= p1[0] <= self.width + 50 and -50 <= p1[1] <= self.height + 50 and
                            -50 <= p2[0] <= self.width + 50 and -50 <= p2[1] <= self.height + 50):
                            pygame.draw.line(surface, self.colors['yellow'], 
                                            (int(p1[0]), int(p1[1])), 
                                            (int(p2[0]), int(p2[1])), 3)
                            lines_drawn += 1
                    except (IndexError, TypeError, ValueError) as e:
                        # Skip if point format is incorrect
                        continue
                
                if lanes_drawn == 0:
                    print(f"Drew {lines_drawn} lane line segments")
                lanes_drawn += 1
    
    def _draw_trajectories(self, surface, predicted_trajectories, planned_trajectory):
        """Draw predicted and planned trajectories."""
        # Draw predicted trajectories for other vehicles
        for vehicle_id, trajectory in predicted_trajectories.items():
            if 'points' in trajectory:
                points = trajectory['points']
                screen_points = self._world_to_screen_coords(points)
                for i in range(len(screen_points) - 1):
                    try:
                        p1 = screen_points[i]
                        p2 = screen_points[i+1]
                        if (0 <= p1[0] <= self.width and 0 <= p1[1] <= self.height and
                            0 <= p2[0] <= self.width and 0 <= p2[1] <= self.height):
                            pygame.draw.line(surface, self.colors['red'], 
                                            (int(p1[0]), int(p1[1])), 
                                            (int(p2[0]), int(p2[1])), 2)
                    except (IndexError, TypeError, ValueError):
                        continue
        
        # Draw planned trajectory for ego vehicle
        if planned_trajectory and 'points' in planned_trajectory:
            points = planned_trajectory['points']
            screen_points = self._world_to_screen_coords(points)
            for i in range(len(screen_points) - 1):
                try:
                    p1 = screen_points[i]
                    p2 = screen_points[i+1]
                    if (0 <= p1[0] <= self.width and 0 <= p1[1] <= self.height and
                        0 <= p2[0] <= self.width and 0 <= p2[1] <= self.height):
                        pygame.draw.line(surface, self.colors['green'], 
                                        (int(p1[0]), int(p1[1])), 
                                        (int(p2[0]), int(p2[1])), 4)
                except (IndexError, TypeError, ValueError):
                    continue
    
    def _draw_explanation_box(self, surface, explanation):
        """Draw explanation box at the bottom of the screen."""
        # Create a semi-transparent overlay for the explanation
        explanation_surface = pygame.Surface((self.width, 100))
        explanation_surface.set_alpha(200)
        explanation_surface.fill(self.colors['black'])
        
        # Add the LLM explanation to the surface
        explanation_text = self.font_medium.render(
            f"Explanation: {explanation}", True, self.colors['white'])
        
        # Center the text
        text_x = (self.width - explanation_text.get_width()) // 2
        text_y = (100 - explanation_text.get_height()) // 2
        
        # Draw the text on the explanation surface
        explanation_surface.blit(explanation_text, (text_x, text_y))
        
        # Draw the explanation surface on the main surface
        surface.blit(explanation_surface, (0, self.height - 100))
    
    def _draw_vehicle_info(self, surface, control_data, behavior):
        """Draw vehicle control information on the left side of the screen."""
        # Create a semi-transparent panel for vehicle info
        info_width = 200
        info_height = 300
        info_surface = pygame.Surface((info_width, info_height))
        info_surface.set_alpha(220)
        info_surface.fill(self.colors['black'])
        
        # Title
        title = self.font_medium.render("Vehicle Controls", True, self.colors['white'])
        title_x = (info_width - title.get_width()) // 2
        info_surface.blit(title, (title_x, 10))
        
        # Get control values
        throttle = control_data.get('throttle', 0.0)
        brake = control_data.get('brake', 0.0)
        steer = control_data.get('steer', 0.0)
        
        # Display current behavior
        behavior_text = self.font_small.render(f"Behavior: {behavior}", True, self.colors['white'])
        info_surface.blit(behavior_text, (10, 50))
        
        # Throttle information
        throttle_text = self.font_small.render(f"Throttle: {throttle:.2f}", True, self.colors['green'])
        info_surface.blit(throttle_text, (10, 80))
        
        # Throttle bar
        pygame.draw.rect(info_surface, self.colors['gray'], (10, 105, 180, 15))
        pygame.draw.rect(info_surface, self.colors['green'], (10, 105, int(180 * throttle), 15))
        
        # Brake information
        brake_text = self.font_small.render(f"Brake: {brake:.2f}", True, self.colors['red'])
        info_surface.blit(brake_text, (10, 130))
        
        # Brake bar
        pygame.draw.rect(info_surface, self.colors['gray'], (10, 155, 180, 15))
        pygame.draw.rect(info_surface, self.colors['red'], (10, 155, int(180 * brake), 15))
        
        # Steering information
        steer_text = self.font_small.render(f"Steering: {steer:.2f}", True, self.colors['blue'])
        info_surface.blit(steer_text, (10, 180))
        
        # Steering indicator (centered at 0, left is negative, right is positive)
        pygame.draw.rect(info_surface, self.colors['gray'], (10, 205, 180, 15))
        center_x = 10 + 90  # Center point of the steering bar
        pygame.draw.rect(info_surface, self.colors['blue'], 
                         (center_x, 205, int(90 * steer), 15) if steer >= 0 else 
                         (center_x + int(90 * steer), 205, int(-90 * steer), 15))
        pygame.draw.line(info_surface, self.colors['white'], (center_x, 200), (center_x, 225), 1)
        
        # Speed information if available
        if 'speed' in control_data:
            speed = control_data.get('speed', 0.0)
            speed_text = self.font_small.render(f"Speed: {speed:.1f} km/h", True, self.colors['cyan'])
            info_surface.blit(speed_text, (10, 230))
        
        # Additional information like lateral error if available
        if 'lateral_error' in control_data:
            lat_error = control_data.get('lateral_error', 0.0)
            error_text = self.font_small.render(f"Lat Error: {lat_error:.2f} m", True, self.colors['yellow'])
            info_surface.blit(error_text, (10, 260))
        
        # Place the info panel on the main surface
        surface.blit(info_surface, (10, 10))
    
    def _world_to_screen_coords(self, world_points):
        """
        Convert world coordinates (vehicle reference frame) to screen pixel coordinates.
        
        Args:
            world_points: Array of [x, y] points in world coordinates (meters)
            
        Returns:
            Array of [x, y] points in screen pixel coordinates
        """
        if not world_points or len(world_points) == 0:
            return []
            
        world_points = np.array(world_points)
        screen_points = []
        
        # Camera parameters (approximate values for visualization)
        # These should match your camera geometry and field of view
        pixels_per_meter = 50  # Rough conversion: 50 pixels per meter
        horizon_y = self.height * 0.4  # Horizon line at 40% from top
        vanishing_point_x = self.width / 2  # Center of screen
        
        for point in world_points:
            x_world, y_world = point[0], point[1]
            
            # Simple perspective projection
            # x_world: forward distance (positive forward)
            # y_world: lateral distance (positive left)
            
            if x_world <= 0:  # Skip points behind or at the vehicle
                continue
                
            # Perspective scaling based on distance
            scale = 1.0 / max(x_world * 0.1, 0.1)  # Perspective scaling
            
            # Convert to screen coordinates
            screen_x = vanishing_point_x - (y_world * pixels_per_meter * scale)
            screen_y = horizon_y + (30 * scale)  # Lower on screen for closer points
            
            # Clamp to reasonable screen bounds (with some margin)
            screen_x = max(-100, min(self.width + 100, screen_x))
            screen_y = max(0, min(self.height, screen_y))
            
            screen_points.append([screen_x, screen_y])
        
        return screen_points
    
    def destroy(self):
        """Clean up resources."""
        pygame.quit() 