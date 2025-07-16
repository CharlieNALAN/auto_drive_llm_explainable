#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import pygame
import carla
import numpy as np
import cv2
from .geometry import carla_img_to_array

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

def should_quit():
    """Check if user wants to quit the simulation."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def get_font():
    """Get system font for display."""
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def create_carla_world(pygame_module, mapid):
    """Create CARLA world and pygame display."""
    pygame_module.init()
    # Increased height to accommodate LLM explanations at the bottom
    display = pygame_module.display.set_mode((800, 720), pygame_module.HWSURFACE | pygame_module.DOUBLEBUF)
    pygame_module.display.set_caption("CARLA Simulation - LLM Explanations at Bottom")
    font = get_font()
    clock = pygame_module.time.Clock()
    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)
    client.load_world('Town0' + mapid)
    world = client.get_world()
    return display, font, clock, world

def draw_image(surface, image, blend=False):
    """Draw CARLA image on pygame surface."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

def find_weather_presets():
    """Find available weather presets."""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def draw_detections_opencv(img, detections):
    """Draw detection boxes on OpenCV image."""
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

def wrap_text(text, max_length):
    """Simple text wrapping function."""
    if len(text) <= max_length:
        return [text]
    
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= max_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                # Word is too long, just add it
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines 