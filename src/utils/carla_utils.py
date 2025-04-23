#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import weakref
import numpy as np
import pygame
import carla

class CarlaWorld:
    """Class to manage the CARLA world, setup actors and traffic."""
    
    def __init__(self, client, sync_mode=True, tm_port=8000):
        self.client = client
        self.world = client.get_world()
        self.tm_port = tm_port
        self.sync_mode = sync_mode
        self.player = None
        self.blueprint_library = self.world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        
        # Configure world settings
        self._setup_world_settings(sync_mode)
        
        # Setup player vehicle
        self._setup_player()
    
    def _setup_world_settings(self, sync_mode):
        """Configure world settings for synchronous mode if enabled."""
        if sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
            self.traffic_manager.set_synchronous_mode(True)
    
    def _setup_player(self, vehicle_type='vehicle.tesla.model3'):
        """Spawn the player vehicle."""
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Get vehicle blueprint
        blueprint = self.blueprint_library.find(vehicle_type)
        blueprint.set_attribute('role_name', 'hero')
        
        # Try to spawn the vehicle
        spawn_point = random.choice(spawn_points)
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        
        if self.player is None:
            raise RuntimeError("Failed to spawn player vehicle. Retry with different spawn point or CARLA map.")
        
        print(f"Player vehicle spawned: {self.player.type_id}")
    
    def spawn_npc_vehicles(self, num_vehicles=30):
        """Spawn NPC vehicles using the Traffic Manager."""
        # Get vehicle blueprints
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        
        # Filter out bicycles and motorcycles for simplicity (optional)
        vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]
        
        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        # Limit the number of vehicles based on available spawn points
        num_vehicles = min(num_vehicles, len(spawn_points) - 1)
        
        # Spawn vehicles
        vehicles = []
        for i, spawn_point in enumerate(spawn_points[:num_vehicles]):
            blueprint = random.choice(vehicle_blueprints)
            
            # Set blueprint attributes
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            
            blueprint.set_attribute('role_name', 'autopilot')
            
            # Spawn vehicle
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                vehicles.append(vehicle)
                vehicle.set_autopilot(True, self.tm_port)
                
                # Set traffic manager parameters for this vehicle
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -10.0)  # Drive faster
                self.traffic_manager.update_vehicle_lights(vehicle, True)  # Enable vehicle lights
                
                # Random lane change behavior
                lane_change = random.choice([carla.TrafficManager.LaneChange.None_, 
                                            carla.TrafficManager.LaneChange.Right,
                                            carla.TrafficManager.LaneChange.Left])
                self.traffic_manager.force_lane_change(vehicle, lane_change)
        
        print(f"Spawned {len(vehicles)} NPC vehicles")
        return vehicles
    
    def spawn_npc_walkers(self, num_walkers=15):
        """Spawn NPC pedestrians."""
        # Get walker blueprints
        walker_blueprints = self.blueprint_library.filter('walker.pedestrian.*')
        
        # Get spawn points (we need to find valid locations for pedestrians)
        spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            location = self.world.get_random_location_from_navigation()
            if location:  # Check if navigation point is valid
                spawn_point.location = location
                spawn_points.append(spawn_point)
        
        # Spawn walker actors
        walkers = []
        walker_controllers = []
        
        # Spawn walker actors
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            
            # Set walker attributes
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
                
            # Spawn walker
            walker = self.world.try_spawn_actor(walker_bp, spawn_point)
            if walker is not None:
                walkers.append(walker)
        
        # Spawn walker controllers
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        for walker in walkers:
            controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            if controller:
                walker_controllers.append(controller)
        
        # Start walker controllers
        for controller in walker_controllers:
            # Start walker with random destination and speed
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1 + random.random())  # Speed between 1 and 2 m/s
        
        print(f"Spawned {len(walkers)} NPC walkers")
        return walkers, walker_controllers
    
    def destroy(self):
        """Destroy all actors and reset world settings."""
        if self.player:
            self.player.destroy()
            self.player = None
        
        # Reset world settings
        if self.sync_mode:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)


class CarlaSensors:
    """Class to manage sensors attached to a vehicle in CARLA."""
    
    def __init__(self, world, vehicle, width=1280, height=720):
        self.world = world
        self.vehicle = vehicle
        self.width = width
        self.height = height
        
        self.camera = None
        self.camera_data = None
        
        self.lidar = None
        self.lidar_data = None
        
        self.radar = None
        self.radar_data = None
        
        self.gnss = None
        self.gnss_data = None
        
        self.imu = None
        self.imu_data = None
        
        # Setup sensors
        self._setup_camera()
        self._setup_lidar()
        self._setup_gnss()
        self._setup_imu()
    
    def _setup_camera(self):
        """Setup RGB camera."""
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        
        # Camera relative position to the vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        # Spawn camera
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Setup camera callback
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: CarlaSensors._camera_callback(weak_self, image))
    
    def _setup_lidar(self):
        """Setup LiDAR sensor."""
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('range', '50')
        
        # LiDAR relative position to the vehicle
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        
        # Spawn LiDAR
        self.lidar = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle)
        
        # Setup LiDAR callback
        weak_self = weakref.ref(self)
        self.lidar.listen(lambda data: CarlaSensors._lidar_callback(weak_self, data))
    
    def _setup_gnss(self):
        """Setup GNSS sensor."""
        gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        
        # GNSS relative position to the vehicle
        gnss_transform = carla.Transform(carla.Location(x=0, z=2.4))
        
        # Spawn GNSS
        self.gnss = self.world.spawn_actor(
            gnss_bp, gnss_transform, attach_to=self.vehicle)
        
        # Setup GNSS callback
        weak_self = weakref.ref(self)
        self.gnss.listen(lambda data: CarlaSensors._gnss_callback(weak_self, data))
    
    def _setup_imu(self):
        """Setup IMU sensor."""
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        
        # IMU relative position to the vehicle
        imu_transform = carla.Transform(carla.Location(x=0, z=2.4))
        
        # Spawn IMU
        self.imu = self.world.spawn_actor(
            imu_bp, imu_transform, attach_to=self.vehicle)
        
        # Setup IMU callback
        weak_self = weakref.ref(self)
        self.imu.listen(lambda data: CarlaSensors._imu_callback(weak_self, data))
    
    @staticmethod
    def _camera_callback(weak_self, image):
        """Callback for camera sensor."""
        self = weak_self()
        if not self:
            return
        
        # Convert image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Store data
        self.camera_data = array
    
    @staticmethod
    def _lidar_callback(weak_self, data):
        """Callback for LiDAR sensor."""
        self = weak_self()
        if not self:
            return
        
        # Convert LiDAR data to numpy array
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Store data
        self.lidar_data = points
    
    @staticmethod
    def _gnss_callback(weak_self, data):
        """Callback for GNSS sensor."""
        self = weak_self()
        if not self:
            return
        
        # Store data
        self.gnss_data = {
            'latitude': data.latitude,
            'longitude': data.longitude,
            'altitude': data.altitude
        }
    
    @staticmethod
    def _imu_callback(weak_self, data):
        """Callback for IMU sensor."""
        self = weak_self()
        if not self:
            return
        
        # Store data
        self.imu_data = {
            'accelerometer': data.accelerometer,
            'gyroscope': data.gyroscope,
            'compass': data.compass
        }
    
    def get_data(self):
        """Check if sensor data is available."""
        return self.camera_data is not None and self.lidar_data is not None
    
    def get_vehicle_speed(self):
        """Get vehicle speed in km/h."""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            # Calculate speed in km/h
            return 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return 0.0
    
    def destroy(self):
        """Destroy all sensors."""
        sensors = [self.camera, self.lidar, self.gnss, self.imu]
        for sensor in sensors:
            if sensor:
                sensor.destroy() 