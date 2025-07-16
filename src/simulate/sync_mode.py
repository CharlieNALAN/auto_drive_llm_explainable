#!/usr/bin/env python
# -*- coding: utf-8 -*-

import queue
import carla

class CarlaSyncMode(object):
    """Synchronous mode manager for CARLA simulation."""
    
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        """Enter synchronous mode."""
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
        """Tick the simulation and get sensor data."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug("🔄 Starting world.tick()...")
        self.frame = self.world.tick()
        logger.debug(f"✅ world.tick() completed, frame: {self.frame}")
        
        logger.debug("🔄 Retrieving sensor data...")
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        logger.debug("✅ All sensor data retrieved successfully")
        
        logger.debug("🔄 Verifying frame consistency...")
        assert all(x.frame == self.frame for x in data)
        logger.debug("✅ Frame consistency verified")
        
        return data

    def __exit__(self, *args, **kwargs):
        """Exit synchronous mode."""
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        """Retrieve data from sensor queue."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"🔄 Waiting for sensor data (timeout: {timeout}s)...")
        while True:
            try:
                data = sensor_queue.get(timeout=timeout)
                logger.debug(f"📡 Got sensor data, frame: {data.frame}, target frame: {self.frame}")
                if data.frame == self.frame:
                    logger.debug("✅ Sensor data frame matches target frame")
                    return data
                else:
                    logger.debug(f"⚠️ Frame mismatch, got {data.frame}, expected {self.frame}, continuing...")
            except Exception as e:
                logger.error(f"❌ Error retrieving sensor data: {e}")
                raise 