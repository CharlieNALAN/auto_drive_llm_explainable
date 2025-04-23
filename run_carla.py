#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import time
import logging
import signal
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Run CARLA Simulator")
    parser.add_argument('--carla-path', default=None, 
                        help='Path to CARLA directory (if not set, uses CARLA_PATH env variable)')
    parser.add_argument('--quality-level', default='Epic', 
                        choices=['Low', 'Medium', 'High', 'Epic'],
                        help='Graphics quality level')
    parser.add_argument('--windowed', action='store_true', 
                        help='Run CARLA in windowed mode')
    parser.add_argument('--resolution', default='1280x720', 
                        help='Window resolution (if windowed)')
    parser.add_argument('--no-rendering', action='store_true', 
                        help='Run CARLA without rendering')
    parser.add_argument('--port', type=int, default=2000, 
                        help='TCP port to listen to')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_carla_path(user_path=None):
    """Get the CARLA path from either the user-provided path or the environment variable."""
    if user_path:
        return user_path
    
    env_path = os.environ.get('CARLA_PATH')
    if env_path:
        return env_path
    
    # Default paths to check
    system = platform.system()
    default_paths = []
    
    if system == 'Windows':
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        default_paths = [
            os.path.join(program_files, 'Epic Games\\CARLA'),
            os.path.join(program_files, 'CARLA'),
        ]
    elif system == 'Linux':
        default_paths = [
            os.path.expanduser('~/CARLA'),
            '/opt/CARLA',
        ]
    elif system == 'Darwin':  # macOS
        default_paths = [
            os.path.expanduser('~/CARLA'),
            '/Applications/CARLA',
        ]
    
    for path in default_paths:
        if os.path.exists(path):
            return path
    
    raise ValueError("CARLA path not found. Please set the CARLA_PATH environment variable or use --carla-path")

def get_carla_executable(carla_path):
    """Get the path to the CARLA executable based on the operating system."""
    system = platform.system()
    
    if system == 'Windows':
        return os.path.join(carla_path, 'CarlaUE4.exe')
    elif system == 'Linux':
        return os.path.join(carla_path, 'CarlaUE4.sh')
    elif system == 'Darwin':  # macOS
        app_path = os.path.join(carla_path, 'CarlaUE4.app')
        if os.path.exists(app_path):
            return os.path.join(app_path, 'Contents/MacOS/CarlaUE4')
        return os.path.join(carla_path, 'CarlaUE4.sh')
    else:
        raise ValueError(f"Unsupported operating system: {system}")

def run_carla(args, logger):
    """Run the CARLA simulator with the specified arguments."""
    carla_path = get_carla_path(args.carla_path)
    executable = get_carla_executable(carla_path)
    
    if not os.path.exists(executable):
        raise FileNotFoundError(f"CARLA executable not found at {executable}")
    
    logger.info(f"Starting CARLA from {executable}")
    
    # Build command
    cmd = [executable]
    
    # Add arguments
    cmd.append(f"-quality-level={args.quality_level}")
    cmd.append(f"-carla-port={args.port}")
    
    if args.no_rendering:
        cmd.append("-RenderOffScreen")
    
    if args.windowed:
        cmd.append("-windowed")
        cmd.append(f"-ResX={args.resolution.split('x')[0]}")
        cmd.append(f"-ResY={args.resolution.split('x')[1]}")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run CARLA
    process = subprocess.Popen(cmd)
    
    # Handle signals
    def signal_handler(sig, frame):
        logger.info("Received signal to terminate, shutting down CARLA...")
        process.terminate()
        process.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Wait for CARLA to start up
        logger.info("Waiting for CARLA to start...")
        time.sleep(10)
        logger.info("CARLA should be running now. Press Ctrl+C to terminate.")
        
        # Wait for process to end
        process.wait()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down CARLA...")
        process.terminate()
        process.wait()
    
    logger.info("CARLA process has ended.")

def main():
    args = parse_args()
    logger = setup_logging()
    
    try:
        run_carla(args, logger)
        return 0
    except Exception as e:
        logger.error(f"Error running CARLA: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 