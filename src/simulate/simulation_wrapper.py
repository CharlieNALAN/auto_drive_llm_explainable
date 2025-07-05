"""
CARLA仿真包装器类
提供标准化的接口来运行CARLA仿真
"""

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import carla
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulate.k_real_simulate import (
    reset_spectator_to_ego,
    ManualController,
    GlobalKeyListener,
    your_autonomous_driving_algorithm
)


class CarlaSimulationWrapper:
    """CARLA仿真包装器类"""
    
    def __init__(self, config_path: str = "configs/simulation.json"):
        """
        初始化仿真包装器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.client = None
        self.world = None
        self.traffic_manager = None
        self.ego_vehicle = None
        self.spectator = None
        self.camera = None
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.walker_controllers = []
        self.manual_controller = None
        self.global_key_listener = None
        self.is_running = False
        self._cleanup_performed = False
        
        # 图像保存计数器
        self.saved_image_count = 0
        
        # 回调函数
        self.on_tick_callbacks: List[Callable] = []
        self.on_sensor_data_callbacks: List[Callable] = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ 配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"⚠️ 配置文件格式错误: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "simulation": {
                "carla": {
                    "host": "localhost",
                    "port": 2000,
                    "timeout": 30.0,
                    "map": "Town04"
                },
                "vehicles": {
                    "npc_count": 20,
                    "ego_vehicle": {
                        "model": "vehicle.tesla.model3",
                        "control_mode": "manual"
                    }
                },
                "pedestrians": {
                    "count": 10
                },
                "recording": {
                    "output_dir": "out",
                    "save_interval": 100
                }
            }
        }
    
    def connect_to_carla(self) -> bool:
        """连接到CARLA服务器"""
        try:
            carla_config = self.config["simulation"]["carla"]
            
            print(f"🔗 连接到CARLA服务器 {carla_config['host']}:{carla_config['port']}...")
            self.client = carla.Client(carla_config["host"], carla_config["port"])
            self.client.set_timeout(carla_config["timeout"])
            
            # 加载地图
            map_name = carla_config.get("map", "Town04")
            print(f"🗺️ 加载地图: {map_name}")
            self.world = self.client.load_world(map_name)
            
            # 获取观察者
            self.spectator = self.world.get_spectator()
            
            print("✅ 成功连接到CARLA服务器")
            return True
            
        except Exception as e:
            print(f"❌ 连接CARLA服务器失败: {e}")
            return False
    
    def setup_world(self) -> bool:
        """设置世界参数"""
        try:
            print("🔧 设置世界参数...")
            
            # 获取设置
            settings = self.world.get_settings()
            carla_settings = self.config["simulation"]["carla"]["settings"]
            
            # 应用设置
            settings.synchronous_mode = carla_settings.get("synchronous_mode", True)
            settings.fixed_delta_seconds = carla_settings.get("fixed_delta_seconds", 0.01)
            settings.no_rendering_mode = carla_settings.get("no_rendering_mode", False)
            
            self.world.apply_settings(settings)
            
            # 设置天气
            weather_config = self.config["simulation"]["carla"].get("weather", {})
            if weather_config:
                weather = carla.WeatherParameters(
                    cloudiness=weather_config.get("cloudiness", 0.0),
                    precipitation=weather_config.get("precipitation", 0.0),
                    precipitation_deposits=weather_config.get("precipitation_deposits", 0.0),
                    wind_intensity=weather_config.get("wind_intensity", 0.0),
                    sun_azimuth_angle=weather_config.get("sun_azimuth_angle", 45.0),
                    sun_altitude_angle=weather_config.get("sun_altitude_angle", 45.0),
                    fog_density=weather_config.get("fog_density", 0.0),
                    fog_distance=weather_config.get("fog_distance", 0.0),
                    wetness=weather_config.get("wetness", 0.0),
                    scattering_intensity=weather_config.get("scattering_intensity", 0.0)
                )
                self.world.set_weather(weather)
            
            print("✅ 世界参数设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 设置世界参数失败: {e}")
            return False
    
    def cleanup_existing_actors(self) -> bool:
        """清理现有的所有actors"""
        try:
            print("🧹 清理现有actors...")
            
            actors = self.world.get_actors()
            vehicles = actors.filter('*vehicle*')
            sensors = actors.filter('*sensor*')
            walkers = actors.filter('*walker*')
            
            print(f"找到 {len(vehicles)} 辆车辆，{len(sensors)} 个传感器，{len(walkers)} 个行人")
            
            # 销毁所有现有actors
            for actor in list(vehicles) + list(sensors) + list(walkers):
                if actor is not None:
                    actor.destroy()
            
            print("✅ 现有actors清理完成")
            return True
            
        except Exception as e:
            print(f"❌ 清理actors失败: {e}")
            return False
    
    def spawn_vehicles(self) -> bool:
        """生成车辆"""
        try:
            print("🚗 生成车辆...")
            
            vehicle_config = self.config["simulation"]["vehicles"]
            
            # 获取车辆蓝图和生成点
            vehicle_blueprints = self.world.get_blueprint_library().filter('*vehicle*')
            spawn_points = self.world.get_map().get_spawn_points()
            
            # 生成NPC车辆
            npc_count = vehicle_config.get("npc_count", 20)
            for i in range(npc_count):
                vehicle = self.world.try_spawn_actor(
                    np.random.choice(vehicle_blueprints),
                    np.random.choice(spawn_points)
                )
                if vehicle is not None:
                    self.spawned_vehicles.append(vehicle)
            
            # 生成Ego车辆
            ego_config = vehicle_config.get("ego_vehicle", {})
            ego_bp = self.world.get_blueprint_library().find(
                ego_config.get("model", "vehicle.tesla.model3")
            )
            
            if ego_bp is None:
                ego_bp = np.random.choice(vehicle_blueprints)
            
            # 设置Ego车辆属性
            ego_bp.set_attribute('role_name', ego_config.get("role_name", "hero"))
            if ego_bp.has_attribute('color'):
                ego_bp.set_attribute('color', ego_config.get("color", "255,0,0"))
            
            # 尝试生成Ego车辆
            for spawn_point in spawn_points:
                self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
                if self.ego_vehicle is not None:
                    break
            
            if self.ego_vehicle is None:
                print("❌ 无法生成Ego车辆")
                return False
            
            print(f"✅ 成功生成 {len(self.spawned_vehicles)} 辆NPC车辆和1辆Ego车辆")
            return True
            
        except Exception as e:
            print(f"❌ 生成车辆失败: {e}")
            return False
    
    def spawn_pedestrians(self) -> bool:
        """生成行人"""
        try:
            print("🚶 生成行人...")
            
            pedestrian_config = self.config["simulation"]["pedestrians"]
            
            # 获取行人蓝图
            walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            
            # 生成行人
            num_walkers = pedestrian_config.get("count", 10)
            walker_spawn_points = []
            
            for i in range(num_walkers):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    walker_spawn_points.append(spawn_point)
            
            # 创建行人和控制器
            for i, spawn_point in enumerate(walker_spawn_points):
                walker_bp = np.random.choice(walker_blueprints)
                
                # 设置行人属性
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 
                                           str(not pedestrian_config.get("is_invincible", True)))
                
                if walker_bp.has_attribute('speed'):
                    speed_range = pedestrian_config.get("speed_range", [1.0, 2.5])
                    speed = np.random.uniform(speed_range[0], speed_range[1])
                    walker_bp.set_attribute('speed', str(speed))
                
                # 生成行人
                walker = self.world.try_spawn_actor(walker_bp, spawn_point)
                if walker is not None:
                    self.spawned_walkers.append(walker)
                    
                    # 创建控制器
                    controller = self.world.spawn_actor(walker_controller_bp, 
                                                      carla.Transform(), 
                                                      attach_to=walker)
                    if controller is not None:
                        self.walker_controllers.append(controller)
            
            print(f"✅ 成功生成 {len(self.spawned_walkers)} 个行人")
            return True
            
        except Exception as e:
            print(f"❌ 生成行人失败: {e}")
            return False
    
    def setup_sensors(self) -> bool:
        """设置传感器"""
        try:
            print("📹 设置传感器...")
            
            if self.ego_vehicle is None:
                print("❌ Ego车辆未生成，无法设置传感器")
                return False
            
            sensor_config = self.config["simulation"]["sensors"]["camera"]
            
            # 创建摄像头变换
            camera_transform = carla.Transform(
                carla.Location(
                    x=sensor_config["position"]["x"],
                    y=sensor_config["position"]["y"],
                    z=sensor_config["position"]["z"]
                ),
                carla.Rotation(
                    pitch=sensor_config["rotation"]["pitch"],
                    yaw=sensor_config["rotation"]["yaw"],
                    roll=sensor_config["rotation"]["roll"]
                )
            )
            
            # 创建摄像头蓝图
            camera_bp = self.world.get_blueprint_library().find(sensor_config["type"])
            camera_bp.set_attribute('image_size_x', str(sensor_config["image_size_x"]))
            camera_bp.set_attribute('image_size_y', str(sensor_config["image_size_y"]))
            camera_bp.set_attribute('fov', str(sensor_config["fov"]))
            
            # 生成摄像头
            self.camera = self.world.spawn_actor(camera_bp, camera_transform, 
                                               attach_to=self.ego_vehicle)
            
            # 设置图像保存回调
            recording_config = self.config["simulation"]["recording"]
            save_interval = recording_config.get("save_interval", 100)
            output_dir = recording_config.get("output_dir", "out")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            def save_image_callback(image):
                if image.frame % save_interval == 0:
                    filename = f"{output_dir}/{self.saved_image_count:06d}.png"
                    image.save_to_disk(filename)
                    self.saved_image_count += 1
                    
                    # 调用传感器数据回调
                    for callback in self.on_sensor_data_callbacks:
                        callback(image)
            
            self.camera.listen(save_image_callback)
            
            print("✅ 传感器设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 设置传感器失败: {e}")
            return False
    
    def setup_traffic_manager(self) -> bool:
        """设置交通管理器"""
        try:
            print("🚦 设置交通管理器...")
            
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_random_device_seed(42)
            
            autopilot_config = self.config["simulation"]["vehicles"]["autopilot"]
            
            # 设置全局参数
            self.traffic_manager.global_percentage_speed_difference(
                autopilot_config.get("global_speed_difference", 50.0)
            )
            self.traffic_manager.set_global_distance_to_leading_vehicle(
                autopilot_config.get("global_distance_to_leading_vehicle", 4.0)
            )
            
            # 为NPC车辆启用autopilot
            for vehicle in self.spawned_vehicles:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.traffic_manager.vehicle_percentage_speed_difference(
                    vehicle, autopilot_config.get("individual_speed_difference", 50.0)
                )
                self.traffic_manager.distance_to_leading_vehicle(
                    vehicle, autopilot_config.get("individual_distance_to_leading_vehicle", 4.0)
                )
                self.traffic_manager.auto_lane_change(
                    vehicle, autopilot_config.get("auto_lane_change", False)
                )
            
            print("✅ 交通管理器设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 设置交通管理器失败: {e}")
            return False
    
    def setup_ego_control(self) -> bool:
        """设置Ego车辆控制"""
        try:
            if self.ego_vehicle is None:
                return False
            
            ego_config = self.config["simulation"]["vehicles"]["ego_vehicle"]
            control_mode = ego_config.get("control_mode", "manual")
            
            if control_mode == "manual":
                self.manual_controller = ManualController(self.ego_vehicle)
                self.manual_controller.print_instructions()
                
                # 设置全局键盘监听器
                self.global_key_listener = GlobalKeyListener(self.ego_vehicle, self.spectator)
                self.global_key_listener.set_manual_controller(self.manual_controller)
                
            elif control_mode == "autopilot":
                self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
                
            print(f"✅ Ego车辆控制模式设置为: {control_mode}")
            return True
            
        except Exception as e:
            print(f"❌ 设置Ego车辆控制失败: {e}")
            return False
    
    def setup_spectator(self) -> bool:
        """设置观察者"""
        try:
            if self.ego_vehicle is None or self.spectator is None:
                return False
            
            spectator_config = self.config["simulation"]["spectator"]
            
            if spectator_config.get("follow_ego", True):
                reset_spectator_to_ego(self.ego_vehicle, self.spectator)
            
            print("✅ 观察者设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 设置观察者失败: {e}")
            return False
    
    def start_simulation(self) -> bool:
        """启动仿真"""
        try:
            print("🎬 启动仿真...")
            
            # 启动行人AI
            for controller in self.walker_controllers:
                controller.start()
                target_location = self.world.get_random_location_from_navigation()
                if target_location is not None:
                    controller.go_to_location(target_location)
                    controller.set_max_speed(np.random.uniform(1.0, 2.5))
            
            self.is_running = True
            print("✅ 仿真启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 启动仿真失败: {e}")
            return False
    
    def run_simulation_loop(self):
        """运行仿真循环"""
        frame_count = 0
        
        print("🔄 开始仿真循环...")
        print("📍 按V键回到Ego车辆视角，按H键查看帮助")
        print("⏹️ 按Ctrl+C停止仿真")
        
        try:
            while self.is_running:
                # 更新世界
                self.world.tick()
                frame_count += 1
                
                # 处理键盘输入
                if self.global_key_listener:
                    self.global_key_listener.update()
                
                # 调用tick回调
                for callback in self.on_tick_callbacks:
                    callback(frame_count)
                
                # 状态显示
                if frame_count % 1000 == 0:
                    elapsed_time = frame_count * 0.01
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    print(f"仿真运行中: {frame_count} 帧 ({minutes}分{seconds:.1f}秒)")
                
        except KeyboardInterrupt:
            print("\n⏹️ 用户停止仿真")
            self.is_running = False
        except Exception as e:
            print(f"❌ 仿真循环错误: {e}")
            self.is_running = False
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self._cleanup_performed:
            return
        
        print("🧹 开始清理资源...")
        
        try:
            # 恢复异步模式
            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            
            # 停止传感器
            if self.camera is not None:
                self.camera.stop()
                self.camera.destroy()
            
            # 销毁车辆
            for vehicle in self.spawned_vehicles:
                if vehicle is not None:
                    vehicle.destroy()
            
            if self.ego_vehicle is not None:
                self.ego_vehicle.destroy()
            
            # 停止行人控制器
            for controller in self.walker_controllers:
                if controller is not None:
                    controller.stop()
                    controller.destroy()
            
            # 销毁行人
            for walker in self.spawned_walkers:
                if walker is not None:
                    walker.destroy()
            
            # 关闭交通管理器同步模式
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)
            
            self._cleanup_performed = True
            print("✅ 资源清理完成")
            
        except Exception as e:
            print(f"❌ 清理资源时出错: {e}")
    
    def add_tick_callback(self, callback: Callable):
        """添加每帧回调函数"""
        self.on_tick_callbacks.append(callback)
    
    def add_sensor_data_callback(self, callback: Callable):
        """添加传感器数据回调函数"""
        self.on_sensor_data_callbacks.append(callback)
    
    def get_ego_vehicle(self) -> Optional[carla.Vehicle]:
        """获取Ego车辆"""
        return self.ego_vehicle
    
    def get_world(self) -> Optional[carla.World]:
        """获取世界对象"""
        return self.world
    
    def get_traffic_manager(self) -> Optional[carla.TrafficManager]:
        """获取交通管理器"""
        return self.traffic_manager


def main():
    """主函数，用于独立运行仿真"""
    print("🚀 启动CARLA仿真包装器...")
    
    # 创建仿真实例
    sim = CarlaSimulationWrapper()
    
    try:
        # 初始化仿真
        if not sim.connect_to_carla():
            return
        
        if not sim.setup_world():
            return
        
        if not sim.cleanup_existing_actors():
            return
        
        if not sim.spawn_vehicles():
            return
        
        if not sim.spawn_pedestrians():
            return
        
        if not sim.setup_sensors():
            return
        
        if not sim.setup_traffic_manager():
            return
        
        if not sim.setup_ego_control():
            return
        
        if not sim.setup_spectator():
            return
        
        if not sim.start_simulation():
            return
        
        # 运行仿真循环
        sim.run_simulation_loop()
        
    except Exception as e:
        print(f"❌ 仿真运行错误: {e}")
    finally:
        sim.cleanup()


if __name__ == "__main__":
    main() 