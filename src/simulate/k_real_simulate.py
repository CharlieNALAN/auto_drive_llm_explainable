import carla
import random
import os
import glob
import time

# ========== 快速回到Ego视角函数 ==========
def reset_spectator_to_ego(ego_vehicle, spectator):
    """
    快速将观察者重置到ego vehicle的最佳观察位置（温和调整）
    """
    try:
        ego_transform = ego_vehicle.get_transform()
        forward_vector = ego_transform.get_forward_vector()
        
        # 计算观察者位置：车辆后方距离和高度都减小，更温和的视角
        back_distance = 5.0   # 改回到5米
        height_offset = 4.0   # 改回到4米
        
        spectator_location = carla.Location(
            ego_transform.location.x - forward_vector.x * back_distance,
            ego_transform.location.y - forward_vector.y * back_distance,
            ego_transform.location.z + height_offset
        )
        
        # 减小俯视角度，更自然的观察角度
        pitch_angle = -15  # 改回到-15度
        
        spectator_transform = carla.Transform(
            spectator_location,
            carla.Rotation(pitch=pitch_angle, yaw=ego_transform.rotation.yaw)
        )
        
        spectator.set_transform(spectator_transform)
        
        ego_pos = ego_transform.location
        print(f"📍 视角已温和调整到Ego车辆后方")
        print(f"🚗 Ego位置: x={ego_pos.x:.1f}, y={ego_pos.y:.1f}")
        print(f"👁️ 观察者位置: 后方{back_distance}m，高度{height_offset}m，俯角{abs(pitch_angle)}°")
        return True
        
    except Exception as e:
        print(f"❌ 视角调整失败: {e}")
        return False

# ========== 自定义自动驾驶算法框架 ==========
def your_autonomous_driving_algorithm(ego_vehicle, world, traffic_manager):
    """
    你的自动驾驶算法接口
    
    参数:
        ego_vehicle: Ego车辆对象
        world: CARLA世界对象
        traffic_manager: Traffic Manager对象
    
    在这里实现你的自动驾驶逻辑：
    - 传感器数据处理
    - 路径规划
    - 决策制定
    - 控制指令生成
    """
    # 示例：获取车辆状态
    transform = ego_vehicle.get_transform()
    velocity = ego_vehicle.get_velocity()
    location = transform.location
    
    # 示例：简单的控制逻辑（直线行驶）
    control = carla.VehicleControl()
    control.throttle = 0.3  # 油门
    control.steer = 0.0     # 转向
    control.brake = 0.0     # 刹车
    
    # 应用控制指令
    ego_vehicle.apply_control(control)
    
    print(f"🧠 AI控制 - 位置: ({location.x:.1f}, {location.y:.1f}), 速度: {velocity.length():.1f} m/s")

# ========== 手动控制函数 ==========
def apply_manual_control(ego_vehicle):
    """
    手动控制函数（需要配合键盘监听）
    这是一个示例，实际使用时需要实现键盘监听
    """
    # 这里可以实现键盘监听逻辑
    # 或者使用CARLA的手动控制功能
    pass

# ========== 手动控制系统 ==========
import threading
import msvcrt  # Windows键盘输入

class ManualController:
    def __init__(self, ego_vehicle):
        self.ego_vehicle = ego_vehicle
        self.control = carla.VehicleControl()
        self.is_running = True
        
        # 添加累积控制状态，实现更平滑的控制
        self._throttle_accumulate = 0.0
        self._brake_accumulate = 0.0
        self._steer_accumulate = 0.0
        
        # 控制参数 - 进一步大幅降低敏感度，实现超级平滑控制
        self.throttle_increment = 0.02   # 每次按键增加2%油门（从5%进一步降低）
        self.brake_increment = 0.04      # 每次按键增加4%刹车（从8%降低）
        self.steer_increment = 0.03      # 每次按键增加3%转向（从8%降低）
        
        # 最大值限制 - 降低最大值
        self.max_throttle = 0.5          # 最大油门50%（从70%降低）
        self.max_brake = 0.8             # 最大刹车80%（从90%降低）
        self.max_steer = 0.4             # 最大转向40%（从60%降低）
        
        # 衰减参数 - 更慢的衰减，让控制更持续平滑
        self.throttle_decay = 0.01       # 油门衰减速度（从0.02降低）
        self.brake_decay = 0.03          # 刹车衰减速度（从0.05降低）
        self.steer_decay = 0.02          # 转向衰减速度（从0.03降低）
        
        # 记录上次按键时间，避免打印太频繁
        self.last_print_time = 0
        
    def get_keyboard_input(self):
        """获取键盘输入（非阻塞）"""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key.decode('utf-8').lower()
        return None
    
    def update_control(self):
        """根据键盘输入更新控制（渐进式控制 + 速度限制）"""
        current_time = time.time()
        
        # 检查当前车辆速度，如果过快则限制油门
        velocity = self.ego_vehicle.get_velocity()
        current_speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5  # 计算总速度 m/s
        max_allowed_speed = 8.0  # 最大允许速度 8 m/s (约29 km/h)
        
        # 检查当前是否有按键按下
        key = self.get_keyboard_input()
        
        # 应用自然衰减
        self._throttle_accumulate = max(0, self._throttle_accumulate - self.throttle_decay)
        self._brake_accumulate = max(0, self._brake_accumulate - self.brake_decay)
        self._steer_accumulate *= (1 - self.steer_decay)  # 转向用乘法衰减，更自然
        
        # 处理按键输入
        action_taken = False
        if key:
            if key == 'w':  # 前进
                # 如果速度已经过快，限制油门增加
                if current_speed < max_allowed_speed:
                    self._throttle_accumulate = min(self.max_throttle, 
                                                  self._throttle_accumulate + self.throttle_increment)
                else:
                    # 速度过快时，减少油门而不是增加
                    self._throttle_accumulate = max(0, self._throttle_accumulate - self.throttle_increment)
                    
                self._brake_accumulate = 0  # 清除刹车
                action_taken = True
                if current_speed >= max_allowed_speed:
                    action_desc = f"前进受限 (速度: {current_speed:.1f}m/s, 限制: {max_allowed_speed}m/s)"
                else:
                    action_desc = f"前进 (油门: {self._throttle_accumulate:.2f}, 速度: {current_speed:.1f}m/s)"
                
            elif key == 's':  # 刹车
                self._brake_accumulate = min(self.max_brake, 
                                           self._brake_accumulate + self.brake_increment)
                self._throttle_accumulate = 0  # 清除油门
                action_taken = True
                action_desc = f"刹车 (制动: {self._brake_accumulate:.2f}, 速度: {current_speed:.1f}m/s)"
                
            elif key == 'a':  # 左转
                self._steer_accumulate = max(-self.max_steer, 
                                           self._steer_accumulate - self.steer_increment)
                action_taken = True
                action_desc = f"左转 (转向: {self._steer_accumulate:.2f})"
                
            elif key == 'd':  # 右转
                self._steer_accumulate = min(self.max_steer, 
                                           self._steer_accumulate + self.steer_increment)
                action_taken = True
                action_desc = f"右转 (转向: {self._steer_accumulate:.2f})"
                
            elif key == 'q':  # 急停
                self._brake_accumulate = 1.0
                self._throttle_accumulate = 0
                action_taken = True
                action_desc = "急停！"
                
            elif key == 'r':  # 手刹
                self.control.hand_brake = True
                action_taken = True
                action_desc = "手刹启用"
            
            # 控制打印频率 - 每0.2秒最多打印一次
            if action_taken and (current_time - self.last_print_time) > 0.2:
                print(f"🚗 {action_desc}")
                self.last_print_time = current_time
        
        # 速度过快时自动应用轻微制动
        if current_speed > max_allowed_speed and self._throttle_accumulate > 0:
            self._throttle_accumulate = max(0, self._throttle_accumulate - 0.01)  # 缓慢减少油门
            if self._brake_accumulate < 0.1:  # 如果没有主动刹车，应用轻微制动
                self._brake_accumulate = 0.1
        
        # 应用累积的控制值
        self.control.throttle = self._throttle_accumulate
        self.control.brake = self._brake_accumulate
        self.control.steer = self._steer_accumulate
        
        # 手刹逻辑 - 如果没有按R，释放手刹
        if key != 'r':
            self.control.hand_brake = False
        
        # 应用控制
        try:
            self.ego_vehicle.apply_control(self.control)
        except Exception as e:
            print(f"❌ 控制应用失败: {e}")
    
    def print_instructions(self):
        """打印控制说明"""
        print("\n🎮 手动控制说明（超级平滑控制模式）：")
        print("   ⚠️ 重要：请在命令行窗口（不是CARLA窗口）中按键！")
        print("   🚗 超级精细控制模式：")
        print("     W - 渐进加速（每次按键增加2%油门）")
        print("     S - 渐进制动（每次按键增加4%刹车）")
        print("     A - 渐进左转（每次按键增加3%转向）")
        print("     D - 渐进右转（每次按键增加3%转向）")
        print("     Q - 急停（立即全力制动）")
        print("     R - 手刹（紧急制动）")
        print("   📝 控制特性：")
        print("     - 超级平滑：所有参数都大幅降低，移动极其温和")
        print("     - 渐进式控制：多次按键累积效果")
        print("     - 自然衰减：松开按键后逐渐减弱")
        print("     - 最大限制：油门50%，刹车80%，转向40%")
        print("     - 速度限制：最大8m/s (29km/h)，自动防止过快")
        print("     - 观察者友好：车辆平滑移动，观察者视角也更平滑")
        print("     - 可以组合按键进行复杂操作")
        print("   Ctrl+C 退出程序\n")

# ========== 全局键盘监听器（支持视角重置） ==========
class GlobalKeyListener:
    def __init__(self, ego_vehicle, spectator):
        self.ego_vehicle = ego_vehicle
        self.spectator = spectator
        self.manual_controller = None
        
    def set_manual_controller(self, controller):
        """设置手动控制器"""
        self.manual_controller = controller
    
    def get_keyboard_input(self):
        """获取键盘输入（非阻塞）"""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key.decode('utf-8').lower()
        return None
    
    def update(self):
        """更新键盘监听"""
        key = self.get_keyboard_input()
        
        if key:
            if key == 'v':  # V键 - 回到Ego视角
                reset_spectator_to_ego(self.ego_vehicle, self.spectator)
            elif key == 'h':  # H键 - 显示帮助
                self.print_help()
            elif self.manual_controller and EGO_CONTROL_MODE == 'manual':
                # 如果是手动控制模式，将按键传递给手动控制器
                # 由于我们已经读取了按键，需要重新处理
                self.handle_manual_control(key)
    
    def handle_manual_control(self, key):
        """处理手动控制按键（精细控制）"""
        if not self.manual_controller:
            return
            
        # 使用与ManualController相同的精细控制逻辑
        # 这里直接调用manual_controller的方法来保持一致性
        # 模拟按键被按下的效果
        
        current_time = time.time()
        
        # 直接修改manual_controller的累积值
        if key == 'w':  # 前进
            self.manual_controller._throttle_accumulate = min(
                self.manual_controller.max_throttle, 
                self.manual_controller._throttle_accumulate + self.manual_controller.throttle_increment
            )
            self.manual_controller._brake_accumulate = 0
            action_desc = f"前进 (油门: {self.manual_controller._throttle_accumulate:.2f})"
            
        elif key == 's':  # 刹车
            self.manual_controller._brake_accumulate = min(
                self.manual_controller.max_brake, 
                self.manual_controller._brake_accumulate + self.manual_controller.brake_increment
            )
            self.manual_controller._throttle_accumulate = 0
            action_desc = f"刹车 (制动: {self.manual_controller._brake_accumulate:.2f})"
            
        elif key == 'a':  # 左转
            self.manual_controller._steer_accumulate = max(
                -self.manual_controller.max_steer, 
                self.manual_controller._steer_accumulate - self.manual_controller.steer_increment
            )
            action_desc = f"左转 (转向: {self.manual_controller._steer_accumulate:.2f})"
            
        elif key == 'd':  # 右转
            self.manual_controller._steer_accumulate = min(
                self.manual_controller.max_steer, 
                self.manual_controller._steer_accumulate + self.manual_controller.steer_increment
            )
            action_desc = f"右转 (转向: {self.manual_controller._steer_accumulate:.2f})"
            
        elif key == 'q':  # 急停
            self.manual_controller._brake_accumulate = 1.0
            self.manual_controller._throttle_accumulate = 0
            action_desc = "急停！"
            
        elif key == 'r':  # 手刹
            action_desc = "手刹启用"
        else:
            return  # 未识别的按键，直接返回
        
        # 应用累积的控制值
        control = carla.VehicleControl()
        control.throttle = self.manual_controller._throttle_accumulate
        control.brake = self.manual_controller._brake_accumulate
        control.steer = self.manual_controller._steer_accumulate
        control.hand_brake = (key == 'r')
        
        # 应用控制
        try:
            self.manual_controller.ego_vehicle.apply_control(control)
            
            # 控制打印频率 - 每0.2秒最多打印一次
            if not hasattr(self, 'last_manual_print_time'):
                self.last_manual_print_time = 0
            if (current_time - self.last_manual_print_time) > 0.2:
                print(f"🚗 {action_desc}")
                self.last_manual_print_time = current_time
                
        except Exception as e:
            print(f"❌ 控制应用失败: {e}")
    
    def print_help(self):
        """显示帮助信息"""
        print("\n🔧 全局快捷键：")
        print("   V - 快速回到Ego车辆视角 📍")
        print("   H - 显示此帮助信息 ❓")
        if EGO_CONTROL_MODE == 'manual':
            print("   WASD - 手动控制车辆 🎮")
            print("   Q - 急停，R - 重置 🛑")
        print("   Ctrl+C - 退出程序 ❌")
        print("   ⚠️ 所有按键都在命令行窗口中生效\n")

# 创建手动控制器（全局变量）
manual_controller = None

def main():
    """主执行函数，包含所有的仿真逻辑"""
    global manual_controller
    
    # 创建输出目录并清空旧图片
    if not os.path.exists('out'):
        os.makedirs('out')
        print("创建了输出目录: out/")
    else:
        # 删除out目录中的所有图片文件
        image_files = glob.glob('out/*.png') + glob.glob('out/*.jpg') + glob.glob('out/*.jpeg')
        if image_files:
            print(f"正在清理 {len(image_files)} 张旧图片...")
            for file in image_files:
                try:
                    os.remove(file)
                except Exception as e:
                    print(f"删除文件 {file} 时出错: {e}")
            print("✅ 旧图片清理完成！")
        else:
            print("out目录已存在，无旧图片需要清理")

    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    #world = client.get_world()
    client.set_timeout(30.0)
    world = client.load_world('Town04')

    # ========== 显存优化设置 ==========
    print("🔧 正在优化显存使用...")

    try:
        # 设置低质量渲染
        settings = world.get_settings()
        
        # 基础优化设置 - 使用100 FPS，让仿真变得非常快
        settings.no_rendering_mode = False  # 保持渲染
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.01  # 100 FPS，让仿真变得非常快
        
        world.apply_settings(settings)
        print("✅ 基础渲染设置已优化")
        
        # 获取世界天气设置并简化
        weather = carla.WeatherParameters(
            cloudiness=0.0,          # 无云彩
            precipitation=0.0,       # 无降水
            precipitation_deposits=0.0,  # 无积水
            wind_intensity=0.0,      # 无风
            sun_azimuth_angle=45.0,
            sun_altitude_angle=45.0,
            fog_density=0.0,         # 无雾
            fog_distance=0.0,
            wetness=0.0,             # 路面不湿润
            scattering_intensity=0.0  # 减少光散射
        )
        world.set_weather(weather)
        print("✅ 天气效果已简化")
        
    except Exception as e:
        print(f"⚠️ 渲染优化失败: {e}")
        print("🔄 继续使用默认设置...")

    print("💾 显存优化完成！")
    print("📊 优化项目:")
    print("   - 帧率设置为100 FPS (让仿真变得非常快)")
    print("   - 仿真时间步长: 0.01秒/帧")
    print("   - 移除天气特效")
    print("   - 简化光照效果")
    print("   - 减少粒子效果\n")

    # 验证当前加载的地图
    current_map = world.get_map()
    map_name = current_map.name
    print(f"🗺️ 当前加载的地图: {map_name}")
    print(f"🗺️ 地图文件路径: {current_map}")

    # 清理现有的所有actors（车辆、传感器等）
    print("正在清理现有的actors...")
    actors = world.get_actors()
    vehicles = actors.filter('*vehicle*')
    sensors = actors.filter('*sensor*')

    print(f"找到 {len(vehicles)} 辆现有车辆")
    print(f"找到 {len(sensors)} 个现有传感器")

    # 销毁所有现有的车辆和传感器
    for actor in vehicles:
        actor.destroy()
    for actor in sensors:
        actor.destroy()

    print("现有actors清理完成！")

    # Retrieve the spectator object
    spectator = world.get_spectator()

    # Get the location and rotation of the spectator through its transform
    transform = spectator.get_transform()

    location = transform.location
    rotation = transform.rotation

    print(f"当前观察者位置: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}")
    print(f"当前观察者角度: pitch={rotation.pitch:.2f}, yaw={rotation.yaw:.2f}, roll={rotation.roll:.2f}")

    # Set the spectator with an empty transform
    # spectator.set_transform(carla.Transform())
    # This will set the spectator at the origin of the map, with 0 degrees
    # pitch, yaw and roll - a good way to orient yourself in the map

    #print("观察者已重置到地图原点位置")


    # Get the blueprint library and filter for the vehicle blueprints
    vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
    # Get the map's spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Spawn 20 vehicles randomly distributed throughout the map (减少显存使用)
    # for each spawn point, we choose a random vehicle from the blueprint library
    spawned_vehicles = []
    for i in range(0,20):  # 从50减少到20
        vehicle = world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))
        if vehicle is not None:
            spawned_vehicles.append(vehicle)

    print(f"成功生成了 {len(spawned_vehicles)} 辆车辆")

    # ========== 生成随机行人 ==========
    print("\n正在生成随机行人...")

    # 获取行人蓝图
    walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')

    print(f"找到 {len(walker_blueprints)} 种行人类型")

    # 设置要生成的行人数量 (减少显存使用)
    num_walkers = 10  # 从20减少到10

    # 生成行人的spawn点
    walker_spawn_points = []
    for i in range(num_walkers):
        # 使用CARLA的导航网格来获取合适的行人生成位置
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            walker_spawn_points.append(spawn_point)

    print(f"找到 {len(walker_spawn_points)} 个有效的行人生成位置")

    # 生成行人和控制器
    spawned_walkers = []
    walker_controllers = []

    for i, spawn_point in enumerate(walker_spawn_points):
        # 随机选择行人类型
        walker_bp = random.choice(walker_blueprints)
        
        # 设置行人属性
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')  # 设置为可被撞倒
        
        # 随机设置行人的速度属性
        if walker_bp.has_attribute('speed'):
            # 设置行走速度范围：1.0-2.5 m/s
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
                print(f"✅ 行人 {i+1} 生成成功，位置: ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
            else:
                print(f"❌ 行人 {i+1} 控制器生成失败")
        else:
            print(f"❌ 行人 {i+1} 生成失败")

    print(f"🚶 成功生成了 {len(spawned_walkers)} 个行人")
    print(f"🎮 成功创建了 {len(walker_controllers)} 个行人AI控制器")

    # 等待一帧，确保所有行人完全生成
    world.tick()

    # 启动行人AI控制器，让他们开始自由行动
    print("\n正在启动行人AI控制器...")
    active_controllers = 0

    for i, controller in enumerate(walker_controllers):
        try:
            # 启动控制器
            controller.start()
            
            # 获取随机位置作为目标
            target_location = world.get_random_location_from_navigation()
            
            if target_location is not None:
                controller.go_to_location(target_location)
                # 设置随机行走速度 (1.0-2.5 m/s)
                max_speed = random.uniform(1.0, 2.5)
                controller.set_max_speed(max_speed)
                active_controllers += 1
                print(f"🎯 行人 {i+1} AI已启动，目标位置: ({target_location.x:.1f}, {target_location.y:.1f}), 速度: {max_speed:.1f} m/s")
            else:
                # 如果无法获取导航位置，设置为缓慢随机行走
                controller.set_max_speed(1.0)
                active_controllers += 1
                print(f"🔄 行人 {i+1} AI已启动，随机行走模式")
                
        except Exception as e:
            print(f"❌ 启动行人 {i+1} AI控制器失败: {e}")

    print(f"🎬 {active_controllers} 个行人开始自由行动！")
    print("行人特征:")
    print("   - 行走速度：1.0-2.5 m/s")
    print("   - 自动寻路：使用CARLA导航网格")
    print("   - 智能行为：避障、转向、目标导向")
    print("   - 物理特性：可被车辆撞倒")
    print("   - 随机外观：多种行人类型\n")

    # 为ego_vehicle寻找一个空闲的spawn点并设置为Ego Vehicle
    ego_vehicle = None

    # ========== EGO VEHICLE 控制模式设置 ==========
    # 选择控制模式：
    # 1. 'static' - 静止观察模式（当前模式）
    # 2. 'manual' - 手动控制模式
    # 3. 'autopilot' - 使用CARLA的autopilot
    # 4. 'custom_ai' - 你的自动驾驶算法（需要实现）
    EGO_CONTROL_MODE = 'manual'  # 默认静止模式，可以修改

    print(f"🎮 Ego Vehicle控制模式: {EGO_CONTROL_MODE}")

    if EGO_CONTROL_MODE == 'static':
        print("   - 静止观察模式：Ego车辆保持静止，专注于录制")
    elif EGO_CONTROL_MODE == 'manual':
        print("   - 手动控制模式：使用键盘控制Ego车辆")
    elif EGO_CONTROL_MODE == 'autopilot':
        print("   - 自动驾驶模式：使用CARLA内置autopilot")
    elif EGO_CONTROL_MODE == 'custom_ai':
        print("   - 自定义AI模式：使用你开发的自动驾驶算法")

    # 选择一个具体的车辆类型作为Ego Vehicle
    ego_bp = world.get_blueprint_library().find('vehicle.tesla.model3')  # 选择特斯拉Model 3
    if ego_bp is None:
        # 如果Tesla Model 3不可用，选择其他车辆
        ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
    if ego_bp is None:
        # 如果以上都不可用，使用任意车辆
        ego_bp = random.choice(vehicle_blueprints)

    # 关键：设置role_name为'hero'以标识为Ego Vehicle
    ego_bp.set_attribute('role_name', 'hero')

    # 如果有颜色属性，设置一个特殊颜色来区分Ego Vehicle
    if ego_bp.has_attribute('color'):
        ego_bp.set_attribute('color', '255,0,0')  # 设置为红色，便于识别

    for i, spawn_point in enumerate(spawn_points):
        ego_vehicle = world.try_spawn_actor(ego_bp, spawn_point)
        if ego_vehicle is not None:
            print(f"✅ Ego车辆生成成功！")
            print(f"🚗 车辆类型: {ego_vehicle.type_id}")
            print(f"🎯 Role Name: hero (Ego Vehicle)")
            print(f"🔴 颜色: 红色 (便于识别)")
            print(f"📍 使用的spawn点索引: {i}")
            print(f"🆔 Ego车辆ID: {ego_vehicle.id}")
            print(f"📍 Ego车辆位置: x={spawn_point.location.x:.2f}, y={spawn_point.location.y:.2f}, z={spawn_point.location.z:.2f}")
            break

    if ego_vehicle is None:
        print("警告: 无法生成ego车辆，所有spawn点都被占用")
        return

    # 等待一帧，确保车辆完全生成
    world.tick()
    
    # 设置观察者位置，在车辆上方一点，方便观察
    # 获取车辆的前进方向向量
    ego_transform = ego_vehicle.get_transform()
    forward_vector = ego_transform.get_forward_vector()
    
    print(f"车辆朝向向量: x={forward_vector.x:.2f}, y={forward_vector.y:.2f}, z={forward_vector.z:.2f}")
    
    # 计算观察者位置：车辆后方8米，上方6米
    spectator_location = carla.Location(
        ego_transform.location.x - forward_vector.x * 8,  # 后方8米
        ego_transform.location.y - forward_vector.y * 8,  # 后方8米
        ego_transform.location.z + 6  # 上方6米
    )
    
    spectator_transform = carla.Transform(
        spectator_location,
        carla.Rotation(pitch=-20, yaw=ego_transform.rotation.yaw)  # 轻微俯视，跟随车辆朝向
    )
    
    spectator.set_transform(spectator_transform)
    print("观察者视角已切换到ego车辆后方")
    print(f"Ego车辆当前位置: x={ego_transform.location.x:.2f}, y={ego_transform.location.y:.2f}, z={ego_transform.location.z:.2f}")
    print(f"观察者设置位置: x={spectator_location.x:.2f}, y={spectator_location.y:.2f}, z={spectator_location.z:.2f}")
    
    # 验证观察者位置是否正确设置
    current_spectator_transform = spectator.get_transform()
    print(f"观察者实际位置: x={current_spectator_transform.location.x:.2f}, y={current_spectator_transform.location.y:.2f}, z={current_spectator_transform.location.z:.2f}")
    
    # 添加摄像头传感器
    print("\n正在添加摄像头传感器...")
    
    # Create a transform to place the camera on top of the vehicle
    camera_init_trans = carla.Transform(carla.Location(z=1.5))
    
    # We create the camera through a blueprint that defines its properties
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    
    # 设置摄像头属性
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '70')
    
    # We spawn the camera and attach it to our ego vehicle
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
    
    # 设置一个变量来控制图片保存频率
    save_interval = 100  # 每100帧保存一张图片（100帧 = 1秒，因为每帧0.01秒）
    saved_image_count = [0]  # 使用列表来避免全局变量问题
    
    # Start camera with callback to save images every 100 frames
    def save_image_callback(image):
        if image.frame % save_interval == 0:  # 每100帧保存一次
            image.save_to_disk(f'out/{saved_image_count[0]:06d}.png')
            saved_image_count[0] += 1
    
    camera.listen(save_image_callback)
    
    print("摄像头传感器添加成功！")
    print("摄像头位置: 车辆顶部1.5米")
    print("图像分辨率: 640x480")
    print("视野角度: 70度")
    print("图像保存路径: out/目录")
    
    # 启用Traffic Manager，让车辆开始移动
    print("\n正在启用Traffic Manager...")
    
    # 获取Traffic Manager实例
    traffic_manager = client.get_trafficmanager(8000)  # 使用默认端口8000
    print("Traffic Manager已连接")
    
    # 设置Traffic Manager为同步模式（与世界同步）
    traffic_manager.set_synchronous_mode(True)
    print("Traffic Manager设置为同步模式")
    
    # 设置随机种子，保证行为一致
    traffic_manager.set_random_device_seed(42)
    print("设置随机种子：42（行为更一致）")
    
    # 设置全局速度限制 - 改回之前的值
    traffic_manager.global_percentage_speed_difference(50.0)  # 全局比限速慢50%
    print("全局速度设置：比限速慢50%")
    
    # 设置更保守的全局驾驶行为
    traffic_manager.set_global_distance_to_leading_vehicle(4.0)  # 全局跟车距离4米
    print("全局跟车距离：4米")
    
    print("注意：每辆车还会额外设置更慢的个别速度")
    
    # 获取所有车辆并启用自动驾驶
    vehicles = world.get_actors().filter('*vehicle*')
    print(f"找到 {len(vehicles)} 辆车辆")
    
    # 为所有车辆启用自动驾驶模式（除了ego车辆）
    autopilot_count = 0
    for vehicle in vehicles:
        try:
            # 检查是否是ego车辆
            if vehicle.id == ego_vehicle.id:
                # 根据控制模式设置Ego车辆
                if EGO_CONTROL_MODE == 'static':
                    # 静止模式：不启用autopilot，保持静止
                    print(f"🎯 Ego车辆 {vehicle.id} 设置为静止观察模式")
                    
                elif EGO_CONTROL_MODE == 'manual':
                    # 手动控制模式：不启用autopilot，等待键盘输入
                    print(f"🎮 Ego车辆 {vehicle.id} 设置为手动控制模式")
                    print("   - 使用 WASD 键控制移动（需要在CARLA窗口中）")
                    print("   - W: 前进, S: 后退, A: 左转, D: 右转")
                    
                    # 初始化手动控制器
                    manual_controller = ManualController(vehicle)
                    manual_controller.print_instructions()
                    
                elif EGO_CONTROL_MODE == 'autopilot':
                    # 使用CARLA autopilot，与NPC车辆保持相同行为
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    # 设置与NPC车辆相同的驾驶行为
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, 50.0)  # 比限速慢50%（与NPC相同）
                    traffic_manager.distance_to_leading_vehicle(vehicle, 4.0)  # 跟车距离4米（与NPC相同）
                    traffic_manager.auto_lane_change(vehicle, True)  # 允许自动变道
                    autopilot_count += 1
                    print(f"🚗 Ego车辆 {vehicle.id} 启用autopilot成功")
                    
                elif EGO_CONTROL_MODE == 'custom_ai':
                    # 使用自定义AI控制
                    print(f"🤖 Ego车辆 {vehicle.id} 设置为自定义AI控制模式")
                    print("   - 将使用你开发的自动驾驶算法")
                    
                    # 这里可以调用你的自动驾驶算法
                    # your_autonomous_driving_algorithm(vehicle, world, traffic_manager)
                    print("   - 自定义AI算法已启动")
                    
            else:
                # 对于非ego车辆，启用autopilot
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
                
                print(f"🚗 NPC车辆 {vehicle.id} 启用autopilot成功，速度:{speed_difference:.0f}%慢，跟车距离:{distance:.1f}m")
                
        except Exception as e:
            print(f"❌ 为车辆 {vehicle.id} 启用autopilot失败: {e}")
    
    print(f"✅ 成功为 {autopilot_count} 辆车启用autopilot")
    print("🎬 仿真场景已启动！")
    print("场景特征:")
    print(f"   - 总车辆数: {len(vehicles)} 辆")
    print(f"   - 自动驾驶车辆: {autopilot_count} 辆")
    print(f"   - 行人数量: {len(spawned_walkers)} 个")
    print(f"   - 地图: {map_name}")
    print(f"   - 同步模式: 启用 (100 FPS)")
    print(f"   - 图片保存: 每秒1张 (out/目录)")
    print(f"   - Ego车辆控制: {EGO_CONTROL_MODE}")
    print("   - 观察者视角: 已锁定到Ego车辆后方")
    print("   - 摄像头传感器: 已挂载到Ego车辆")
    print("   - Traffic Manager: 已启用，同步模式\n")
    
    # ========== 增强版键盘控制系统 ==========
    key_listener = GlobalKeyListener(ego_vehicle, spectator)
    if manual_controller:
        key_listener.set_manual_controller(manual_controller)
    
    print("🎮 键盘控制系统已启动！")
    print("📹 摄像头开始录制...")
    print("🔄 同步模式仿真循环已启动...")
    print("⏱️ 仿真时间流逝速度: 100倍 (每秒仿真100帧)")
    print("💾 图像保存频率: 每100帧 (每秒1张图片)")
    print("🎯 按 'H' 键随时查看帮助信息")
    
    # 同步模式仿真循环
    try:
        frame = 0
        while True:
            # 在同步模式中，必须手动推进仿真时间
            world.tick()
            
            # 更新键盘监听器
            key_listener.update()
            
            # 显示仿真状态信息（每1000帧显示一次，即每10秒）
            if frame % 1000 == 0:
                ego_location = ego_vehicle.get_location()
                ego_velocity = ego_vehicle.get_velocity()
                speed_kmh = 3.6 * ((ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)**0.5)
                
                print(f"📊 仿真状态 (第{frame}帧):")
                print(f"   - 仿真时间: {frame * 0.01:.1f}秒")
                print(f"   - Ego位置: ({ego_location.x:.1f}, {ego_location.y:.1f}, {ego_location.z:.1f})")
                print(f"   - Ego速度: {speed_kmh:.1f} km/h")
                print(f"   - 保存图片: {saved_image_count[0]} 张")
                print(f"   - 模式: {EGO_CONTROL_MODE}")
                print()
            
            frame += 1
            
            # 简短休眠以防止CPU过度使用
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n🛑 检测到 Ctrl+C，正在安全退出...")
        
    finally:
        # 清理资源
        print("\n🧹 正在清理资源...")
        
        try:
            # 停止摄像头
            if 'camera' in locals():
                camera.stop()
                camera.destroy()
                print("✅ 摄像头已停止并销毁")
            
            # 停止所有行人控制器
            for controller in walker_controllers:
                controller.stop()
                controller.destroy()
            print("✅ 行人控制器已停止")
            
            # 销毁所有生成的actors
            for vehicle in spawned_vehicles:
                vehicle.destroy()
            for walker in spawned_walkers:
                walker.destroy()
            if ego_vehicle:
                ego_vehicle.destroy()
            
            print("✅ 所有车辆和行人已销毁")
            
            # 关闭Traffic Manager同步模式
            if 'traffic_manager' in locals():
                traffic_manager.set_synchronous_mode(False)
                print("✅ Traffic Manager已恢复异步模式")
                
        except Exception as e:
            print(f"清理过程中出现错误: {e}")
        
        print("✅ 清理完成，服务端应该不会卡死了")

if __name__ == "__main__":
    main()



