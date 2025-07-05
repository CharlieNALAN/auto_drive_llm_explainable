import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass


import carla
#Import the library Transform used to explicitly spawn an actor
from carla import Transform, Location, Rotation
from carla import Map
from carla import Vector3D
 
import random
import time

actor_list = []

#--------------------------------Initialization-----------------------------------#
try:
	

	##General connection to server and get blue_print
	client = carla.Client('localhost',2000)
	client.set_timeout(5.0)

	world = client.get_world()
	mp = world.get_map()#get the map of the current world.
	blueprint_library = world.get_blueprint_library()
	

	##Search for specific actor and get its blueprint.
	vehicle_bp1 = blueprint_library.filter('tt')[0]
	vehicle_bp2 = blueprint_library.filter('model3')[0]

	##Change the color of the actors (attribute)
	vehicle_bp1.set_attribute('color', '255,255,255')# change the color to white
	vehicle_bp2.set_attribute('color', '0,0,0')
	
	##Spawn an actor at specified point by using Transform
	#lag Vehicle
	spawn_point_v1 = Transform(Location(x=-65.4, y=4.0, z=11), 
		Rotation(pitch=0, yaw=180, roll=0))

	#Lead Vehicle
	spawn_point_v2 = Transform(Location(x=-105.4, y=4.0, z=11), 
		Rotation(pitch=0, yaw=180, roll=0))
		
	vehicle1 = world.spawn_actor(vehicle_bp1, spawn_point_v1)
	vehicle2 = world.spawn_actor(vehicle_bp2, spawn_point_v2)

	actor_list.append(vehicle1)
	actor_list.append(vehicle2)
	
	##添加行人
	# 获取行人蓝图
	walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
	walker_controller_bp = blueprint_library.find('controller.ai.walker')
	
	# 在车辆后面生成几个行人
	pedestrian_spawn_points = [
		Transform(Location(x=-50.0, y=1.0, z=11.0), Rotation(pitch=0, yaw=0, roll=0)),
		Transform(Location(x=-48.0, y=7.0, z=11.0), Rotation(pitch=0, yaw=180, roll=0)),
		Transform(Location(x=-45.0, y=4.0, z=11.0), Rotation(pitch=0, yaw=90, roll=0)),
		Transform(Location(x=-52.0, y=0.5, z=11.0), Rotation(pitch=0, yaw=270, roll=0))
	]
	
	# 生成行人和控制器
	walkers = []
	walker_controllers = []
	
	for i, spawn_point in enumerate(pedestrian_spawn_points):
		walker_bp = random.choice(walker_blueprints)
		# 设置行人属性
		if walker_bp.has_attribute('is_invincible'):
			walker_bp.set_attribute('is_invincible', 'false')
		if walker_bp.has_attribute('speed'):
			walker_bp.set_attribute('speed', str(random.uniform(1.0, 2.5)))  # 设置行走速度
		
		# 生成行人
		walker = world.spawn_actor(walker_bp, spawn_point)
		walkers.append(walker)
		actor_list.append(walker)
		
		# 为每个行人生成AI控制器
		walker_controller = world.spawn_actor(walker_controller_bp, Transform(), attach_to=walker)
		walker_controllers.append(walker_controller)
		actor_list.append(walker_controller)
		
		print(f"生成行人 {i+1} 在位置: {spawn_point.location}")
	
	# 等待所有行人完全生成
	world.tick()
	
	# 启动行人AI控制器
	for controller in walker_controllers:
		# 设置随机目标点让行人走动
		target_location = world.get_random_location_from_navigation()
		if target_location is not None:
			controller.start()
			controller.go_to_location(target_location)
			controller.set_max_speed(random.uniform(1.0, 2.5))  # 设置最大速度
		else:
			# 如果没有导航点，让行人原地随机移动
			controller.start()
			controller.set_max_speed(1.5)
	
	##Keep the leading vehicle static
	vehicle2.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=1.0, hand_brake=True))

#---------------------------------Control Part------------------------------------#
	while True:
		x_v1 = vehicle1.get_location().x
		x_v2 = vehicle2.get_location().x

		y_v1 = vehicle1.get_location().y
		y_v2 = vehicle2.get_location().y

		if abs(x_v2 - x_v1) > 12:#Drive with "safe distance"
		#Control the vehicle: 
			vehicle1.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
			
		elif abs(x_v2 - x_v1) <= 12:#When the lag vehicle  leave the "safe zone"

			while True:
				y_v1 = vehicle1.get_location().y
				y_v2 = vehicle2.get_location().y
				
				#lag vehicle turn left.
				vehicle1.apply_control(carla.VehicleControl(throttle=0.2, steer=-0.5))
				#if the rotation angle is too big, change the direction of steer
				if abs(y_v1 - y_v2) > 2:
					vehicle1.apply_control(carla.VehicleControl(throttle=0.5, steer=0.4))
					break

			while True:			
				x_v1 = vehicle1.get_location().x
				x_v2 = vehicle2.get_location().x


				if abs(x_v1 - x_v2) < 1.0:
				#The number in this if statement depends on the safe distance set before.
					vehicle1.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
					if abs(x_v1 - x_v2) > 25:
						break

			break
	time.sleep(100)

finally:
	for actor in actor_list:
		actor.destroy()
	print("All cleaned up!")




