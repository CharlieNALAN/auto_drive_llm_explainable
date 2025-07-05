import carla
import random
import time

actor_list=[]
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(100.0)

    world=client.get_world()
    blueprint_library=world.get_blueprint_library()
    vehicle_bp=blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point=random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)
    vehicle.apply_control(carla.VehicleControl(throttle=1,steer=0))
    time.sleep(30)

finally:
    for actor in actor_list:
        actor.destroy()
    print("All actors destroyed")
    


