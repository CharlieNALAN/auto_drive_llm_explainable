#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显示CARLA中所有可用的天气选项
"""

import carla
import re

def find_weather_presets():
    """获取所有可用的天气预设"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def main():
    print("CARLA 可用天气选项:")
    print("=" * 80)
    
    weather_presets = find_weather_presets()
    
    for i, (weather, name) in enumerate(weather_presets):
        # 获取天气参数的详细信息
        sun_altitude = weather.sun_altitude_angle
        sun_azimuth = weather.sun_azimuth_angle
        clouds = weather.cloudiness
        rain = weather.precipitation
        wind = weather.wind_intensity
        fog = weather.fog_density
        
        # 判断是否为白天 (太阳高度角大于0通常是白天)
        time_of_day = "白天" if sun_altitude > 0 else "夜晚"
        
        print(f"{i:2d}: {name}")
        print(f"    时间: {time_of_day} (太阳高度: {sun_altitude:.1f}°)")
        print(f"    太阳方位: {sun_azimuth:.1f}°")
        print(f"    云量: {clouds:.1f}%, 降雨: {rain:.1f}%")
        print(f"    风力: {wind:.1f}%, 雾霾: {fog:.1f}%")
        print()
    
    print("=" * 80)
    print("推荐的白天天气选项:")
    for i, (weather, name) in enumerate(weather_presets):
        if weather.sun_altitude_angle > 0:
            print(f"  --weather {i}: {name}")
    
    print("\n推荐的夜晚天气选项:")
    for i, (weather, name) in enumerate(weather_presets):
        if weather.sun_altitude_angle <= 0:
            print(f"  --weather {i}: {name}")
    
    print("\n" + "=" * 80)
    print("使用方法:")
    print("  python carla_sim_llm.py --weather <索引号>")
    print("  例如:")
    print("    python carla_sim_llm.py --weather 1  # 白天晴朗")
    print("    python carla_sim_llm.py --weather 6  # 白天多云")

if __name__ == "__main__":
    main() 