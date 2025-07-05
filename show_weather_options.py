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
    print("=" * 50)
    
    weather_presets = find_weather_presets()
    
    for i, (weather, name) in enumerate(weather_presets):
        print(f"{i:2d}: {name}")
    
    print("\n" + "=" * 50)
    print("使用方法:")
    print("  python carla_sim_llm.py --weather <索引号>")
    print("  例如:")
    print("    python carla_sim_llm.py --weather 0  # 默认天气")
    print("    python carla_sim_llm.py --weather 5  # 第6个天气选项")
    print("    python carla_sim_llm.py --weather 10 # 第11个天气选项")

if __name__ == "__main__":
    main() 