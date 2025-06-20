# 集成指南：修复车辆控制问题

本文档说明了我们如何将第二个项目（self-driving-carla-main）的成功车道检测和控制算法集成到第一个项目中，以解决车辆左右摇摆和撞车的问题。

## 🚗 问题描述

原始的LLM可解释自动驾驶系统存在以下问题：
- 车辆启动后左右乱晃
- 经常撞墙或偏离道路
- 车道检测不准确
- 路径规划不稳定

## 🔧 解决方案

我们集成了第二个项目已经验证可行的以下组件：

### 1. 改进的车道检测 (`src/perception/lane_detection.py`)
- **相机几何变换**：集成了精确的像素到世界坐标转换
- **深度学习模型**：使用DeepLabV3+ + MobileNetV2架构
- **多项式拟合**：对检测到的车道线进行三次多项式拟合
- **轨迹生成**：直接生成车辆坐标系下的轨迹点

### 2. 纯追踪控制器 (`src/control/pure_pursuit.py`)
- **纯追踪算法**：基于前瞻距离的路径跟踪
- **PID速度控制**：平滑的油门/刹车控制
- **自适应前瞻距离**：基于车速动态调整
- **圆线相交算法**：精确的目标点选择

### 3. 相机几何模块 (`src/perception/camera_geometry.py`)
- **坐标系转换**：完整的图像到世界坐标变换
- **ISO8855标准**：符合车辆动力学坐标系
- **预计算网格**：提高实时性能
- **俯仰角补偿**：考虑相机安装角度

### 4. 轨迹工具 (`src/utils/trajectory_utils.py`)
- **曲率计算**：用于自适应速度控制
- **地图回退**：车道检测失败时的安全策略
- **控制接口**：统一的车辆控制函数

## 🚀 主要改进

### 控制逻辑重构
```python
# 旧的复杂流程
perception -> prediction -> behavior_planning -> path_planning -> control

# 新的简化流程（基于工作项目）
lane_detection -> trajectory_generation -> pure_pursuit_control
```

### 速度自适应控制
```python
# 根据道路曲率调整速度
max_curvature = get_curvature(trajectory)
if max_curvature > 0.005:
    move_speed = max(3.0, 5.56 - 20 * max_curvature)  # 降速过弯
else:
    move_speed = 5.56  # 直道保持20km/h
```

### 鲁棒性增强
- **多重回退机制**：车道检测 -> 地图导航 -> 直行
- **异常处理**：完善的错误捕获和恢复
- **参数验证**：输入数据的有效性检查

## 📁 文件修改列表

### 新增文件
- `src/perception/camera_geometry.py` - 相机几何变换
- `src/control/pure_pursuit.py` - 纯追踪控制器  
- `src/utils/trajectory_utils.py` - 轨迹处理工具
- `test_integration.py` - 集成测试脚本

### 修改文件
- `src/perception/lane_detection.py` - 完全重写车道检测
- `main.py` - 重构主控制循环
- `requirements.txt` - 添加新依赖项

### 保留文件（用于LLM解释）
- `src/explainability/llm_explainer.py` - 保持原有LLM解释功能
- `src/planning/behavior_planner.py` - 保留行为规划用于解释
- 其他模块保持不变

## 🧪 测试方法

1. **运行集成测试**：
```bash
cd auto_drive_llm_explainable
python test_integration.py
```

2. **安装新依赖**：
```bash
pip install -r requirements.txt
```

3. **运行主程序**：
```bash
# 启动CARLA服务器
python run_carla.py

# 在另一个终端运行自动驾驶系统
python main.py
```

## 🎯 预期效果

集成后的系统应该具备：
- ✅ 平滑的车道跟踪，不再左右摇摆
- ✅ 准确的转弯控制，基于道路曲率自适应速度
- ✅ 稳定的直线行驶
- ✅ 保持原有的LLM解释功能
- ✅ 鲁棒的错误处理和回退机制

## 🔍 调试提示

如果仍有问题，请检查：

1. **车道检测模型**：确保能加载第二个项目的预训练模型
2. **坐标系对齐**：验证相机几何参数与CARLA设置匹配
3. **控制参数**：可调整PID参数和前瞻距离
4. **速度设置**：检查目标速度是否合理

## 🔧 参数调优

### 控制器参数
```python
# PID控制器增益
Kp = 2.0    # 比例增益
Ki = 0.0    # 积分增益  
Kd = 0.0    # 微分增益

# 纯追踪参数
K_dd = 0.4           # 前瞻距离系数
wheel_base = 2.65    # 车辆轴距
waypoint_shift = 1.4 # 路径点偏移
```

### 速度参数
```python
max_speed = 5.56      # 最大速度 (20 km/h)
min_speed = 3.0       # 最小速度 (10.8 km/h)
curvature_threshold = 0.005  # 曲率阈值
```

通过这些改进，车辆应该能够稳定地沿车道行驶，而不会出现左右摇摆或撞车的问题。 