# 使用指南：集成后的LLM可解释自动驾驶系统

## 🚀 快速开始

### 1. 安装依赖

```bash
cd auto_drive_llm_explainable
pip install -r requirements.txt
```

### 2. 复制车道检测模型（重要！）

将第二个项目中已经训练好的模型文件复制到当前目录：

```bash
# 方法1：直接复制
cp ../self-driving-carla-main/lane_detection/Deeplabv3+\(MobilenetV2\).pth ./

# 方法2：如果路径不同，请找到模型文件并复制
find .. -name "Deeplabv3+*" -type f
```

### 3. 运行集成测试

```bash
python test_integration.py
```

如果所有测试通过，说明集成成功！

### 4. 启动CARLA服务器

```bash
# 启动CARLA（替换为您的CARLA路径）
cd /path/to/your/carla
./CarlaUE4.exe -quality-level=Low
```

### 5. 运行自动驾驶系统

```bash
python main.py
```

## 🔧 核心改进

### 集成的组件

- ✅ **车道检测**：使用第二个项目的DeepLabV3+ + MobileNetV2模型
- ✅ **相机几何**：精确的像素到世界坐标转换
- ✅ **纯追踪控制**：经过验证的路径跟踪算法
- ✅ **自适应速度**：基于道路曲率的智能速度调节
- ✅ **LLM解释**：保持原有的自然语言解释功能

### 控制流程

```
摄像头图像 → 车道检测 → 轨迹生成 → 纯追踪控制 → 车辆控制
     ↓
  LLM解释器 ← 行为规划 ← 目标检测
```

## 📊 性能特点

- **稳定行驶**：不再左右摇摆
- **精确转弯**：基于曲率的自适应速度
- **鲁棒回退**：车道检测失败时使用地图导航
- **实时解释**：保持原有的LLM解释能力

## 🛠 故障排除

### 常见问题

1. **ImportError: No module named 'segmentation_models_pytorch'**
   ```bash
   pip install segmentation_models_pytorch albumentations
   ```

2. **车道检测模型未找到**
   - 确保已复制 `Deeplabv3+(MobilenetV2).pth` 文件
   - 系统会自动回退到计算机视觉方法

3. **车辆仍然不稳定**
   - 检查 `test_integration.py` 的输出
   - 确保所有模块都通过测试

### 调试选项

```bash
# 使用不同的LLM
python main.py --llm-model=openai  # 需要API密钥
python main.py --llm-model=local   # 默认

# 降低分辨率以提高性能
python main.py --res=640x480

# 禁用可视化
python main.py --no-rendering
```

## 📈 预期效果

集成后的系统应该表现出：

- ✅ 平滑的车道跟踪
- ✅ 智能的转弯行为
- ✅ 稳定的直线行驶
- ✅ 清晰的行为解释

如果车辆仍有问题，请检查CARLA版本兼容性和模型文件完整性。 