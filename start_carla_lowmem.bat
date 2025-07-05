@echo off
echo 正在以低显存模式启动CARLA服务器...
echo.
echo 优化设置:
echo - 低渲染质量
echo - 减少特效
echo - 降低分辨率
echo.

REM 设置CARLA启动参数
set CARLA_PARAMS=-quality-level=Low -benchmark -fps=20 -windowed -ResX=1280 -ResY=720

REM 启动CARLA (请根据你的CARLA安装路径修改)
REM 示例路径，需要修改为你的实际安装路径
"C:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe" %CARLA_PARAMS%

pause