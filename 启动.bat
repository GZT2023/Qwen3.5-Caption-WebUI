@echo off
chcp 65001 >nul

echo 正在初始化环境，请耐心等待...
echo 激活虚拟环境中...

call .\venv\Scripts\activate

echo 正在启动程序，请勿关闭本窗口...
echo 首次运行需要加载模型，需要较长时间，请稍候...

.\venv\Scripts\python.exe -s gui.py --inbrowser

pause