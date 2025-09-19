# 无人机拼接建图和评估算法

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目是一个基于Python的无人机图像拼接建图和评估系统，主要用于处理无人机采集的图像数据，实现图像拼接、热成像处理和图像质量评估等功能。项目集成了多种图像处理算法，包括相位相关算法(POC)、特征匹配算法以及多种图像质量评估指标。

## 主要功能

### 🔧 核心功能
- **图像拼接**: 基于相位相关算法(POC)和特征匹配的图像拼接
- **热成像处理**: 红外热成像数据的实时处理和可视化
- **图像融合**: 支持加权融合算法的图像无缝拼接
- **质量评估**: 多种图像质量评估指标(PSNR、SSIM、MSE、互信息)

### 📊 评估指标
- **PSNR (峰值信噪比)**: 评估图像重建质量
- **SSIM (结构相似性)**: 评估图像结构保持度
- **MSE (均方误差)**: 评估像素级误差
- **互信息**: 评估图像间的信息相关性

### 🛠️ 技术特性
- **多通信协议支持**: 串口、TCP、UDP通信
- **实时数据处理**: 支持实时图像流处理
- **多种特征检测器**: ORB、SIFT、SURF、AKAZE、KAZE
- **图像预处理**: 下采样、旋转、噪声添加等

## 项目结构

```
mapcombination/
├── main.py                 # 主程序入口
├── imregpoc.py            # 相位相关算法实现
├── fusion.py              # 图像融合算法
├── evaluation.py          # 图像质量评估
├── image_process.py       # 图像预处理工具
├── PSNR.py               # PSNR计算模块
├── SSIM.py               # SSIM计算模块
├── judge.py              # MSE计算模块
├── infrared.py           # 红外数据处理
├── link.py               # 通信模块(串口/TCP/UDP)
├── image.py              # 图像处理工具类
├── display.py            # 图像显示工具
├── draw.py               # 图像绘制工具
├── plot.py               # 数据可视化
├── mapping.py            # 地图处理
├── mapcreating.py        # 地图创建
├── mapdividing.py        # 地图分割
├── test.py               # 测试脚本
└── data/                 # 数据目录
    ├── heatmap/          # 热成像数据
    ├── mappiece/         # 地图片段
    └── yongzhou-small/   # 测试数据集
```

## 安装要求

### 系统要求
- Python 3.7+
- Windows/Linux/macOS

### 依赖包
```bash
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
pip install scikit-learn
pip install pyserial
```

## 使用方法

### 1. 基本图像拼接
```python
import imregpoc
import cv2

# 读取图像
img1 = cv2.imread("data/map1_1.jpg", 0)
img2 = cv2.imread("data/map1_2.jpg", 0)

# 创建POC匹配器
matcher = imregpoc.imregpoc(img1, img2)

# 执行拼接
result = matcher.stitching()
```

### 2. 图像质量评估
```python
import PSNR
import SSIM
import judge

# 计算PSNR
psnr_value = PSNR.calculate_psnr(img1, img2, crop_border=4)

# 计算SSIM
ssim_value = SSIM.calculate_ssim(img1, img2, crop_border=4)

# 计算MSE
mse_value = judge.MSE_calculate(img1, img2)
```

### 3. 红外数据处理
```python
import infrared
import image

# 创建红外处理器
ir = infrared.Infrared()
img_processor = image.Image()

# 处理红外数据
temp_matrix = ir.calculateCelsius(frame_data)
normalized_img = img_processor.normalization(temp_matrix)
```

### 4. 实时数据处理
```python
# 运行主程序进行实时处理
python main.py
```

## 算法说明

### 相位相关算法 (POC)
本项目实现了基于相位相关的图像配准算法，具有以下特点：
- 高精度亚像素级配准
- 支持旋转、缩放、平移变换
- 对数极坐标变换处理
- 多种子像素拟合方法

### 图像融合算法
- **加权融合**: 基于Sigmoid函数的权重计算
- **重叠区域处理**: 智能重叠区域融合
- **无缝拼接**: 减少拼接痕迹

### 特征匹配算法
支持多种特征检测器：
- **ORB**: 快速二进制特征检测
- **SIFT**: 尺度不变特征变换
- **SURF**: 加速鲁棒特征
- **AKAZE**: 加速KAZE特征
- **KAZE**: 非线性尺度空间特征

## 性能评估

项目提供了全面的图像质量评估工具：

| 指标 | 描述 | 范围 |
|------|------|------|
| PSNR | 峰值信噪比 | 越高越好 |
| SSIM | 结构相似性 | 0-1，越接近1越好 |
| MSE | 均方误差 | 越小越好 |
| MI | 互信息 | 越高越好 |

## 通信协议

支持多种通信方式：
- **串口通信**: 用于硬件设备连接
- **TCP通信**: 网络数据传输
- **UDP通信**: 实时数据流传输

## 测试数据

项目包含测试数据集：
- `data/heatmap/`: 热成像测试数据
- `data/mappiece/`: 地图片段数据
- `data/yongzhou-small/`: 永州地区测试数据

## 运行示例

```bash
# 运行主程序
python main.py

# 运行评估测试
python evaluation.py

# 运行图像处理测试
python test.py
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

