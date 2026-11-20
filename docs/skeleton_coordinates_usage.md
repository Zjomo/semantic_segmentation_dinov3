# 骨架坐标提取功能使用说明

## 概述

本功能实现了以左上角为原点的坐标系，用于提取全局地图中的骨架坐标。

## 坐标系说明

- **原点**: 左上角 (0, 0)
- **X轴**: 水平向右
- **Y轴**: 垂直向下
- **单位**: 默认像素，可选择转换为微米

## 功能特点

1. **完整骨架提取**: 提取所有骨架分支的坐标点
2. **最长路径提取**: 提取最长主路径的坐标点
3. **物理单位转换**: 支持将像素坐标转换为微米
4. **JSON 格式输出**: 保存为结构化的 JSON 文件
5. **可视化支持**: 在图像上标记骨架坐标点

## 使用方法

### 1. 运行主程序

```bash
python train/5_core_clip_simulate.py
```

### 2. 设置模式

在 `5_core_clip_simulate.py` 中设置:

```python
MODE = "snake"  # 使用蛇形扫描模式
```

### 3. 执行扫描

程序会自动执行以下步骤:
1. 选择 ROI 区域（可选）
2. 执行蛇形扫描
3. 构建全局地图
4. 提取骨架坐标
5. 保存结果

## 输出文件

扫描完成后，会在输出目录生成以下文件:

### 主要输出

- **skeleton_coordinates.json**: 骨架坐标数据（JSON 格式）
- **skeleton_coordinates_visualization.png**: 骨架坐标可视化结果

### 全局地图

- **global_map.png**: 全局地图（含完整骨架，红色）
- **global_mask.png**: 原始语义掩码
- **global_mask_color.png**: 彩色语义掩码（含完整骨架，黄色）

### 其他

- **scan_process.mp4**: 扫描过程视频（带骨架）
- **scan_process_clean.mp4**: 纯分割视频
- **motion_log.txt**: 运动日志

## JSON 输出格式

`skeleton_coordinates.json` 文件包含以下字段:

```json
{
  "skeleton_points": [[x1, y1], [x2, y2], ...],
  "longest_path_points": [[x1, y1], [x2, y2], ...],
  "skeleton_count": 1234,
  "longest_path_count": 567,
  "physical_units": false,
  "um_per_pixel": 1.0
}
```

### 字段说明

- `skeleton_points`: 完整骨架点列表，每个点为 [x, y] 坐标
- `longest_path_points`: 最长路径点列表，每个点为 [x, y] 坐标
- `skeleton_count`: 完整骨架点的总数量
- `longest_path_count`: 最长路径点的总数量
- `physical_units`: 是否使用物理单位（微米）
- `um_per_pixel`: 每个像素对应的微米数

## 编程接口

### 提取骨架坐标

```python
from src.skeleton_coordinates import extract_skeleton_coordinates
from src.snake_scan import ScanConfig

# 创建配置
scan_cfg = ScanConfig(
    frame_w=1028, frame_h=776,
    um_per_pixel=1.0, um_per_step=1.0,
    output_dir="output/path"
)

# 提取坐标（像素单位）
coordinates = extract_skeleton_coordinates(mask, scan_cfg, use_physical_units=False)

# 提取坐标（微米单位）
coordinates = extract_skeleton_coordinates(mask, scan_cfg, use_physical_units=True)
```

### 保存坐标

```python
from src.skeleton_coordinates import save_skeleton_coordinates

save_skeleton_coordinates(coordinates, "output/coordinates.json")
```

### 可视化坐标

```python
from src.skeleton_coordinates import visualize_skeleton_coordinates

visualized = visualize_skeleton_coordinates(
    image, coordinates,
    point_radius=2,
    skeleton_color=(0, 0, 255),  # 红色
    path_color=(0, 255, 0)      # 绿色
)
cv2.imwrite("output/visualization.png", visualized)
```

## 示例代码

### 完整示例

```python
import cv2
import numpy as np
from src.snake_scan import ScanConfig
from src.skeleton_coordinates import (
    extract_skeleton_coordinates,
    save_skeleton_coordinates,
    visualize_skeleton_coordinates
)

# 1. 加载语义掩码
mask = cv2.imread("global_mask.png", cv2.IMREAD_GRAYSCALE)

# 2. 创建配置
scan_cfg = ScanConfig(
    frame_w=1028, frame_h=776,
    um_per_pixel=1.0, um_per_step=1.0,
    output_dir="output"
)

# 3. 提取骨架坐标
coordinates = extract_skeleton_coordinates(mask, scan_cfg)

print(f"完整骨架点数量: {coordinates['skeleton_count']}")
print(f"最长路径点数量: {coordinates['longest_path_count']}")

# 4. 保存坐标
save_skeleton_coordinates(coordinates, "skeleton_coordinates.json")

# 5. 可视化
image = cv2.imread("global_map.png")
visualized = visualize_skeleton_coordinates(image, coordinates)
cv2.imwrite("visualization.png", visualized)
```

## 注意事项

1. **坐标系**: 所有坐标都以左上角为原点，X轴向右，Y轴向下
2. **单位转换**: 使用物理单位时，需要正确设置 `um_per_pixel` 参数
3. **内存使用**: 大型地图可能占用较多内存，建议分批处理
4. **精度**: 骨架提取精度取决于语义分割的质量

## 故障排除

### 问题: 未找到骨架

**可能原因**:
- 语义掩码中没有道路区域
- 道路区域太小，不满足最小长度要求

**解决方法**:
- 检查语义分割结果
- 调整 `skeleton_min_length` 参数

### 问题: 坐标不正确

**可能原因**:
- 坐标系理解错误
- 图像尺寸不匹配

**解决方法**:
- 确认使用左上角为原点的坐标系
- 检查图像和掩码的尺寸是否一致

## 技术细节

### 坐标转换

骨架提取使用 `np.argwhere()` 返回 (y, x) 格式的坐标，需要转换为 (x, y) 格式:

```python
# 从 (y, x) 转换为 (x, y)
skeleton_pixels = np.argwhere(full_skeleton > 0)
skeleton_points = [(int(x), int(y)) for y, x in skeleton_pixels]
```

### 骨架提取算法

使用 `skimage.morphology.skeletonize` 进行骨架提取，然后使用图算法找到最长路径。

## 更新日志

### v1.0 (2026-03-17)
- 初始版本
- 实现骨架坐标提取功能
- 支持像素和微米单位
- 添加可视化功能
- 集成到主程序

## 联系方式

如有问题或建议，请联系开发团队。
