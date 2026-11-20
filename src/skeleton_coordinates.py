"""
骨架坐标提取器
================
以左上角为原点构建坐标系，提取全局地图中的骨架坐标
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from src.snake_scan import PathSkeletonExtractor, ScanConfig


def extract_skeleton_coordinates(
    mask: np.ndarray, 
    config: ScanConfig,
    use_physical_units: bool = False
) -> Dict[str, Any]:
    """
    从语义分割掩码提取骨架坐标，以左上角为原点构建坐标系
    
    Args:
        mask: 语义分割掩码，SegClass.ROAD = 1 表示道路
        config: 扫描配置，包含像素到物理单位的转换参数
        use_physical_units: 是否返回物理单位坐标（微米）
        
    Returns:
        Dict 包含以下键:
            - 'skeleton_points': 完整骨架点列表 [(x, y), ...]，以左上角为原点
            - 'longest_path_points': 最长路径点列表 [(x, y), ...]，以左上角为原点
            - 'skeleton_count': 完整骨架点数量
            - 'longest_path_count': 最长路径点数量
            - 'physical_units': 是否使用物理单位
            - 'um_per_pixel': 每个像素的微米数
    """
    # 创建骨架提取器
    extractor = PathSkeletonExtractor(config)
    
    # 提取骨架
    full_skeleton, longest_path_mask, longest_path_points = extractor.extract_from_mask(mask)
    
    # 获取完整骨架的所有点坐标
    skeleton_pixels = np.argwhere(full_skeleton > 0)
    
    # 转换坐标系统：从 (y, x) 转换为 (x, y)，以左上角为原点
    # 注意：np.argwhere 返回的是 (行, 列)，对应 (y, x)
    skeleton_points = [(int(x), int(y)) for y, x in skeleton_pixels]
    
    # 转换最长路径的坐标系统
    longest_path_points_converted = [(int(x), int(y)) for y, x in longest_path_points]
    
    # 如果需要使用物理单位，进行转换
    if use_physical_units:
        scale = config.um_per_pixel
        skeleton_points = [(x * scale, y * scale) for x, y in skeleton_points]
        longest_path_points_converted = [(x * scale, y * scale) for x, y in longest_path_points_converted]
    
    return {
        'skeleton_points': skeleton_points,
        'longest_path_points': longest_path_points_converted,
        'skeleton_count': len(skeleton_points),
        'longest_path_count': len(longest_path_points_converted),
        'physical_units': use_physical_units,
        'um_per_pixel': config.um_per_pixel
    }


def save_skeleton_coordinates(
    coordinates: Dict[str, Any], 
    output_path: str
) -> None:
    """
    保存骨架坐标到 JSON 文件
    
    Args:
        coordinates: 由 extract_skeleton_coordinates 返回的坐标字典
        output_path: 输出 JSON 文件路径
    """
    import json
    
    # 确保路径存在
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存为 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coordinates, f, indent=2, ensure_ascii=False)
    
    print(f"骨架坐标已保存到: {output_path}")


def visualize_skeleton_coordinates(
    image: np.ndarray, 
    coordinates: Dict[str, Any],
    point_radius: int = 2,
    skeleton_color: Tuple[int, int, int] = (0, 0, 255),  # 红色
    path_color: Tuple[int, int, int] = (0, 255, 0)      # 绿色
) -> np.ndarray:
    """
    在图像上可视化骨架坐标
    
    Args:
        image: 输入图像
        coordinates: 由 extract_skeleton_coordinates 返回的坐标字典
        point_radius: 点的半径
        skeleton_color: 完整骨架点的颜色 (B, G, R)
        path_color: 最长路径点的颜色 (B, G, R)
        
    Returns:
        带有骨架坐标标记的图像
    """
    import cv2
    
    output = image.copy()
    
    # 绘制完整骨架点
    for (x, y) in coordinates['skeleton_points']:
        cv2.circle(output, (int(x), int(y)), point_radius, skeleton_color, -1)
    
    # 绘制最长路径点
    for (x, y) in coordinates['longest_path_points']:
        cv2.circle(output, (int(x), int(y)), point_radius + 1, path_color, -1)
    
    # 添加统计信息
    height, width = image.shape[:2]
    text = f"Skeleton: {coordinates['skeleton_count']} points | Longest Path: {coordinates['longest_path_count']} points"
    if coordinates['physical_units']:
        text += f" | Scale: {coordinates['um_per_pixel']} um/px"
    
    cv2.putText(output, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output, text, (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return output
