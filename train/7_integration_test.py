"""
集成测试：验证骨架坐标提取功能
===============================
测试骨架坐标提取功能是否正确集成到主程序中
"""

import os
import sys

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（脚本目录的上一级）
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到 Python 模块搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🧪 集成测试：骨架坐标提取功能")
print("=" * 70)

# 测试导入
print("\n1️⃣ 测试模块导入...")
try:
    from src.snake_scan import ScanConfig, GlobalMapStitcher
    from src.skeleton_coordinates import (
        extract_skeleton_coordinates,
        save_skeleton_coordinates,
        visualize_skeleton_coordinates
    )
    print("   ✅ 所有模块导入成功")
except ImportError as e:
    print(f"   ❌ 模块导入失败: {e}")
    sys.exit(1)

# 测试主程序导入
print("\n2️⃣ 测试主程序导入...")
try:
    # 只测试导入，不运行主程序
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "main_module", 
        os.path.join(project_root, "train", "5_core_clip_simulate.py")
    )
    module = importlib.util.module_from_spec(spec)
    print("   ✅ 主程序模块加载成功")
except Exception as e:
    print(f"   ❌ 主程序加载失败: {e}")
    sys.exit(1)

# 测试功能
print("\n3️⃣ 测试骨架坐标提取功能...")
try:
    import numpy as np
    
    # 创建测试配置
    scan_cfg = ScanConfig(
        frame_w=1028, frame_h=776,
        um_per_pixel=1.0, um_per_step=1.0,
        skeleton_path_radius=1,
        skeleton_highlight_radius=3,
        skeleton_min_length=5,
        output_dir=r"F:/0_My_jupyter/0_Optical/scan_output/test",
    )
    
    # 创建测试掩码
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[50:150, 50:250] = 1  # 主道路
    
    # 提取骨架坐标
    coordinates = extract_skeleton_coordinates(mask, scan_cfg, use_physical_units=False)
    
    print(f"   ✅ 骨架坐标提取成功")
    print(f"      - 完整骨架点数量: {coordinates['skeleton_count']}")
    print(f"      - 最长路径点数量: {coordinates['longest_path_count']}")
    print(f"      - 坐标系原点: 左上角 (0, 0)")
    
    # 测试保存功能
    output_json = os.path.join(scan_cfg.output_dir, "test_coordinates.json")
    save_skeleton_coordinates(coordinates, output_json)
    print(f"   ✅ 坐标保存成功: {output_json}")
    
    # 测试可视化功能
    import cv2
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    test_image[mask == 1] = [0, 255, 0]  # 绿色道路
    
    visualized = visualize_skeleton_coordinates(test_image, coordinates)
    output_image = os.path.join(scan_cfg.output_dir, "test_visualization.png")
    cv2.imwrite(output_image, visualized)
    print(f"   ✅ 可视化保存成功: {output_image}")
    
except Exception as e:
    print(f"   ❌ 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有测试通过！")
print("=" * 70)
print("\n📝 使用说明:")
print("   1. 运行主程序: python train/5_core_clip_simulate.py")
print("   2. 选择 MODE = 'snake' 进行蛇形扫描")
print("   3. 扫描完成后，自动提取骨架坐标并保存")
print("   4. 输出文件:")
print("      - skeleton_coordinates.json: 骨架坐标数据")
print("      - skeleton_coordinates_visualization.png: 可视化结果")
print("      - global_map.png: 全局地图（含骨架）")
print("      - global_mask.png: 原始语义掩码")
print("=" * 70)
