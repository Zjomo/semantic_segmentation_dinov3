"""
测试全路径覆盖切割路径功能
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import cv2
from src.snake_scan import PathSkeletonExtractor, ScanConfig

def create_test_mask():
    """创建一个复杂的测试掩码，包含多条路径和分支"""
    h, w = 500, 500
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 绘制主路径
    for i in range(150):
        x = 50 + i * 2
        y = 50 + i
        if x < w and y < h:
            cv2.circle(mask, (x, y), 3, 1, -1)
    
    # 添加分支1
    for i in range(50):
        x = 200 + i
        y = 150 + i * 2
        if x < w and y < h:
            cv2.circle(mask, (x, y), 3, 1, -1)
    
    # 添加分支2
    for i in range(70):
        x = 250 + i * 1.5
        y = 200 - i
        if x < w and y < h:
            cv2.circle(mask, (x, y), 3, 1, -1)
    
    # 添加环路
    for i in range(100):
        angle = 2 * np.pi * i / 100
        x = int(350 + 50 * np.cos(angle))
        y = int(250 + 50 * np.sin(angle))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(mask, (x, y), 2, 1, -1)
    
    return mask

def test_full_coverage_cutting():
    """测试全路径覆盖切割路径功能"""
    print("=" * 70)
    print("🧪 测试全路径覆盖切割路径功能")
    print("=" * 70)
    
    # 创建测试配置
    config = ScanConfig(
        frame_w=500, frame_h=500,
        um_per_pixel=1.0,
        um_per_step=1.0,
        skeleton_min_length=5,
        output_dir="test_output"
    )
    
    # 创建提取器
    extractor = PathSkeletonExtractor(config)
    
    # 创建测试掩码
    mask = create_test_mask()
    print(f"✅ 测试掩码创建完成: {mask.shape}")
    print(f"   - 道路像素数: {np.sum(mask > 0)}")
    
    # 提取原始骨架（用于对比）
    road_mask = mask.copy()
    from skimage.morphology import skeletonize
    original_skel = skeletonize(road_mask > 0).astype(np.uint8)
    print(f"✅ 原始骨架提取完成: {np.sum(original_skel > 0)} 像素")
    
    # 提取全路径覆盖切割路径
    print("\n🔍 提取全路径覆盖切割路径...")
    cutting_mask, cutting_points, cutting_directions = extractor.extract_cutting_path(mask, cover_all_paths=True)
    
    print(f"✅ 全路径覆盖切割路径提取完成:")
    print(f"   - 切割路径点数量: {len(cutting_points)}")
    print(f"   - 方向标注数量: {len(cutting_directions)}")
    print(f"   - 覆盖的骨架像素数: {np.sum(cutting_mask > 0)}")
    print(f"   - 原始骨架像素数: {np.sum(original_skel > 0)}")
    print(f"   - 覆盖率: {np.sum(cutting_mask > 0) / np.sum(original_skel > 0) * 100:.2f}%")
    
    if cutting_points:
        print(f"   - 起始点: {cutting_points[0]}")
        print(f"   - 终止点: {cutting_points[-1]}")
    
    # 绘制结果
    print("\n🎨 绘制结果...")
    
    # 绘制原始骨架
    original_skel_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for y, x in np.argwhere(original_skel > 0):
        original_skel_img[y, x] = [100, 100, 100]
    
    # 绘制切割路径
    result = extractor.draw_cutting_path_with_arrows(
        original_skel_img, cutting_mask, cutting_directions,
        path_color=(0, 255, 255), arrow_color=(0, 0, 255),
        arrow_spacing=15, arrow_size=15
    )
    
    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    output_path = os.path.join(config.output_dir, "full_coverage_test.png")
    cv2.imwrite(output_path, result)
    print(f"✅ 结果已保存到: {output_path}")
    
    # 显示方向统计
    print("\n📊 方向统计:")
    direction_names = ['右', '下', '左', '上', '右下', '左下', '右上', '左上']
    dir_counts = {}
    for _, _, dir_idx in cutting_directions:
        dir_counts[dir_idx] = dir_counts.get(dir_idx, 0) + 1
    
    for dir_idx, count in sorted(dir_counts.items()):
        dir_name = direction_names[dir_idx] if dir_idx < len(direction_names) else f'方向{dir_idx}'
        print(f"   {dir_name}: {count} 次")
    
    print("\n" + "=" * 70)
    print("✅ 全路径覆盖测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_full_coverage_cutting()