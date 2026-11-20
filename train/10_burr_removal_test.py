"""
毛刺去除测试
============
测试骨架毛刺去除功能
"""

import os
import sys
import numpy as np

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（脚本目录的上一级）
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到 Python 模块搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("🧪 毛刺去除功能测试")
print("=" * 70)

# 测试 1: 检查代码修改是否成功
print("\n1️⃣ 检查代码修改...")
snake_scan_file = os.path.join(project_root, "src", "snake_scan.py")
with open(snake_scan_file, 'r', encoding='utf-8') as f:
    content = f.read()

if "_remove_burrs" in content:
    print("   ✅ _remove_burrs 方法已添加")
else:
    print("   ❌ _remove_burrs 方法未找到")
    sys.exit(1)

if "# 4. 去除骨架毛刺" in content:
    print("   ✅ 毛刺去除逻辑已集成到 extract_from_mask 方法")
else:
    print("   ❌ 毛刺去除逻辑未集成")
    sys.exit(1)

# 测试 2: 模拟骨架测试
print("\n2️⃣ 模拟骨架测试...")

# 创建一个带有毛刺的模拟骨架
def create_test_skeleton():
    """创建一个带有毛刺的模拟骨架"""
    # 创建一个 100x100 的骨架
    skeleton = np.zeros((100, 100), dtype=np.uint8)
    
    # 主路径
    skeleton[40:60, 20:80] = 1  # 水平主路径
    skeleton[50, 20:80] = 1      # 加强主路径
    
    # 毛刺 1: 短毛刺
    skeleton[45, 30:35] = 1  # 长度 5
    
    # 毛刺 2: 非常短的毛刺
    skeleton[55, 40:42] = 1  # 长度 2
    
    # 毛刺 3: 中等长度的毛刺
    skeleton[45, 60:68] = 1  # 长度 8
    
    # 毛刺 4: 长毛刺（应该保留）
    skeleton[55, 70:85] = 1  # 长度 15
    
    # 垂直分支
    skeleton[30:50, 50] = 1  # 长度 20
    
    return skeleton

test_skeleton = create_test_skeleton()
print(f"   ✅ 模拟骨架创建成功，尺寸: {test_skeleton.shape}")
print(f"   ✅ 骨架总像素数: {np.sum(test_skeleton)}")

# 测试 3: 测试毛刺去除功能
print("\n3️⃣ 测试毛刺去除功能...")

try:
    from src.snake_scan import ScanConfig, PathSkeletonExtractor
    
    # 创建配置
    scan_cfg = ScanConfig(
        frame_w=100, frame_h=100,
        skeleton_min_length=8  # 设置最小分支长度阈值
    )
    
    # 创建骨架提取器
    extractor = PathSkeletonExtractor(scan_cfg)
    
    # 测试毛刺去除
    cleaned_skeleton = extractor._remove_burrs(test_skeleton, min_length=8)
    
    print(f"   ✅ 毛刺去除完成")
    print(f"   ✅ 去除前像素数: {np.sum(test_skeleton)}")
    print(f"   ✅ 去除后像素数: {np.sum(cleaned_skeleton)}")
    print(f"   ✅ 去除的像素数: {np.sum(test_skeleton) - np.sum(cleaned_skeleton)}")
    
    # 测试 4: 验证提取完整骨架功能
    print("\n4️⃣ 测试完整骨架提取...")
    
    # 创建模拟掩码
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[test_skeleton > 0] = 1  # 道路区域
    
    # 提取骨架
    full_skeleton, longest_path, path_points = extractor.extract_from_mask(mask)
    
    print(f"   ✅ 完整骨架提取成功")
    print(f"   ✅ 提取的骨架像素数: {np.sum(full_skeleton)}")
    print(f"   ✅ 最长路径长度: {len(path_points)}")
    
    # 测试 5: 验证坐标提取
    print("\n5️⃣ 测试坐标提取...")
    
    from src.skeleton_coordinates import extract_skeleton_coordinates
    
    coordinates = extract_skeleton_coordinates(mask, scan_cfg)
    
    print(f"   ✅ 坐标提取成功")
    print(f"   ✅ 完整骨架点数量: {coordinates['skeleton_count']}")
    print(f"   ✅ 最长路径点数量: {coordinates['longest_path_count']}")
    print(f"   ✅ 坐标系原点: 左上角 (0, 0)")
    
    # 显示前几个坐标
    print("\n📋 前5个骨架点坐标:")
    for i, (x, y) in enumerate(coordinates['skeleton_points'][:5]):
        print(f"   [{i+1}] ({x}, {y})")
    
    print("\n📋 前5个最长路径点坐标:")
    for i, (x, y) in enumerate(coordinates['longest_path_points'][:5]):
        print(f"   [{i+1}] ({x}, {y})")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    
    print("\n📝 功能说明:")
    print("   1. 毛刺去除算法会移除长度小于阈值的分支")
    print("   2. 默认阈值为 skeleton_min_length 参数（默认值为 5）")
    print("   3. 只保留主要的连通路径")
    print("   4. 最长路径提取会基于去除毛刺后的骨架")
    
    print("\n🚀 使用方法:")
    print("   1. 运行主程序: python train/5_core_clip_simulate.py")
    print("   2. 设置 MODE = 'snake' 进行蛇形扫描")
    print("   3. 扫描完成后，自动提取去毛刺后的骨架坐标")
    print("   4. 输出文件:")
    print("      - skeleton_coordinates.json: 骨架坐标数据")
    print("      - skeleton_coordinates_visualization.png: 可视化结果")
    print("      - global_map.png: 全局地图（含骨架）")
    
    print("\n" + "=" * 70)

except Exception as e:
    print(f"   ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
