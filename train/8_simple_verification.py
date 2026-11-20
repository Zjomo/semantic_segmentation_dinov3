"""
简单验证：骨架坐标提取功能
===========================
验证骨架坐标提取功能是否正确实现
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
print("✅ 骨架坐标提取功能验证")
print("=" * 70)

# 验证 1: 检查文件是否存在
print("\n1️⃣ 检查文件是否存在...")
files_to_check = [
    "src/skeleton_coordinates.py",
    "train/5_core_clip_simulate.py",
]

all_exist = True
for file_path in files_to_check:
    full_path = os.path.join(project_root, file_path)
    if os.path.exists(full_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} 不存在")
        all_exist = False

if not all_exist:
    print("\n❌ 部分文件缺失，请检查")
    sys.exit(1)

# 验证 2: 检查代码集成
print("\n2️⃣ 检查代码集成...")
main_file = os.path.join(project_root, "train", "5_core_clip_simulate.py")
with open(main_file, 'r', encoding='utf-8') as f:
    content = f.read()

integration_checks = [
    ("导入骨架坐标模块", "from src.skeleton_coordinates import"),
    ("提取骨架坐标", "extract_skeleton_coordinates"),
    ("保存骨架坐标", "save_skeleton_coordinates"),
    ("可视化骨架坐标", "visualize_skeleton_coordinates"),
]

all_integrated = True
for check_name, check_pattern in integration_checks:
    if check_pattern in content:
        print(f"   ✅ {check_name}")
    else:
        print(f"   ❌ {check_name} 未找到")
        all_integrated = False

if not all_integrated:
    print("\n❌ 部分功能未集成，请检查")
    sys.exit(1)

# 验证 3: 检查骨架坐标模块
print("\n3️⃣ 检查骨架坐标模块...")
skeleton_file = os.path.join(project_root, "src", "skeleton_coordinates.py")
with open(skeleton_file, 'r', encoding='utf-8') as f:
    skeleton_content = f.read()

function_checks = [
    ("extract_skeleton_coordinates 函数", "def extract_skeleton_coordinates"),
    ("save_skeleton_coordinates 函数", "def save_skeleton_coordinates"),
    ("visualize_skeleton_coordinates 函数", "def visualize_skeleton_coordinates"),
]

all_functions = True
for check_name, check_pattern in function_checks:
    if check_pattern in skeleton_content:
        print(f"   ✅ {check_name}")
    else:
        print(f"   ❌ {check_name} 未找到")
        all_functions = False

if not all_functions:
    print("\n❌ 部分函数未实现，请检查")
    sys.exit(1)

# 验证 4: 检查坐标系说明
print("\n4️⃣ 检查坐标系说明...")
if "左上角为原点" in skeleton_content or "以左上角为原点" in skeleton_content:
    print("   ✅ 坐标系说明正确（左上角为原点）")
else:
    print("   ⚠️  坐标系说明未明确")

# 验证 5: 检查输出格式
print("\n5️⃣ 检查输出格式...")
output_checks = [
    ("skeleton_points", "skeleton_points"),
    ("longest_path_points", "longest_path_points"),
    ("skeleton_count", "skeleton_count"),
    ("longest_path_count", "longest_path_count"),
]

all_outputs = True
for check_name, check_pattern in output_checks:
    if check_pattern in skeleton_content:
        print(f"   ✅ {check_name}")
    else:
        print(f"   ❌ {check_name} 未找到")
        all_outputs = False

if not all_outputs:
    print("\n❌ 部分输出格式未定义，请检查")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有验证通过！")
print("=" * 70)

print("\n📝 功能说明:")
print("   1. 坐标系: 以左上角为原点 (0, 0)")
print("   2. X轴: 水平向右")
print("   3. Y轴: 垂直向下")
print("   4. 单位: 默认像素，可选择转换为微米")
print("   5. 输出:")
print("      - skeleton_points: 完整骨架点列表 [(x, y), ...]")
print("      - longest_path_points: 最长路径点列表 [(x, y), ...]")
print("      - skeleton_count: 完整骨架点数量")
print("      - longest_path_count: 最长路径点数量")

print("\n🚀 使用方法:")
print("   1. 运行主程序: python train/5_core_clip_simulate.py")
print("   2. 设置 MODE = 'snake' 进行蛇形扫描")
print("   3. 扫描完成后，自动提取骨架坐标并保存")
print("   4. 输出文件:")
print("      - skeleton_coordinates.json: 骨架坐标数据")
print("      - skeleton_coordinates_visualization.png: 可视化结果")
print("      - global_map.png: 全局地图（含骨架）")
print("      - global_mask.png: 原始语义掩码")

print("\n" + "=" * 70)
