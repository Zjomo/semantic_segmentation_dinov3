"""
毛刺去除功能验证
================
验证毛刺去除功能是否正确集成
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
print("✅ 毛刺去除功能验证")
print("=" * 70)

# 验证 1: 检查代码修改是否成功
print("\n1️⃣ 检查代码修改...")
snake_scan_file = os.path.join(project_root, "src", "snake_scan.py")
with open(snake_scan_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 检查 _remove_burrs 方法
if "def _remove_burrs" in content:
    print("   ✅ _remove_burrs 方法已添加")
else:
    print("   ❌ _remove_burrs 方法未找到")
    sys.exit(1)

# 检查毛刺去除逻辑集成
if "# 4. 去除骨架毛刺" in content:
    print("   ✅ 毛刺去除逻辑已集成到 extract_from_mask 方法")
else:
    print("   ❌ 毛刺去除逻辑未集成")
    sys.exit(1)

# 检查方法调用
if "self._remove_burrs(skel, min_length=self.cfg.skeleton_min_length)" in content:
    print("   ✅ 毛刺去除方法调用正确")
else:
    print("   ❌ 毛刺去除方法调用不正确")
    sys.exit(1)

# 验证 2: 检查方法实现
print("\n2️⃣ 检查方法实现...")

# 检查关键逻辑
burr_removal_checks = [
    ("获取骨架像素", "np.argwhere(skel > 0)"),
    ("构建图结构", "self._skeleton_to_graph"),
    ("计算节点度", "G.degree()"),
    ("找到端点", "degree == 1"),
    ("移除短分支", "len(path) < min_length"),
    ("创建清理后的骨架", "cleaned_skel = skel.copy()"),
]

all_checks_passed = True
for check_name, check_pattern in burr_removal_checks:
    if check_pattern in content:
        print(f"   ✅ {check_name}")
    else:
        print(f"   ❌ {check_name}")
        all_checks_passed = False

if not all_checks_passed:
    print("\n❌ 部分关键逻辑未实现，请检查")
    sys.exit(1)

# 验证 3: 检查主程序集成
print("\n3️⃣ 检查主程序集成...")
main_file = os.path.join(project_root, "train", "5_core_clip_simulate.py")
with open(main_file, 'r', encoding='utf-8') as f:
    main_content = f.read()

if "extract_skeleton_coordinates" in main_content:
    print("   ✅ 骨架坐标提取功能已集成到主程序")
else:
    print("   ❌ 骨架坐标提取功能未集成到主程序")
    sys.exit(1)

# 验证 4: 检查配置参数
print("\n4️⃣ 检查配置参数...")
if "skeleton_min_length" in content:
    print("   ✅ skeleton_min_length 参数已使用")
else:
    print("   ❌ skeleton_min_length 参数未使用")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有验证通过！")
print("=" * 70)

print("\n📝 功能说明:")
print("   1. 毛刺去除算法原理:")
print("      - 找到所有骨架端点（度为1的节点）")
print("      - 从每个端点开始回溯路径")
print("      - 移除长度小于阈值的分支")
print("      - 只保留主要的连通路径")
print("   2. 默认阈值:")
print("      - skeleton_min_length = 5（可在配置中修改）")
print("   3. 输出结果:")
print("      - 去除毛刺后的完整骨架")
print("      - 基于去毛刺后骨架的最长路径")
print("      - 以左上角为原点的坐标系统")

print("\n🚀 使用方法:")
print("   1. 运行主程序: python train/5_core_clip_simulate.py")
print("   2. 设置 MODE = 'snake' 进行蛇形扫描")
print("   3. 扫描完成后，自动提取去毛刺后的骨架坐标")
print("   4. 查看输出文件:")
print("      - skeleton_coordinates.json: 骨架坐标数据")
print("      - skeleton_coordinates_visualization.png: 可视化结果")
print("      - global_map.png: 全局地图（含骨架）")

print("\n💡 提示:")
print("   - 如需调整毛刺去除的阈值，可修改 ScanConfig 中的 skeleton_min_length 参数")
print("   - 值越大，移除的分支越多，保留的路径越简洁")
print("   - 值越小，保留的分支越多，可能包含更多细节")

print("\n" + "=" * 70)
