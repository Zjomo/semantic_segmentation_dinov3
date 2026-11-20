"""
骨架毛刺去除参数测试工具
========================
快速测试不同参数对骨架毛刺去除效果的影响
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
print("🔧 骨架毛刺去除参数测试工具")
print("=" * 70)

# 测试配置
test_configs = [
    {
        "name": "保守模式（保留细节）",
        "skeleton_min_length": 5,
        "description": "保留大部分分支，仅移除极短毛刺",
        "use_case": "细节丰富的骨架，需要保留更多分支"
    },
    {
        "name": "平衡模式（默认推荐）",
        "skeleton_min_length": 20,
        "description": "平衡模式，移除明显毛刺",
        "use_case": "一般场景，默认推荐"
    },
    {
        "name": "激进模式（简化骨架）",
        "skeleton_min_length": 50,
        "description": "较激进，移除更多分支",
        "use_case": "简化骨架，只保留主干道"
    },
    {
        "name": "极简模式（单一主路径）",
        "skeleton_min_length": 100,
        "description": "非常激进，只保留最长路径",
        "use_case": "极简骨架，单一主路径"
    },
]

print("\n📋 可用的测试配置:")
print("-" * 70)
for i, config in enumerate(test_configs, 1):
    print(f"\n{i}. {config['name']}")
    print(f"   参数值: skeleton_min_length = {config['skeleton_min_length']}")
    print(f"   说明: {config['description']}")
    print(f"   适用场景: {config['use_case']}")

print("\n" + "=" * 70)
print("💡 使用建议")
print("=" * 70)

print("\n1️⃣ 如何选择参数:")
print("   - 从小值开始测试，逐步增大")
print("   - 观察可视化结果，找到最佳平衡点")
print("   - 根据实际需求调整")

print("\n2️⃣ 参数调整策略:")
print("   - 毛刺较多 → 增大 skeleton_min_length")
print("   - 细节丢失 → 减小 skeleton_min_length")
print("   - 骨架断裂 → 减小 skeleton_min_length 或调整形态学核")

print("\n3️⃣ 高级功能:")
print("   - 使用 extract_clean_main_path() 获取无环无分支的主路径")
print("   - 适用于需要极简骨架的场景")

print("\n" + "=" * 70)
print("📝 代码示例")
print("=" * 70)

print("""
# 示例 1: 使用默认参数
scan_cfg = ScanConfig(
    skeleton_min_length=5,  # 默认值
)

# 示例 2: 使用激进参数
scan_cfg = ScanConfig(
    skeleton_min_length=350,  # 大值，只保留长分支
)

# 示例 3: 使用干净主路径提取
from src.snake_scan import PathSkeletonExtractor

extractor = PathSkeletonExtractor(scan_cfg)
clean_mask, path_points = extractor.extract_clean_main_path(mask)
""")

print("\n" + "=" * 70)
print("🔍 参数影响对比")
print("=" * 70)

print("\n| 参数值 | 效果 | 适用场景 |")
print("|--------|------|----------|")
print("| 1-5    | 保留大部分分支，仅移除极短毛刺 | 细节丰富的骨架 |")
print("| 5-20   | 平衡模式，移除明显毛刺 | 一般场景（推荐）|")
print("| 20-50  | 较激进，移除更多分支 | 简化骨架 |")
print("| 50-100+| 非常激进，只保留最长路径 | 极简骨架 |")

print("\n" + "=" * 70)
print("⚠️  注意事项")
print("=" * 70)

print("\n1. 参数值过大会导致:")
print("   - 重要分支被移除")
print("   - 骨架断裂")
print("   - 细节丢失")

print("\n2. 参数值过小会导致:")
print("   - 毛刺仍然存在")
print("   - 骨架过于复杂")
print("   - 后续处理困难")

print("\n3. 建议调试流程:")
print("   ① 使用默认参数运行")
print("   ② 查看可视化结果")
print("   ③ 根据效果调整参数")
print("   ④ 重复步骤 ②-③ 直到满意")

print("\n" + "=" * 70)
print("📚 相关文档")
print("=" * 70)

print("\n详细参数分析文档:")
print(f"  {os.path.join(project_root, 'docs', 'skeleton_burr_removal_analysis.md')}")

print("\n" + "=" * 70)
print("✅ 参数测试工具准备就绪")
print("=" * 70)

print("\n💡 提示:")
print("   - 运行主程序: python train/5_core_clip_simulate.py")
print("   - 设置 MODE = 'snake' 进行蛇形扫描")
print("   - 查看 output_dir 中的可视化结果")
print("   - 根据效果调整 skeleton_min_length 参数")

print("\n" + "=" * 70)
