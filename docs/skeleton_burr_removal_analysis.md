# 骨架毛刺去除参数分析

## 概述

本文档详细分析了代码中去除"骨架毛刺"的相关参数及其作用机制。

---

## 1. 核心参数：`skeleton_min_length`

### 定义位置
- **文件**: `src/snake_scan.py`
- **类**: `ScanConfig`
- **默认值**: `5`

```python
@dataclass
class ScanConfig:
    ...
    skeleton_min_length: int = 5  # 最小分支长度阈值
    ...
```

### 作用
`skeleton_min_length` 是控制毛刺去除效果的核心参数，它定义了**最小分支长度阈值**。

### 工作原理

#### 1. 毛刺识别
- 算法会找到所有**骨架端点**（度为1的节点）
- 从每个端点开始**回溯路径**
- 计算路径长度

#### 2. 毛刺移除
- 如果路径长度 < `skeleton_min_length`，则该分支被判定为毛刺
- 所有毛刺像素点会被移除

#### 3. 保留主路径
- 只保留长度 ≥ `skeleton_min_length` 的分支
- 确保骨架只包含主要连通路径

### 参数调整建议

| 参数值 | 效果 | 适用场景 |
|--------|------|----------|
| **1-5** | 保留大部分分支，仅移除极短毛刺 | 细节丰富的骨架，需要保留更多分支 |
| **5-20** | 平衡模式，移除明显毛刺 | 一般场景，默认推荐 |
| **20-50** | 较激进，移除更多分支 | 简化骨架，只保留主干道 |
| **50-100+** | 非常激进，只保留最长路径 | 极简骨架，单一主路径 |

### 实际配置示例

#### 示例 1: 默认配置
```python
scan_cfg = ScanConfig(
    skeleton_min_length=5,  # 默认值
)
```

#### 示例 2: 激进配置（简化骨架）
```python
scan_cfg = ScanConfig(
    skeleton_min_length=350,  # 大值，只保留长分支
)
```

---

## 2. 辅助参数：形态学预处理

### 定义位置
- **文件**: `src/snake_scan.py`
- **方法**: `extract_from_mask`
- **代码行**: 约 220-225

```python
# ★ 形态学预处理：闭运算填充小空洞，再开运算去除孤立噪点
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  kernel_open)
```

### 参数说明

#### 闭运算（`MORPH_CLOSE`）
- **核大小**: `(9, 9)`
- **作用**: 填充小空洞，连接邻近区域
- **效果**: 减少骨架断裂

#### 开运算（`MORPH_OPEN`）
- **核大小**: `(3, 3)`
- **作用**: 去除孤立噪点
- **效果**: 减少骨架毛刺源头

### 调整建议

| 参数 | 当前值 | 调整方向 | 效果 |
|------|--------|----------|------|
| 闭运算核 | `(9, 9)` | 增大 | 填充更大空洞，但可能过度连接 |
| 闭运算核 | `(9, 9)` | 减小 | 保留更多细节，但可能有断裂 |
| 开运算核 | `(3, 3)` | 增大 | 去除更多噪点，但可能丢失细节 |
| 开运算核 | `(3, 3)` | 减小 | 保留更多细节，但可能有噪点 |

---

## 3. 辅助参数：最大连通域保留

### 定义位置
- **文件**: `src/snake_scan.py`
- **方法**: `extract_from_mask`
- **代码行**: 约 227-231

```python
# ★ 保留最大连通域（去除孤立短骨架片段）
n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
if n_labels > 2:  # 0=背景，1+为各连通域
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    skel = (labels == largest_label).astype(np.uint8)
```

### 参数说明

#### 连通性参数
- **当前值**: `connectivity=8`
- **可选值**: `4` 或 `8`
- **作用**: 定义像素连接方式

| 连通性 | 说明 | 效果 |
|--------|------|------|
| **4** | 上下左右4邻域 | 更严格的连接，可能产生更多孤立区域 |
| **8** | 包含对角线8邻域 | 更宽松的连接，更容易形成连通区域 |

### 效果
- 只保留面积最大的连通域
- 自动移除所有孤立的短骨架片段
- 减少后续毛刺处理的复杂度

---

## 4. 高级功能：干净主路径提取

### 定义位置
- **文件**: `src/snake_scan.py`
- **方法**: `extract_clean_main_path`

### 功能说明
除了基本的毛刺去除，代码还提供了**干净主路径提取**功能，可以：
- 彻底去除所有环路
- 彻底去除所有分支
- 只保留一条无分叉的像素链

### 算法原理

#### 1. 链段分解
- 将骨架分解为**链段**（两关节节点之间的连续像素序列）
- 关节节点：端点（度=1）或分叉点（度≥3）

#### 2. DFS 搜索
- 在链段图中用 DFS 枚举所有简单路径
- 每条链段最多用一次 → 消除环路
- 每个关节节点最多经过一次 → 消除分支

#### 3. 性能优化
- 节点数 > 300 时自动切换为**贪心策略**
- 避免大图计算时间过长

### 参数说明

```python
_MAX_DFS_NODES = 300  # DFS 节点数上限
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `_MAX_DFS_NODES` | 300 | DFS 搜索的节点数上限 |

---

## 5. 参数调整策略

### 策略 1: 保守模式（保留细节）
```python
scan_cfg = ScanConfig(
    skeleton_min_length=5,  # 小值，保留更多分支
)
```
**适用场景**:
- 需要保留骨架的所有细节
- 骨架结构复杂，有多个重要分支
- 对精度要求高

### 策略 2: 平衡模式（默认推荐）
```python
scan_cfg = ScanConfig(
    skeleton_min_length=20,  # 中等值，平衡细节和简洁
)
```
**适用场景**:
- 一般应用场景
- 需要去除明显毛刺，但保留主要分支
- 默认推荐配置

### 策略 3: 激进模式（极简骨架）
```python
scan_cfg = ScanConfig(
    skeleton_min_length=350,  # 大值，只保留主干道
)
```
**适用场景**:
- 只需要主干道骨架
- 骨架结构简单，单一主路径
- 对简洁性要求高

---

## 6. 参数调优流程

### 步骤 1: 初始评估
1. 使用默认参数运行程序
2. 查看输出的骨架可视化结果
3. 评估毛刺情况

### 步骤 2: 参数调整
1. 如果毛刺较多 → **增大** `skeleton_min_length`
2. 如果细节丢失 → **减小** `skeleton_min_length`
3. 如果有断裂 → **增大** 闭运算核
4. 如果有噪点 → **增大** 开运算核

### 步骤 3: 验证结果
1. 检查骨架是否干净
2. 检查是否保留了必要的分支
3. 检查是否有断裂或丢失

### 步骤 4: 迭代优化
1. 根据结果微调参数
2. 重复步骤 2-3 直到满意

---

## 7. 常见问题与解决方案

### 问题 1: 骨架断裂
**原因**: `skeleton_min_length` 过大或形态学核过小

**解决方案**:
```python
scan_cfg = ScanConfig(
    skeleton_min_length=10,  # 减小阈值
)
# 或在代码中增大闭运算核
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
```

### 问题 2: 毛刺仍然存在
**原因**: `skeleton_min_length` 过小

**解决方案**:
```python
scan_cfg = ScanConfig(
    skeleton_min_length=50,  # 增大阈值
)
```

### 问题 3: 重要分支被移除
**原因**: `skeleton_min_length` 过大

**解决方案**:
```python
scan_cfg = ScanConfig(
    skeleton_min_length=15,  # 适中阈值
)
```

### 问题 4: 骨架过于复杂
**原因**: 需要使用干净主路径提取

**解决方案**:
```python
# 使用 extract_clean_main_path 方法
clean_mask, path_points = extractor.extract_clean_main_path(mask)
```

---

## 8. 参数影响对比表

| 参数 | 值范围 | 影响 | 推荐值 |
|------|--------|------|--------|
| `skeleton_min_length` | 1-500+ | 毛刺去除强度 | 5-50 |
| 闭运算核 | (3,3)-(15,15) | 空洞填充程度 | (9,9) |
| 开运算核 | (1,1)-(7,7) | 噪点去除程度 | (3,3) |
| 连通性 | 4 或 8 | 连接严格程度 | 8 |
| `_MAX_DFS_NODES` | 100-1000 | DFS 性能限制 | 300 |

---

## 9. 总结

### 核心参数
- **`skeleton_min_length`**: 最重要，直接控制毛刺去除强度

### 辅助参数
- **形态学核**: 预处理阶段，影响骨架质量
- **连通性**: 影响连通域判断
- **DFS 节点限制**: 性能优化

### 调优建议
1. **从小值开始**: 先用小值测试，逐步增大
2. **观察可视化**: 每次调整后查看可视化结果
3. **平衡细节与简洁**: 根据实际需求找到平衡点
4. **使用高级功能**: 需要极简骨架时使用干净主路径提取

---

## 10. 代码位置参考

| 功能 | 文件 | 类/方法 | 行号 |
|------|------|---------|------|
| 参数定义 | `src/snake_scan.py` | `ScanConfig` | 104-123 |
| 毛刺去除 | `src/snake_scan.py` | `_remove_burrs` | 293-357 |
| 形态学预处理 | `src/snake_scan.py` | `extract_from_mask` | 220-225 |
| 最大连通域 | `src/snake_scan.py` | `extract_from_mask` | 227-231 |
| 干净主路径 | `src/snake_scan.py` | `extract_clean_main_path` | 370-410 |
| 实际配置 | `train/5_core_clip_simulate.py` | `main` | 516-529 |

---

**文档版本**: v1.0  
**更新日期**: 2026-03-17  
**适用代码版本**: snake_scan.py v6.5
