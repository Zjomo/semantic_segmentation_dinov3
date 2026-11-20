"""
蛇形扫描全局地图构建系统 (Snake Scan Global Map Builder) v6.5
==========================================================
更新: 新增干净主路径提取（去除环路和分支）
      - 灰色细线：完整骨架网络
      - 红色粗线：最长路径（原样保留，含环/分支）
      - 黄色粗线：干净主路径（链段图DFS，彻底去除环路和分支）
"""

import os
os.chdir("..")

import torch
from torch import nn
import numpy as np
import cv2
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional, Tuple, List, Dict, Set

import networkx as nx
from skimage.morphology import skeletonize

from src.model_head import ASPPDecoder
from src.model_backbone import DinoBackbone
from src.common import image_to_tensor
from src.utils import outputs_to_maps, visualize_maps
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config.config as cfg

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("SnakeScan")

IMG_SIZE             = cfg.IMG_SIZE
PATCH_SIZE           = cfg.PATCH_SIZE
IMG_MEAN             = np.array(cfg.IMG_MEAN, dtype=np.float32)[:, None, None]
IMG_STD              = np.array(cfg.IMG_STD, dtype=np.float32)[:, None, None]
HIDDEN_DIM           = cfg.HIDDEN_DIM
TARGET_SIZE          = cfg.TARGET_SIZE
DINOV3_DIR           = cfg.DINOV3_DIR
DINO_MODEL           = cfg.DINO_MODEL
DINO_WEIGHTS         = cfg.DINO_WEIGHTS
MODEL_TO_NUM_LAYERS  = cfg.MODEL_TO_NUM_LAYERS
MODEL_TO_EMBED_DIM   = cfg.MODEL_TO_EMBED_DIM
MODEL_PATH_INFERENCE = cfg.MODEL_PATH_INFERENCE
CLASS_NAMES_PATH     = cfg.CLASS_NAMES_PATH


class SegClass(IntEnum):
    BACKGROUND = 0
    ROAD       = 1
    OBSTACLE   = 2
    BOUNDARY   = 3

class PhysicalCorner(IntEnum):
    TOP_LEFT     = auto()
    TOP_RIGHT    = auto()
    BOTTOM_LEFT  = auto()
    BOTTOM_RIGHT = auto()

class Direction(IntEnum):
    TOP    = auto()
    BOTTOM = auto()
    LEFT   = auto()
    RIGHT  = auto()

class ViewCorner(IntEnum):
    TOP_LEFT     = auto()
    TOP_RIGHT    = auto()
    BOTTOM_LEFT  = auto()
    BOTTOM_RIGHT = auto()

SEG_PALETTE = {
    SegClass.BACKGROUND: (0,   0,   0),
    SegClass.ROAD:       (0,   255, 0),
    SegClass.OBSTACLE:   (255, 0,   0),
    SegClass.BOUNDARY:   (0,   0,   255),
}

PHYSICAL_CORNER_REQUIREMENTS = {
    PhysicalCorner.TOP_LEFT:     {ViewCorner.TOP_LEFT, ViewCorner.TOP_RIGHT, ViewCorner.BOTTOM_LEFT},
    PhysicalCorner.TOP_RIGHT:    {ViewCorner.TOP_LEFT, ViewCorner.TOP_RIGHT, ViewCorner.BOTTOM_RIGHT},
    PhysicalCorner.BOTTOM_LEFT:  {ViewCorner.TOP_LEFT, ViewCorner.BOTTOM_LEFT, ViewCorner.BOTTOM_RIGHT},
    PhysicalCorner.BOTTOM_RIGHT: {ViewCorner.TOP_RIGHT, ViewCorner.BOTTOM_LEFT, ViewCorner.BOTTOM_RIGHT},
}


# ============================================================
# 0. 扫描配置
# ============================================================
@dataclass
class ScanConfig:
    frame_w: int = 1028
    frame_h: int = 776
    um_per_pixel: float = 1.0
    um_per_step: float = 1.0

    move_speed_steps: int = 100
    row_overlap_ratio: float = 0.2

    corner_roi_size: int = 80
    corner_ratio_threshold: float = 0.25
    corner_confirm_count: int = 4
    boundary_lock_margin_steps: int = 10

    skeleton_path_radius: int = 1
    skeleton_highlight_radius: int = 5
    skeleton_min_length: int = 100000      # 最小骨架长度【用于筛选最小的骨架 -- 过滤噪声】

    output_dir: str = r"F:/0_My_jupyter/0_Optical/scan_output"
    video_fps: int = 10
    save_debug_frames: bool = True

    @property
    def pixels_per_step(self) -> float:
        return self.um_per_step / self.um_per_pixel

    @property
    def frame_h_steps(self) -> int:
        effective_h = self.frame_h * (1.0 - self.row_overlap_ratio)
        return int(effective_h / 100 / 2 * self.um_per_pixel / self.um_per_step)

    @property
    def frame_w_steps(self) -> int:
        return int(self.frame_w * self.um_per_pixel / self.um_per_step * 3)


# ============================================================
# 1. 语义分割推理器
# ============================================================
class SegmentationInference:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(CLASS_NAMES_PATH) as f:
            self.class_names = [line.strip() for line in f]
        self.num_classes = len(self.class_names)
        logger.info(f"类别数: {self.num_classes}  类别: {self.class_names}")

        n_layers_dino = MODEL_TO_NUM_LAYERS[DINO_MODEL]
        dino_model = torch.hub.load(
            repo_or_dir=DINOV3_DIR, model=DINO_MODEL,
            source="local", weights=DINO_WEIGHTS,
        )
        self.backbone = DinoBackbone(dino_model, n_layers_dino).to(self.device)
        self.backbone.eval()

        embed_dim = MODEL_TO_EMBED_DIM[DINO_MODEL]
        self.head = ASPPDecoder(
            in_ch=embed_dim, num_classes=self.num_classes,
            target_size=(TARGET_SIZE, TARGET_SIZE),
        ).to(self.device)
        self.head.load_state_dict(torch.load(MODEL_PATH_INFERENCE, map_location=self.device))
        self.head.eval()
        logger.info(f"模型加载完成 → {self.device}")

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        original_h, original_w = frame_bgr.shape[:2]
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        image_tensor = image_to_tensor(image_resized, IMG_MEAN, IMG_STD).unsqueeze(0).to(self.device)
        feat = self.backbone(image_tensor)
        semantic_logits = self.head(feat)
        semantic_map = outputs_to_maps(semantic_logits, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(semantic_map.astype(np.uint8), (original_w, original_h),
                          interpolation=cv2.INTER_NEAREST)
        return mask


# ============================================================
# 2. 路径骨架提取器（新增干净主路径提取）
# ============================================================
class PathSkeletonExtractor:
    """
    路径骨架提取器。
    新增 extract_clean_main_path()：将骨架分解为链段，
    通过链段图 DFS 找到无环无分支的最长像素路径。
    """

    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg

    # ── 原有方法（保持不变）─────────────────────────────

    def extract_from_mask(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        h, w = mask.shape
        scale = 0.5
        small_h, small_w = int(h * scale), int(w * scale)

        if small_h < 10 or small_w < 10:
            road_mask = (mask == SegClass.ROAD).astype(np.uint8)
        else:
            small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            road_mask = (small_mask == SegClass.ROAD).astype(np.uint8)

        if np.sum(road_mask) == 0:
            logger.debug("  掩码中未检测到道路")
            return np.zeros_like(mask), np.zeros_like(mask), []
        
        # ★ 形态学预处理：闭运算填充小空洞，再开运算去除孤立噪点
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))         # 减少骨架断裂
        kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))         # 减少骨架毛刺源头
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  kernel_open)

        skel = skeletonize(road_mask > 0).astype(np.uint8)
        if np.sum(skel) == 0:
            return np.zeros_like(mask), np.zeros_like(mask), []

        if small_h != h or small_w != w:
            skel = cv2.resize(skel, (w, h), interpolation=cv2.INTER_NEAREST)

        # ★ 保留最大连通域（去除孤立短骨架片段）
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
        if n_labels > 2:  # 0=背景，1+为各连通域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            skel = (labels == largest_label).astype(np.uint8)

        skel = self._remove_burrs(skel, min_length=self.cfg.skeleton_min_length)
        if np.sum(skel) == 0:
            return np.zeros_like(mask), np.zeros_like(mask), []

        skeleton_pixels = np.argwhere(skel > 0)
        if len(skeleton_pixels) < 2:
            return skel, skel, []

        G = self._skeleton_to_graph(skel, skeleton_pixels)
        endpoints = self._find_endpoints(skel, skeleton_pixels)
        longest_path_points = self._find_longest_path(G, endpoints)

        longest_path_mask = np.zeros_like(skel)
        if len(longest_path_points) > 0:
            for (y, x) in longest_path_points:
                longest_path_mask[y, x] = 1

        logger.debug(f"  完整骨架像素数: {len(skeleton_pixels)}, 最长路径长度: {len(longest_path_points)}")
        return skel, longest_path_mask, longest_path_points

    def _skeleton_to_graph(self, skel: np.ndarray, skeleton_pixels: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for y, x in skeleton_pixels:
            G.add_node((y, x))
        for y, x in skeleton_pixels:
            for dy, dx in dirs:
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G:
                    G.add_edge((y, x), (ny, nx_), weight=1)
        return G

    def _find_endpoints(self, skel: np.ndarray, skeleton_pixels: np.ndarray) -> List[Tuple[int, int]]:
        endpoints = []
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for y, x in skeleton_pixels:
            nbs = sum(
                1 for dy, dx in dirs
                if 0 <= y+dy < skel.shape[0] and 0 <= x+dx < skel.shape[1]
                and skel[y+dy, x+dx] == 1
            )
            if nbs == 1:
                endpoints.append((y, x))
        return endpoints

    def _remove_burrs(self, skel: np.ndarray, min_length: int = 5) -> np.ndarray:
        skeleton_pixels = np.argwhere(skel > 0)
        if len(skeleton_pixels) < 2:
            return skel

        G = self._skeleton_to_graph(skel, skeleton_pixels)
        degrees = dict(G.degree())
        endpoints = [node for node, degree in degrees.items() if degree == 1]

        nodes_to_remove = set()
        for endpoint in endpoints:
            current = endpoint
            path = [current]
            visited = {current}
            while True:
                unvisited = [n for n in G.neighbors(current) if n not in visited]
                if not unvisited:
                    break
                nxt = unvisited[0]
                path.append(nxt)
                visited.add(nxt)
                if degrees[nxt] > 2:
                    break
                current = nxt
            if len(path) < min_length:
                nodes_to_remove.update(path)

        cleaned_skel = skel.copy()
        for (y, x) in nodes_to_remove:
            cleaned_skel[y, x] = 0
        return cleaned_skel

    def _find_longest_path(self, G: nx.Graph, endpoints: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(endpoints) < 2:
            return []

        max_endpoints = 20
        if len(endpoints) > max_endpoints:
            np.random.shuffle(endpoints)
            endpoints = endpoints[:max_endpoints]

        max_dist = -1
        best_pair = None
        early_stop_threshold = 0.9

        for i, start in enumerate(endpoints):
            if start not in G:
                continue
            try:
                distances = nx.single_source_shortest_path_length(G, start)
            except:
                continue
            for end in endpoints[i+1:]:
                if end not in distances:
                    continue
                dist = distances[end]
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (start, end)
                    if max_dist > len(G.nodes) * early_stop_threshold:
                        return nx.shortest_path(G, best_pair[0], best_pair[1])

        if best_pair is None:
            return []
        return nx.shortest_path(G, best_pair[0], best_pair[1])

    def draw_all_skeleton(self, image: np.ndarray, skeleton: np.ndarray,
                          color: Tuple[int, int, int] = (0, 0, 255),
                          radius: Optional[int] = None) -> np.ndarray:
        out = image.copy()
        r = radius if radius is not None else self.cfg.skeleton_path_radius
        for (y, x) in np.argwhere(skeleton > 0):
            cv2.circle(out, (int(x), int(y)), r, color, -1)
        return out

    def draw_skeleton_with_highlight(self, image: np.ndarray,
                                     full_skeleton: np.ndarray,
                                     longest_path: np.ndarray,
                                     base_color: Tuple[int, int, int] = (128, 128, 128),
                                     highlight_color: Tuple[int, int, int] = (0, 0, 255),
                                     base_radius: Optional[int] = None,
                                     highlight_radius: Optional[int] = None) -> np.ndarray:
        out = image.copy()
        br = base_radius if base_radius is not None else self.cfg.skeleton_path_radius
        hr = highlight_radius if highlight_radius is not None else self.cfg.skeleton_highlight_radius
        for (y, x) in np.argwhere(full_skeleton > 0):
            cv2.circle(out, (int(x), int(y)), br, base_color, -1)
        for (y, x) in np.argwhere(longest_path > 0):
            cv2.circle(out, (int(x), int(y)), hr, highlight_color, -1)
        return out

    # ── ★ 新增：干净主路径提取 ─────────────────────────────

    def extract_clean_main_path(
        self,
        mask: np.ndarray,
        precomputed_skel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        提取干净主路径：去除骨架中所有环路和分支，返回一条无分叉的像素链。

        算法：
          1. 将骨架分解为「链段」（两关节节点之间的连续像素序列）
          2. 在链段图中用 DFS 找像素总数最多的简单路径
             - 每条链段最多用一次 → 消除环路
             - 每个关节节点最多经过一次 → 消除分支

        Args:
            mask            : 语义分割掩码
            precomputed_skel: 已有完整骨架（None 则内部重新提取）

        Returns:
            clean_mask  : (H,W) uint8，路径像素=1
            path_points : [(y,x), ...] 按路径顺序排列
        """
        if precomputed_skel is not None:
            full_skel = precomputed_skel.copy()
        else:
            full_skel, _, _ = self.extract_from_mask(mask)

        if np.sum(full_skel) == 0:
            return np.zeros_like(mask, dtype=np.uint8), []

        skel_pts = np.argwhere(full_skel > 0)
        if len(skel_pts) < 2:
            return full_skel.copy(), [(int(r[0]), int(r[1])) for r in skel_pts]

        G = self._skeleton_to_graph(full_skel, skel_pts)
        chains = self._extract_chains(G)
        if not chains:
            return np.zeros_like(mask, dtype=np.uint8), []

        path_pixels = self._find_longest_chain_path(chains)
        if not path_pixels:
            return np.zeros_like(mask, dtype=np.uint8), []

        clean_mask = np.zeros(mask.shape, dtype=np.uint8)
        h, w = clean_mask.shape
        for y, x in path_pixels:
            if 0 <= y < h and 0 <= x < w:
                clean_mask[y, x] = 1

        return clean_mask, path_pixels

    def _extract_chains(self, G: nx.Graph) -> List[List[Tuple[int, int]]]:
        """
        将骨架图分解为链段列表。
        关节节点：端点(度=1) 或 分叉点(度≥3)。
        链段 = 从一个关节节点沿度=2 的节点延伸到下一个关节节点的像素序列。
        """
        junction_set = frozenset(n for n in G.nodes() if G.degree(n) != 2)

        # 纯环路（所有节点度=2）
        if not junction_set:
            start = next(iter(G.nodes()))
            return [list(nx.dfs_preorder_nodes(G, start))]

        chains = []
        visited_edges: set = set()

        for jn in junction_set:
            for nb in list(G.neighbors(jn)):
                edge_key = (jn, nb) if jn < nb else (nb, jn)
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

                chain = [jn, nb]
                prev, cur = jn, nb

                while cur not in junction_set:
                    nexts = [n for n in G.neighbors(cur) if n != prev]
                    if not nexts:
                        break
                    nxt = nexts[0]
                    ek = (cur, nxt) if cur < nxt else (nxt, cur)
                    visited_edges.add(ek)
                    chain.append(nxt)
                    prev, cur = cur, nxt

                if len(chain) >= 2:
                    chains.append(chain)

        return chains

    def _find_longest_chain_path(
        self,
        chains: List[List[Tuple[int, int]]],
    ) -> List[Tuple[int, int]]:
        """
        在链段图中用 DFS 枚举所有简单路径，返回像素总数最多的一条。
        节点数超过 300 时自动切换为贪心策略以控制耗时。
        """
        _MAX_DFS_NODES = 300

        if not chains:
            return []

        # 构建链段邻接表
        adj: Dict = {}
        for idx, chain in enumerate(chains):
            a, b = chain[0], chain[-1]
            adj.setdefault(a, []).append((idx, b))
            if a != b:
                adj.setdefault(b, []).append((idx, a))

        all_nodes = list(adj.keys())

        if len(all_nodes) > _MAX_DFS_NODES:
            return self._greedy_chain_path(chains, adj)

        best_len:  List[int]  = [0]
        best_segs: List[list] = [[]]

        def dfs(node, used_chains, used_nodes, cur_len, cur_segs):
            if cur_len > best_len[0]:
                best_len[0] = cur_len
                best_segs[0] = cur_segs[:]
            for cidx, nxt in adj.get(node, []):
                if cidx in used_chains or nxt in used_nodes:
                    continue
                used_chains.add(cidx)
                used_nodes.add(nxt)
                cur_segs.append((cidx, node))
                dfs(nxt, used_chains, used_nodes,
                    cur_len + len(chains[cidx]), cur_segs)
                cur_segs.pop()
                used_nodes.discard(nxt)
                used_chains.discard(cidx)

        for start in all_nodes:
            dfs(start, set(), {start}, 0, [])

        if not best_segs[0]:
            longest = max(chains, key=len)
            return [(int(y), int(x)) for y, x in longest]

        # 按顺序拼接像素（共享端点去重）
        pixel_path: List[Tuple[int, int]] = []
        for cidx, from_node in best_segs[0]:
            chain = chains[cidx]
            seg = chain if chain[0] == from_node else chain[::-1]
            a = (int(seg[0][0]), int(seg[0][1]))
            if pixel_path and pixel_path[-1] == a:
                pixel_path.extend((int(y), int(x)) for y, x in seg[1:])
            else:
                pixel_path.extend((int(y), int(x)) for y, x in seg)

        return pixel_path

    def _greedy_chain_path(
        self,
        chains: List[List[Tuple[int, int]]],
        adj: Dict,
    ) -> List[Tuple[int, int]]:
        """贪心：每步选最长未用链段延伸（大图快速估算）。"""
        if not chains:
            return []

        degrees = {node: len(adj[node]) for node in adj}
        endpoints = [n for n, d in degrees.items() if d == 1]
        start = endpoints[0] if endpoints else next(iter(adj))

        pixel_path: List[Tuple[int, int]] = []
        used_chains: set = set()
        used_nodes:  set = {start}
        cur = start

        while True:
            candidates = [
                (idx, nxt, len(chains[idx]))
                for idx, nxt in adj.get(cur, [])
                if idx not in used_chains and nxt not in used_nodes
            ]
            if not candidates:
                break
            cidx, nxt, _ = max(candidates, key=lambda t: t[2])

            chain = chains[cidx]
            seg = chain if chain[0] == cur else chain[::-1]
            a = (int(seg[0][0]), int(seg[0][1]))
            if pixel_path and pixel_path[-1] == a:
                pixel_path.extend((int(y), int(x)) for y, x in seg[1:])
            else:
                pixel_path.extend((int(y), int(x)) for y, x in seg)

            used_chains.add(cidx)
            used_nodes.add(nxt)
            cur = nxt

        return pixel_path

    def _full_coverage_cutting_path(
        self,
        G: nx.Graph,
        start_point: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """
        全路径覆盖切割算法：确保切割路径经过骨架上的所有路径。

        算法策略：
          1. 从左上角起始点开始
          2. 使用深度优先搜索(DFS)遍历所有节点
          3. 遇到分支时，优先选择向右、向下的方向
          4. 遍历完成后，形成一条覆盖所有路径的切割路径

        Args:
            G           : 骨架图
            start_point : 起始点
            shape       : 图像形状 (h, w)

        Returns:
            cutting_points : 切割路径点列表
            directions     : 方向信息列表
        """
        h, w = shape
        visited = set()
        cutting_points = []
        directions = []
        
        def dfs(current: Tuple[int, int], prev: Optional[Tuple[int, int]]):
            if current not in G:
                return
            
            # 每次访问都添加到路径中，确保所有路径都被覆盖
            cutting_points.append(current)
            
            # 计算方向
            if prev is not None:
                dy = current[0] - prev[0]
                dx = current[1] - prev[1]
                dir_idx = self._get_direction_index(dy, dx)
                directions.append((current[0], current[1], dir_idx))
            
            # 标记为已访问
            visited.add(current)
            
            # 获取邻居并按优先级排序
            neighbors = list(G.neighbors(current))
            if not neighbors:
                return
            
            # 按方向优先级排序：右 > 下 > 右下 > 右上 > 左下 > 左 > 上 > 左上
            def direction_priority(neighbor):
                dy = neighbor[0] - current[0]
                dx = neighbor[1] - current[1]
                # 计算方向得分
                score = dx * 2 + dy * 1.5
                # 距离惩罚
                dist = abs(dy) + abs(dx)
                if dist > 1:
                    score -= dist * 0.5
                # 未访问的邻居优先
                if neighbor not in visited:
                    score += 10
                return score
            
            # 按优先级排序邻居
            neighbors.sort(key=direction_priority, reverse=True)
            
            # 递归访问邻居
            for neighbor in neighbors:
                if neighbor not in visited:
                    dfs(neighbor, current)
                elif neighbor != prev:
                    # 已经访问过的节点，但不是前一个节点，可能是环路
                    # 记录返回方向
                    dy = neighbor[0] - current[0]
                    dx = neighbor[1] - current[1]
                    dir_idx = self._get_direction_index(dy, dx)
                    directions.append((current[0], current[1], dir_idx))
                    cutting_points.append(neighbor)
        
        # 开始DFS遍历
        dfs(start_point, None)
        
        return cutting_points, directions

    def extract_cutting_path(
        self,
        mask: np.ndarray,
        precomputed_skel: Optional[np.ndarray] = None,
        cover_all_paths: bool = False
    ):
        """
        提取芯片实际切割路径：从左上角开始，尽可能一刀切到底。

        算法策略：
          1. 从骨架的左上角点开始
          2. 沿着骨架向右下方延伸，优先选择向右、向下的方向
          3. 遇到分支时，选择最接近目标方向（右下）的路径
          4. 允许必要的返回，但尽量保持前进方向

        Args:
            mask            : 语义分割掩码
            precomputed_skel: 已有完整骨架（None 则内部重新提取）
            cover_all_paths : 是否覆盖所有路径（默认False）

        Returns:
            cutting_mask   : (H,W) uint8，切割路径像素=1
            cutting_points : [(y,x), ...] 按切割顺序排列的像素坐标
            directions     : [(y,x,dir_idx), ...] 每个点的方向索引
                            dir_idx: 0=右, 1=下, 2=左, 3=上, 4=右下, 5=左下, 6=右上, 7=左上
        """
        if precomputed_skel is not None:
            full_skel = precomputed_skel.copy()
        else:
            if cover_all_paths:
                # 对于全路径覆盖，使用原始骨架（不移除分支）
                road_mask = (mask == SegClass.ROAD).astype(np.uint8)
                if np.sum(road_mask) == 0:
                    return np.zeros_like(mask, dtype=np.uint8), [], []
                
                # 形态学预处理
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_close)
                road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_open)
                
                full_skel = skeletonize(road_mask > 0).astype(np.uint8)
                if np.sum(full_skel) == 0:
                    return np.zeros_like(mask, dtype=np.uint8), [], []
            else:
                # 对于普通切割路径，使用处理后的骨架
                full_skel, _, _ = self.extract_from_mask(mask)

        if np.sum(full_skel) == 0:
            return np.zeros_like(mask, dtype=np.uint8), [], []

        skel_pts = np.argwhere(full_skel > 0)
        if len(skel_pts) < 2:
            return full_skel.copy(), [(int(r[0]), int(r[1])) for r in skel_pts], []

        G = self._skeleton_to_graph(full_skel, skel_pts)
        
        # 找到左上角的起始点
        start_point = self._find_top_left_start_point(skel_pts)
        if start_point is None:
            return np.zeros_like(mask, dtype=np.uint8), [], []

        # 根据需求选择不同的路径提取策略
        if cover_all_paths:
            # 使用全路径覆盖算法
            cutting_points, directions = self._full_coverage_cutting_path(G, start_point, full_skel.shape)
        else:
            # 使用贪心策略从左上角向右下方延伸
            cutting_points, directions = self._greedy_cutting_path(G, start_point, full_skel.shape)
        
        if not cutting_points:
            return np.zeros_like(mask, dtype=np.uint8), [], []

        cutting_mask = np.zeros(mask.shape, dtype=np.uint8)
        h, w = cutting_mask.shape
        for y, x in cutting_points:
            if 0 <= y < h and 0 <= x < w:
                cutting_mask[y, x] = 1

        return cutting_mask, cutting_points, directions

    def _find_top_left_start_point(self, skel_pts: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        找到骨架的左上角起始点。
        优先选择最左上角的端点，如果没有端点则选择最左上角的分叉点。
        """
        if len(skel_pts) == 0:
            return None

        # 计算每个点的"左上角程度"：y + x 值越小越接近左上角
        scores = [(y + x, (y, x)) for y, x in skel_pts]
        scores.sort(key=lambda x: x[0])
        
        # 返回最左上角的点
        return scores[0][1]

    def _get_direction_index(self, dy: int, dx: int) -> int:
        """
        根据方向向量获取方向索引。

        Args:
            dy : y方向的变化
            dx : x方向的变化

        Returns:
            dir_idx: 方向索引
                    0=右, 1=下, 2=左, 3=上, 4=右下, 5=左下, 6=右上, 7=左上
        """
        if dy == 0 and dx == 1:
            return 0  # 右
        elif dy == 1 and dx == 0:
            return 1  # 下
        elif dy == 0 and dx == -1:
            return 2  # 左
        elif dy == -1 and dx == 0:
            return 3  # 上
        elif dy == 1 and dx == 1:
            return 4  # 右下
        elif dy == 1 and dx == -1:
            return 5  # 左下
        elif dy == -1 and dx == 1:
            return 6  # 右上
        elif dy == -1 and dx == -1:
            return 7  # 左上
        else:
            return 0  # 默认右

    def _greedy_cutting_path(
        self,
        G: nx.Graph,
        start_point: Tuple[int, int],
        shape: Tuple[int, int]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """
        使用贪心策略从起始点向右下方延伸，生成切割路径。

        方向优先级（从高到低）：
          0: 右 (0, 1)
          1: 下 (1, 0)
          4: 右下 (1, 1)
          6: 右上 (-1, 1)
          5: 左下 (1, -1)
          2: 左 (0, -1)
          3: 上 (-1, 0)
          7: 左上 (-1, -1)
        """
        h, w = shape
        
        # 定义8个方向及其优先级
        directions = [
            (0, 1),    # 0: 右
            (1, 0),    # 1: 下
            (1, 1),    # 4: 右下
            (-1, 1),   # 6: 右上
            (1, -1),   # 5: 左下
            (0, -1),   # 2: 左
            (-1, 0),   # 3: 上
            (-1, -1),  # 7: 左上
        ]
        
        cutting_points = []
        point_directions = []
        visited = set()
        current = start_point
        
        max_steps = 10000  # 防止无限循环
        steps = 0
        
        while current in G and steps < max_steps:
            cutting_points.append(current)
            visited.add(current)
            
            # 获取邻居节点
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            
            # 过滤已访问的节点，但允许返回（保留一定比例的未访问节点）
            unvisited = [n for n in neighbors if n not in visited]
            
            if unvisited:
                # 有未访问的邻居，选择最符合前进方向的节点
                best_neighbor = None
                best_score = float('-inf')
                best_dir_idx = 0
                
                for neighbor in unvisited:
                    dy, dx = neighbor[0] - current[0], neighbor[1] - current[1]
                    
                    # 计算方向得分：优先选择向右、向下的方向
                    # 向右(dx>0)和向下(dy>0)得分更高
                    score = dx * 2 + dy * 1.5
                    
                    # 距离惩罚：避免跳跃
                    dist = abs(dy) + abs(dx)
                    if dist > 1:
                        score -= dist * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
                        best_dir_idx = self._get_direction_index(dy, dx)
                
                if best_neighbor is not None:
                    point_directions.append((current[0], current[1], best_dir_idx))
                    current = best_neighbor
                else:
                    break
            else:
                # 所有邻居都已访问，选择最近访问的节点（允许返回）
                # 但优先选择距离当前点较近的节点
                if len(cutting_points) > 1:
                    # 尝试回退到上一个节点
                    prev_point = cutting_points[-2]
                    dy, dx = prev_point[0] - current[0], prev_point[1] - current[1]
                    point_directions.append((current[0], current[1], self._get_direction_index(dy, dx)))
                    current = prev_point
                    # 移除当前点，避免无限循环
                    cutting_points.pop()
                    visited.remove(current)
                else:
                    break
            
            steps += 1
        
        return cutting_points, point_directions

    def _get_direction_index(self, dy: int, dx: int) -> int:
        """
        根据位移向量获取方向索引。
        返回值: 0=右, 1=下, 2=左, 3=上, 4=右下, 5=左下, 6=右上, 7=左上
        """
        direction_map = {
            (0, 1): 0,    # 右
            (1, 0): 1,    # 下
            (0, -1): 2,   # 左
            (-1, 0): 3,   # 上
            (1, 1): 4,    # 右下
            (1, -1): 5,   # 左下
            (-1, 1): 6,   # 右上
            (-1, -1): 7,  # 左上
        }
        return direction_map.get((dy, dx), 0)

    def draw_cutting_path_with_arrows(
        self,
        image: np.ndarray,
        cutting_mask: np.ndarray,
        directions: List[Tuple[int, int, int]],
        path_color: Tuple[int, int, int] = (0, 255, 255),
        arrow_color: Tuple[int, int, int] = (255, 0, 0),
        arrow_spacing: int = 20,
        arrow_size: int = 10
    ) -> np.ndarray:
        """
        绘制切割路径，并在路径上标注方向箭头。

        Args:
            image         : 输入图像
            cutting_mask  : 切割路径掩码
            directions    : 方向信息列表 [(y,x,dir_idx), ...]
            path_color    : 路径颜色 (B, G, R)
            arrow_color   : 箭头颜色 (B, G, R)
            arrow_spacing : 箭头间距（像素）
            arrow_size    : 箭头大小

        Returns:
            带有切割路径和方向箭头的图像
        """
        import cv2
        
        output = image.copy()
        
        # 绘制路径
        for (y, x) in np.argwhere(cutting_mask > 0):
            cv2.circle(output, (int(x), int(y)), self.cfg.skeleton_path_radius, path_color, -1)
        
        # 绘制方向箭头
        arrow_directions = [
            (0, 1),    # 0: 右
            (1, 0),    # 1: 下
            (0, -1),   # 2: 左
            (-1, 0),   # 3: 上
            (1, 1),    # 4: 右下
            (1, -1),   # 5: 左下
            (-1, 1),   # 6: 右上
            (-1, -1),  # 7: 左上
        ]
        
        # 按间隔绘制箭头
        for i in range(0, len(directions), arrow_spacing):
            y, x, dir_idx = directions[i]
            if dir_idx < len(arrow_directions):
                dy, dx = arrow_directions[dir_idx]
                
                # 计算箭头终点
                end_y = y + dy * arrow_size
                end_x = x + dx * arrow_size
                
                # 绘制箭头
                cv2.arrowedLine(output, (int(x), int(y)), (int(end_x), int(end_y)), 
                               arrow_color, 2, tipLength=0.5)
        
        return output
    def simulate_cutting_process(
        self,
        skeleton_mask: np.ndarray,
        cutting_points: List[Tuple[int, int]],
        cutting_directions: List[Tuple[int, int, int]],
        display_delay: int = 100
    ):
        """
        在最终骨架上模拟切割过程，逐步显示切割进度。

        Args:
            skeleton_mask    : 最终骨架掩码
            cutting_points   : 切割路径点列表
            cutting_directions: 方向信息列表
            display_delay    : 显示延迟（毫秒）
        """
        import cv2
        
        if not cutting_points:
            print("❌ 没有切割路径，无法模拟切割过程")
            return
        
        print("\n" + "=" * 70)
        print("🔪 切割过程模拟")
        print("=" * 70)
        print(f"   - 切割路径点数量: {len(cutting_points)}")
        print(f"   - 方向标注数量: {len(cutting_directions)}")
        print(f"   - 起始点: {cutting_points[0]}")
        print(f"   - 终止点: {cutting_points[-1]}")
        print(f"   - 显示延迟: {display_delay}ms")
        print("\n🏃 开始模拟切割过程...")
        print("   按 ESC 键退出模拟")
        print("=" * 70 + "\n")
        
        # 创建显示窗口
        window_name = "切割过程模拟"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 计算路径长度
        path_length = 0
        for i in range(1, len(cutting_points)):
            y1, x1 = cutting_points[i-1]
            y2, x2 = cutting_points[i]
            path_length += np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        print(f"   - 路径总长度: {path_length:.2f} 像素")
        
        # 模拟切割过程
        for i in range(len(cutting_points)):
            # 创建当前帧
            current_frame = np.zeros((*skeleton_mask.shape, 3), dtype=np.uint8)
            
            # 绘制完整骨架（灰色）
            for y, x in np.argwhere(skeleton_mask > 0):
                cv2.circle(current_frame, (int(x), int(y)), 1, (100, 100, 100), -1)
            
            # 绘制已切割的部分（绿色）
            for j in range(i + 1):
                yj, xj = cutting_points[j]
                cv2.circle(current_frame, (int(xj), int(yj)), 2, (0, 255, 0), -1)
            
            # 绘制未切割的部分（红色）
            for j in range(i + 1, len(cutting_points)):
                yj, xj = cutting_points[j]
                cv2.circle(current_frame, (int(xj), int(yj)), 2, (0, 0, 255), -1)
            
            # 绘制当前位置（蓝色）
            y, x = cutting_points[i]
            cv2.circle(current_frame, (int(x), int(y)), 4, (255, 0, 0), -1)
            
            # 添加方向信息
            if i < len(cutting_directions):
                _, _, dir_idx = cutting_directions[i]
                direction_names = ['右', '下', '左', '上', '右下', '左下', '右上', '左上']
                dir_name = direction_names[dir_idx] if dir_idx < len(direction_names) else f'方向{dir_idx}'
                
                # 绘制方向箭头
                arrow_directions = [
                    (0, 1), (1, 0), (0, -1), (-1, 0),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                if dir_idx < len(arrow_directions):
                    dy, dx = arrow_directions[dir_idx]
                    arrow_size = 15
                    end_y = y + dy * arrow_size
                    end_x = x + dx * arrow_size
                    cv2.arrowedLine(current_frame, (int(x), int(y)), 
                                   (int(end_x), int(end_y)), 
                                   (0, 255, 255), 2, tipLength=0.5)
                
                # 添加文字信息
                cv2.putText(current_frame, f"方向: {dir_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 添加进度信息
            progress = (i + 1) / len(cutting_points) * 100
            cv2.putText(current_frame, f"进度: {progress:.1f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(current_frame, f"当前点: ({x}, {y})", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(current_frame, f"已切割: {i+1}/{len(cutting_points)}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示结果
            cv2.imshow(window_name, current_frame)
            
            # 检查键盘输入
            key = cv2.waitKey(display_delay)
            if key == 27:  # ESC 键
                print("\n🛑 切割模拟已取消")
                break
        
        cv2.destroyAllWindows()
        print("\n✅ 切割模拟完成！")



# ============================================================
# 3. 几何角落检测器
# ============================================================
@dataclass
class PhysicalCornerState:
    corner_type: PhysicalCorner
    is_locked: bool = False
    lock_position_steps: Optional[Tuple[int, int]] = None
    consecutive_count: int = 0

    def update(self, detected: bool, current_pos: Tuple[int, int], confirm_n: int) -> bool:
        if self.is_locked:
            return False
        if detected:
            self.consecutive_count += 1
            if self.consecutive_count >= confirm_n:
                self.is_locked = True
                self.lock_position_steps = current_pos
                return True
        else:
            self.consecutive_count = 0
        return False

    def reset(self):
        self.is_locked = False
        self.lock_position_steps = None
        self.consecutive_count = 0


class GeometricCornerDetector:
    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        self.roi_size = cfg.corner_roi_size
        self.threshold = cfg.corner_ratio_threshold
        self.confirm_n = cfg.corner_confirm_count
        self.margin = cfg.boundary_lock_margin_steps

        self._corners: Dict[PhysicalCorner, PhysicalCornerState] = {
            corner: PhysicalCornerState(corner) for corner in PhysicalCorner
        }
        self._view_rois = self._compute_view_rois(cfg.frame_w, cfg.frame_h)
        self._current_pos: Tuple[int, int] = (0, 0)
        self._cached_boundaries: Dict[Direction, Optional[int]] = {
            Direction.TOP: None, Direction.BOTTOM: None,
            Direction.LEFT: None, Direction.RIGHT: None,
        }

    def update_position(self, x_steps: int, y_steps: int):
        self._current_pos = (x_steps, y_steps)

    def _compute_view_rois(self, w: int, h: int) -> Dict[ViewCorner, Tuple[int, int, int, int]]:
        s = self.roi_size
        return {
            ViewCorner.TOP_LEFT:     (0, 0, s, s),
            ViewCorner.TOP_RIGHT:    (w - s, 0, w, s),
            ViewCorner.BOTTOM_LEFT:  (0, h - s, s, h),
            ViewCorner.BOTTOM_RIGHT: (w - s, h - s, w, h),
        }

    def _get_region(self, mask: np.ndarray, corner: ViewCorner) -> np.ndarray:
        x1, y1, x2, y2 = self._view_rois[corner]
        return mask[y1:y2, x1:x2]

    def _ratio(self, region: np.ndarray) -> float:
        if region.size == 0:
            return 0.0
        return float(np.sum(region == SegClass.BOUNDARY)) / region.size

    def _detect_view_corners(self, mask: np.ndarray) -> Set[ViewCorner]:
        detected = set()
        for corner in ViewCorner:
            region = self._get_region(mask, corner)
            if self._ratio(region) >= self.threshold:
                detected.add(corner)
        return detected

    def _check_physical_corner(self, mask: np.ndarray, physical_corner: PhysicalCorner) -> bool:
        required = PHYSICAL_CORNER_REQUIREMENTS[physical_corner]
        detected = self._detect_view_corners(mask)
        return required.issubset(detected)

    def _update_and_lock_corners(self, mask: np.ndarray):
        for corner in PhysicalCorner:
            if self._corners[corner].is_locked:
                continue
            detected = self._check_physical_corner(mask, corner)
            if self._corners[corner].update(detected, self._current_pos, self.confirm_n):
                logger.info(f"  🔒 物理角锁定: {corner.name} @ {self._current_pos}")

    def _infer_boundaries_from_corners(self):
        tl = self._corners[PhysicalCorner.TOP_LEFT]
        tr = self._corners[PhysicalCorner.TOP_RIGHT]
        if tl.is_locked and tr.is_locked and self._cached_boundaries[Direction.TOP] is None:
            y = (tl.lock_position_steps[1] + tr.lock_position_steps[1]) // 2 + self.margin
            self._cached_boundaries[Direction.TOP] = y
            logger.info(f"  📏 推断上边界: Y={y}")

        bl = self._corners[PhysicalCorner.BOTTOM_LEFT]
        br = self._corners[PhysicalCorner.BOTTOM_RIGHT]
        if bl.is_locked and br.is_locked and self._cached_boundaries[Direction.BOTTOM] is None:
            y = (bl.lock_position_steps[1] + br.lock_position_steps[1]) // 2 - self.margin
            self._cached_boundaries[Direction.BOTTOM] = y
            logger.info(f"  📏 推断下边界: Y={y}")

        if tl.is_locked and bl.is_locked and self._cached_boundaries[Direction.LEFT] is None:
            x = (tl.lock_position_steps[0] + bl.lock_position_steps[0]) // 2 + self.margin
            self._cached_boundaries[Direction.LEFT] = x
            logger.info(f"  📏 推断左边界: X={x}")

        if tr.is_locked and br.is_locked and self._cached_boundaries[Direction.RIGHT] is None:
            x = (tr.lock_position_steps[0] + br.lock_position_steps[0]) // 2 - self.margin
            self._cached_boundaries[Direction.RIGHT] = x
            logger.info(f"  📏 推断右边界: X={x}")

    def has_top_boundary(self, mask: np.ndarray) -> bool:
        self._update_and_lock_corners(mask)
        self._infer_boundaries_from_corners()
        if self._cached_boundaries[Direction.TOP] is not None:
            return self._current_pos[1] <= self._cached_boundaries[Direction.TOP]
        detected = self._detect_view_corners(mask)
        return (ViewCorner.TOP_LEFT in detected and ViewCorner.TOP_RIGHT in detected)

    def has_bottom_boundary(self, mask: np.ndarray) -> bool:
        self._update_and_lock_corners(mask)
        self._infer_boundaries_from_corners()
        if self._cached_boundaries[Direction.BOTTOM] is not None:
            return self._current_pos[1] >= self._cached_boundaries[Direction.BOTTOM]
        detected = self._detect_view_corners(mask)
        return (ViewCorner.BOTTOM_LEFT in detected and ViewCorner.BOTTOM_RIGHT in detected)

    def has_left_boundary(self, mask: np.ndarray) -> bool:
        self._update_and_lock_corners(mask)
        self._infer_boundaries_from_corners()
        if self._cached_boundaries[Direction.LEFT] is not None:
            return self._current_pos[0] <= self._cached_boundaries[Direction.LEFT]
        detected = self._detect_view_corners(mask)
        return (ViewCorner.TOP_LEFT in detected and ViewCorner.BOTTOM_LEFT in detected)

    def has_right_boundary(self, mask: np.ndarray) -> bool:
        self._update_and_lock_corners(mask)
        self._infer_boundaries_from_corners()
        if self._cached_boundaries[Direction.RIGHT] is not None:
            return self._current_pos[0] >= self._cached_boundaries[Direction.RIGHT]
        detected = self._detect_view_corners(mask)
        return (ViewCorner.TOP_RIGHT in detected and ViewCorner.BOTTOM_RIGHT in detected)

    def reset_direction(self, direction: str):
        dir_map = {
            "top": Direction.TOP, "bottom": Direction.BOTTOM,
            "left": Direction.LEFT, "right": Direction.RIGHT
        }
        if direction not in dir_map:
            return
        d = dir_map[direction]
        self._cached_boundaries[d] = None
        if d == Direction.TOP:
            self._corners[PhysicalCorner.TOP_LEFT].reset()
            self._corners[PhysicalCorner.TOP_RIGHT].reset()
        elif d == Direction.BOTTOM:
            self._corners[PhysicalCorner.BOTTOM_LEFT].reset()
            self._corners[PhysicalCorner.BOTTOM_RIGHT].reset()
        elif d == Direction.LEFT:
            self._corners[PhysicalCorner.TOP_LEFT].reset()
            self._corners[PhysicalCorner.BOTTOM_LEFT].reset()
        elif d == Direction.RIGHT:
            self._corners[PhysicalCorner.TOP_RIGHT].reset()
            self._corners[PhysicalCorner.BOTTOM_RIGHT].reset()
        logger.info(f"  ↺ 重置 {direction} 边界及相关物理角")

    def reset_all(self):
        for c in PhysicalCorner:
            self._corners[c].reset()
        for d in Direction:
            self._cached_boundaries[d] = None
        logger.info("  ↺ 重置所有边界和物理角")

    def get_status(self) -> Dict:
        return {
            "physical_corners": {
                c.name: {
                    "locked": self._corners[c].is_locked,
                    "pos": self._corners[c].lock_position_steps,
                    "consecutive": self._corners[c].consecutive_count
                } for c in PhysicalCorner
            },
            "inferred_boundaries": {d.name: self._cached_boundaries[d] for d in Direction},
            "current_pos": self._current_pos
        }

    def status_str(self, mask: np.ndarray) -> str:
        detected = self._detect_view_corners(mask)
        view_status = "".join([
            "TL" if ViewCorner.TOP_LEFT in detected else "..",
            "TR" if ViewCorner.TOP_RIGHT in detected else "..",
            "BL" if ViewCorner.BOTTOM_LEFT in detected else "..",
            "BR" if ViewCorner.BOTTOM_RIGHT in detected else "..",
        ])
        corner_symbols = []
        for c in PhysicalCorner:
            state = self._corners[c]
            if state.is_locked:
                corner_symbols.append(f"{c.name[0]}{c.name[-1]}🔒")
            elif state.consecutive_count > 0:
                corner_symbols.append(f"{c.name[0]}{c.name[-1]}{state.consecutive_count}")
            else:
                corner_symbols.append(f"{c.name[0]}{c.name[-1]}○")
        locked_bounds = [f"{d.name[0]}@{v}" for d, v in self._cached_boundaries.items() if v is not None]
        lock_str = f" | {','.join(locked_bounds)}" if locked_bounds else ""
        return f"[{view_status}] [{' '.join(corner_symbols)}]{lock_str}"


# ============================================================
# 4. 硬件接口
# ============================================================
class HardwareInterface:
    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg

    def grab_frame(self) -> np.ndarray:
        raise NotImplementedError

    def grab_frame_and_mask(self, seg_model: SegmentationInference) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.grab_frame()
        mask = seg_model.predict(frame)
        return frame, mask

    def move_x(self, steps: int):
        raise NotImplementedError

    def move_y(self, steps: int):
        raise NotImplementedError


# ============================================================
# 5. 全局地图拼接器
# ============================================================
class GlobalMapStitcher:
    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        self.tiles: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
        self.cur_gx = 0
        self.cur_gy = 0
        self.skeleton_extractor = PathSkeletonExtractor(cfg)
        self.det, self.matcher, self.method = self._create_detector()

    def _create_detector(self):
        try:
            det = cv2.SIFT_create(nfeatures=3000)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            logger.info("[特征] 使用 SIFT 特征检测器")
            return det, matcher, "SIFT"
        except:
            det = cv2.ORB_create(nfeatures=3000)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            logger.info("[特征] 使用 ORB 特征检测器")
            return det, matcher, "ORB"

    def _match_features(self, desc1, desc2, ratio: float = 0.75) -> list:
        if desc1 is None or desc2 is None:
            return []
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < ratio * n.distance:
                    good.append(m)
        return good

    def _estimate_homography(self, kp1, kp2, matches, min_match: int = 10):
        if len(matches) < min_match:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0,
                                     maxIters=2000, confidence=0.995)
        if H is None:
            return None, None
        inliers = mask.ravel().sum()
        logger.debug(f"    匹配点={len(matches)}, 内点={inliers}")
        if inliers < min_match:
            return None, None
        return H, mask

    def record_tile(self, frame_bgr: np.ndarray, mask: np.ndarray):
        self.tiles.append((self.cur_gx, self.cur_gy, frame_bgr.copy(), mask.copy()))

    def update_position(self, dx_pixels: int, dy_pixels: int):
        self.cur_gx += dx_pixels
        self.cur_gy += dy_pixels

    def build_global_map(self, use_feature_matching: bool = True,
                         output_size_limit: int = 8000):
        """
        构建全局地图。

        Returns (10元组):
            global_map_with_skeleton         : 全局地图（灰骨架+红最长路径+黄干净路径+青切割路径）
            global_mask                      : 原始语义掩码
            global_mask_color_with_skeleton  : 彩色语义掩码+完整骨架
            full_skeleton                    : 完整骨架二值图
            longest_path_mask                : 最长路径二值图（含环/分支，原样保留）
            longest_path_points              : 最长路径点列表
            clean_path_mask                  : ★ 干净主路径二值图（无环无分支）
            clean_path_points                : ★ 干净主路径点列表
            cutting_path_mask                : ★ 切割路径二值图（从左上角开始）
            cutting_path_points              : ★ 切割路径点列表
            cutting_directions                : ★ 切割路径方向信息 [(y,x,dir_idx), ...]
        """
        if not self.tiles:
            empty = np.zeros((1, 1, 3), np.uint8)
            return (empty, np.zeros((1, 1), np.uint8), empty,
                    np.zeros((1, 1)), np.zeros((1, 1)), [],
                    np.zeros((1, 1)), [],
                    np.zeros((1, 1)), [], [])

        logger.info(f"[全局地图] 开始构建，共 {len(self.tiles)} 个tiles...")

        if use_feature_matching and len(self.tiles) >= 2:
            return self._build_with_feature_matching(output_size_limit)
        else:
            return self._build_simple()

    def _build_with_feature_matching(self, output_size_limit: int):
        logger.info("[特征提取] 计算所有tile的特征点...")
        kps, descs = [], []
        for i, (gx, gy, frame, mask) in enumerate(self.tiles):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            kp, desc = self.det.detectAndCompute(gray, None)
            kps.append(kp)
            descs.append(desc)
            if (i+1) % 10 == 0:
                logger.info(f"  已处理 {i+1}/{len(self.tiles)} tiles")

        logger.info("[变换估计] 计算tile间的变换关系...")
        H_global = [np.eye(3, dtype=np.float64)]
        for i in range(1, len(self.tiles)):
            good_matches = self._match_features(descs[i-1], descs[i])
            H_local, _ = self._estimate_homography(kps[i-1], kps[i], good_matches)
            if H_local is not None:
                H_global.append(H_global[-1] @ H_local)
            else:
                dx = self.tiles[i][0] - self.tiles[i-1][0]
                dy = self.tiles[i][1] - self.tiles[i-1][1]
                H_pos = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
                H_global.append(H_global[-1] @ H_pos)

        h, w = self.tiles[0][2].shape[:2]
        corners_base = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        all_corners = []
        for H in H_global:
            c = cv2.perspectiveTransform(corners_base.reshape(-1, 1, 2), H).reshape(-1, 2)
            all_corners.append(c)
        all_corners = np.vstack(all_corners)

        x_min, y_min = all_corners.min(axis=0)
        x_max, y_max = all_corners.max(axis=0)
        canvas_w = int(np.ceil(x_max - x_min))
        canvas_h = int(np.ceil(y_max - y_min))

        scale = 1.0
        if canvas_w > output_size_limit or canvas_h > output_size_limit:
            scale = output_size_limit / max(canvas_w, canvas_h)
            canvas_w = int(canvas_w * scale)
            canvas_h = int(canvas_h * scale)
            logger.info(f"[画布缩放] 输出过大，缩放比例={scale:.3f}")

        logger.info(f"[画布] 尺寸: {canvas_w} x {canvas_h} px")
        T = np.array([[scale, 0, -x_min*scale],
                      [0, scale, -y_min*scale],
                      [0, 0, 1]], dtype=np.float64)

        logger.info("[拼接] 正在融合tiles到全局地图...")
        canvas      = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        canvas_mask = np.zeros((canvas_h, canvas_w),    dtype=np.float32)
        weight_map  = np.zeros((canvas_h, canvas_w),    dtype=np.float32)

        for i, ((gx, gy, frame, mask), H) in enumerate(zip(self.tiles, H_global)):
            H_final = T @ H
            warped_frame = cv2.warpPerspective(
                frame.astype(np.float32), H_final, (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warped_mask = cv2.warpPerspective(
                mask.astype(np.float32), H_final, (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            mask_valid = (warped_mask > 0).astype(np.uint8)
            dist_weight = cv2.distanceTransform(mask_valid, cv2.DIST_L2, 3).astype(np.float32)
            dist_weight = np.clip(dist_weight / (dist_weight.max() + 1e-6), 0, 1)
            canvas      += warped_frame * dist_weight[:, :, None]
            canvas_mask += warped_mask  * dist_weight
            weight_map  += dist_weight
            if (i+1) % 10 == 0:
                logger.info(f"  已融合 {i+1}/{len(self.tiles)} tiles")

        weight_map  = np.maximum(weight_map, 1e-6)
        canvas      = canvas      / weight_map[:, :, None]
        canvas_mask = canvas_mask / weight_map
        canvas_bgr  = np.clip(canvas,      0, 255).astype(np.uint8)
        canvas_mask = np.clip(canvas_mask, 0, 255).astype(np.uint8)
        return self._finalize_global_map(canvas_bgr, canvas_mask)

    def _build_simple(self):
        min_x = min(t[0] for t in self.tiles)
        min_y = min(t[1] for t in self.tiles)
        max_x = max(t[0] + t[2].shape[1] for t in self.tiles)
        max_y = max(t[1] + t[2].shape[0] for t in self.tiles)
        W, H = max_x - min_x, max_y - min_y

        canvas_bgr  = np.zeros((H, W, 3), dtype=np.float32)
        canvas_mask = np.zeros((H, W),    dtype=np.float32)
        weight_map  = np.zeros((H, W),    dtype=np.float32)

        for gx, gy, frame, mask in self.tiles:
            x, y = gx - min_x, gy - min_y
            h, w = frame.shape[:2]
            paste_h = min(h, H - y)
            paste_w = min(w, W - x)
            if paste_h <= 0 or paste_w <= 0:
                continue
            frame_float = frame[:paste_h, :paste_w].astype(np.float32)
            mask_float  = mask[:paste_h,  :paste_w].astype(np.float32)
            dist_x    = np.minimum(np.arange(paste_w), paste_w - 1 - np.arange(paste_w))
            dist_y    = np.minimum(np.arange(paste_h), paste_h - 1 - np.arange(paste_h))
            dist_grid = np.minimum.outer(dist_y, dist_x)
            weight    = np.exp(-dist_grid / 50.0)
            overlap   = (canvas_mask[y:y+paste_h, x:x+paste_w] > 0)
            if np.any(overlap):
                ew  = weight_map[y:y+paste_h, x:x+paste_w]
                tw  = ew + weight
                ne  = ew / (tw + 1e-6)
                nn  = weight / (tw + 1e-6)
                canvas_bgr [y:y+paste_h, x:x+paste_w] = (
                    canvas_bgr[y:y+paste_h, x:x+paste_w] * ne[:, :, None] + frame_float * nn[:, :, None])
                canvas_mask[y:y+paste_h, x:x+paste_w] = (
                    canvas_mask[y:y+paste_h, x:x+paste_w] * ne + mask_float * nn)
                weight_map [y:y+paste_h, x:x+paste_w] = tw
            else:
                canvas_bgr [y:y+paste_h, x:x+paste_w] = frame_float
                canvas_mask[y:y+paste_h, x:x+paste_w] = mask_float
                weight_map [y:y+paste_h, x:x+paste_w] = weight

        canvas_bgr  = np.clip(canvas_bgr,  0, 255).astype(np.uint8)
        canvas_mask = np.clip(canvas_mask, 0, 255).astype(np.uint8)
        return self._finalize_global_map(canvas_bgr, canvas_mask)

    def _finalize_global_map(self, canvas_bgr: np.ndarray, canvas_mask: np.ndarray):
        """
        完成全局地图后处理，同时生成干净主路径（无环无分支）和切割路径。
        返回 10 元组（比原来多 cutting_path_mask、cutting_path_points、cutting_directions 三项）。
        """
        logger.info("  🔍 提取完整骨架网络...")
        full_skeleton, longest_path_mask, longest_path_points = \
            self.skeleton_extractor.extract_from_mask(canvas_mask)

        # ★ 提取干净主路径
        logger.info("  🔍 提取干净主路径（去除环路与分支）...")
        clean_path_mask, clean_path_points = \
            self.skeleton_extractor.extract_clean_main_path(
                canvas_mask, precomputed_skel=full_skeleton)
        logger.info(
            f"  ✓ 干净路径: {len(clean_path_points)} px  "
            f"(原始骨架 {int(np.sum(full_skeleton > 0))} px)"
        )

        # ★ 提取切割路径（从左上角开始）
        logger.info("  🔍 提取切割路径（从左上角开始，一刀切到底）...")
        cutting_path_mask, cutting_path_points, cutting_directions = \
            self.skeleton_extractor.extract_cutting_path(
                canvas_mask, precomputed_skel=full_skeleton)
        logger.info(
            f"  ✓ 切割路径: {len(cutting_path_points)} px  "
            f"(方向标注: {len(cutting_directions)} 个)"
        )

        canvas_mask_color    = self.colorize_mask(canvas_mask)
        skeleton_pixel_count = int(np.sum(full_skeleton > 0))

        if skeleton_pixel_count > 0:
            logger.info(
                f"  ✓ 完整骨架 {skeleton_pixel_count} px，"
                f"最长路径 {len(longest_path_points)} px"
            )
            # 灰色：完整骨架
            global_map_with_skeleton = self.skeleton_extractor.draw_all_skeleton(
                canvas_bgr, full_skeleton,
                color=(128, 128, 128), radius=self.cfg.skeleton_path_radius)
            # 红色：最长路径（原样保留）
            if len(longest_path_points) > 0:
                global_map_with_skeleton = self.skeleton_extractor.draw_all_skeleton(
                    global_map_with_skeleton, longest_path_mask,
                    color=(0, 0, 255), radius=self.cfg.skeleton_highlight_radius)
            # 黄色：干净主路径
            if len(clean_path_points) > 0:
                global_map_with_skeleton = self.skeleton_extractor.draw_all_skeleton(
                    global_map_with_skeleton, clean_path_mask,
                    color=(0, 255, 255), radius=self.cfg.skeleton_highlight_radius)
            # 青色：切割路径（带方向箭头）
            if len(cutting_path_points) > 0:
                global_map_with_skeleton = self.skeleton_extractor.draw_cutting_path_with_arrows(
                    global_map_with_skeleton, cutting_path_mask, cutting_directions,
                    path_color=(255, 0, 255), arrow_color=(0, 255, 0),
                    arrow_spacing=30, arrow_size=15)

            global_mask_color_with_skeleton = self.skeleton_extractor.draw_all_skeleton(
                canvas_mask_color, full_skeleton,
                color=(0, 255, 255), radius=self.cfg.skeleton_path_radius)
        else:
            logger.warning("  ⚠ 未找到有效骨架")
            global_map_with_skeleton        = canvas_bgr.copy()
            global_mask_color_with_skeleton = canvas_mask_color.copy()

        return (global_map_with_skeleton, canvas_mask, global_mask_color_with_skeleton,
                full_skeleton, longest_path_mask, longest_path_points,
                clean_path_mask, clean_path_points,
                cutting_path_mask, cutting_path_points, cutting_directions)

    @staticmethod
    def colorize_mask(mask: np.ndarray) -> np.ndarray:
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in SEG_PALETTE.items():
            vis[mask == cls] = color
        return vis


# ============================================================
# 6. 运动日志
# ============================================================
@dataclass
class MotionRecord:
    direction: str
    steps: int
    timestamp: float

@dataclass
class MotionLogger:
    records: List[MotionRecord] = field(default_factory=list)

    def log(self, direction: str, steps: int):
        self.records.append(MotionRecord(direction, steps, time.time()))
        logger.info(f"  ▶ {direction:>5s}  {steps:>6d} steps")

    @property
    def total_x_steps(self) -> int:
        return sum(
            r.steps if r.direction == "RIGHT" else (-r.steps if r.direction == "LEFT" else 0)
            for r in self.records)

    @property
    def total_y_steps(self) -> int:
        return sum(
            r.steps if r.direction == "DOWN" else (-r.steps if r.direction == "UP" else 0)
            for r in self.records)

    def summary(self) -> str:
        lines = ["=" * 60, "驱动器运动总结", "=" * 60]
        for i, r in enumerate(self.records):
            lines.append(f"  [{i+1:>3d}] {r.direction:>5s}  {r.steps:>6d} steps")
        lines += [
            "-" * 60,
            f"  净水平位移 (Net X): {self.total_x_steps:+d} steps",
            f"  净垂直位移 (Net Y): {self.total_y_steps:+d} steps",
            f"  总动作数:           {len(self.records)}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ============================================================
# 7. 过程视频录制器
# ============================================================
class ScanVideoRecorder:
    def __init__(self, cfg: ScanConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        self.video_path = str(Path(cfg.output_dir) / "scan_process.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.video_path, fourcc, cfg.video_fps, (cfg.frame_w, cfg.frame_h))

        self.video_path_clean = str(Path(cfg.output_dir) / "scan_process_clean.mp4")
        self.writer_clean = cv2.VideoWriter(
            self.video_path_clean, fourcc, cfg.video_fps, (cfg.frame_w, cfg.frame_h))

        self.frame_count = 0
        self.debug_dir = Path(cfg.output_dir) / "debug_frames" if cfg.save_debug_frames else None
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        s = cfg.corner_roi_size
        w, h = cfg.frame_w, cfg.frame_h
        self.corner_rects = [
            (0, 0, s, s, "TL"), (w-s, 0, w, s, "TR"),
            (0, h-s, s, h, "BL"), (w-s, h-s, w, h, "BR"),
        ]
        self.skeleton_extractor = PathSkeletonExtractor(cfg)

    def write_frame(self, frame_bgr: np.ndarray, mask: np.ndarray,
                    row: int, direction: str, step_x: int, step_y: int,
                    boundary_info: str = "", draw_skeleton: bool = True):
        overlay = frame_bgr.copy()
        for cls, color in SEG_PALETTE.items():
            if cls == SegClass.BACKGROUND:
                continue
            overlay[mask == cls] = color
        blended = cv2.addWeighted(frame_bgr, 0.6, overlay, 0.4, 0)

        self.writer_clean.write(blended)

        if draw_skeleton:
            try:
                full_skeleton, longest_path, _ = self.skeleton_extractor.extract_from_mask(mask)
                skeleton_count = np.sum(full_skeleton > 0)
                if skeleton_count > 0:
                    blended = self.skeleton_extractor.draw_all_skeleton(
                        blended, full_skeleton, color=(128, 128, 128), radius=1)
                    blended = self.skeleton_extractor.draw_all_skeleton(
                        blended, longest_path, color=(0, 0, 255), radius=2)
                    cv2.putText(blended, f"Skeleton:{skeleton_count}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            except Exception as e:
                logger.debug(f"实时骨架提取失败: {e}")

        for x1, y1, x2, y2, label in self.corner_rects:
            color = (0, 255, 255)
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
            label_pos = (x1 + 5, y1 + 20) if label in ["TL", "BL"] else (x1 - 45, y1 + 20)
            cv2.putText(blended, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        line1 = f"Row:{row} Dir:{direction} X:{step_x} Y:{step_y}"
        cv2.putText(blended, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if boundary_info:
            cv2.putText(blended, boundary_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        self.writer.write(blended)
        if self.debug_dir and self.frame_count % 10 == 0:
            cv2.imwrite(str(self.debug_dir / f"frame_{self.frame_count:05d}.jpg"), blended)
        self.frame_count += 1

    def release(self):
        self.writer.release()
        self.writer_clean.release()
        logger.info(f"视频已保存: {self.video_path}  ({self.frame_count} frames)")
        logger.info(f"纯分割视频已保存: {self.video_path_clean}  ({self.frame_count} frames)")


# ============================================================
# 8. 蛇形扫描控制器
# ============================================================
class SnakeScanController:
    def __init__(self, cfg: ScanConfig, hw: HardwareInterface):
        self.cfg = cfg
        self.hw = hw
        self.seg = SegmentationInference()
        self.bd = GeometricCornerDetector(cfg)
        self.stitcher = GlobalMapStitcher(cfg)
        self.motion = MotionLogger()
        self.recorder = ScanVideoRecorder(cfg)
        self.cur_x_steps = 0
        self.cur_y_steps = 0
        self.current_row = 0

    def _capture_and_segment(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.hw.grab_frame_and_mask(self.seg)

    def _record(self, frame, mask, direction, draw_skeleton=True):
        bd_info = self.bd.status_str(mask)
        self.stitcher.record_tile(frame, mask)
        self.recorder.write_frame(
            frame, mask, self.current_row, direction,
            self.cur_x_steps, self.cur_y_steps, bd_info,
            draw_skeleton=draw_skeleton)

    def _move_right(self, steps: int):
        self.hw.move_x(+steps)
        self.cur_x_steps += steps
        self.stitcher.update_position(int(steps * self.cfg.pixels_per_step), 0)
        self.motion.log("RIGHT", steps)
        self.bd.update_position(self.cur_x_steps, self.cur_y_steps)

    def _move_left(self, steps: int):
        self.hw.move_x(-steps)
        self.cur_x_steps -= steps
        self.stitcher.update_position(-int(steps * self.cfg.pixels_per_step), 0)
        self.motion.log("LEFT", steps)
        self.bd.update_position(self.cur_x_steps, self.cur_y_steps)

    def _move_down(self, steps: int):
        self.hw.move_y(+steps)
        self.cur_y_steps += steps
        self.stitcher.update_position(0, int(steps * self.cfg.pixels_per_step))
        self.motion.log("DOWN", steps)
        self.bd.update_position(self.cur_x_steps, self.cur_y_steps)

    def _move_up(self, steps: int):
        self.hw.move_y(-steps)
        self.cur_y_steps -= steps
        self.stitcher.update_position(0, -int(steps * self.cfg.pixels_per_step))
        self.motion.log("UP", steps)
        self.bd.update_position(self.cur_x_steps, self.cur_y_steps)

    def _init_to_top_left(self):
        logger.info(">>> Step 2: 初始化到左上角")
        step = self.cfg.move_speed_steps
        self.bd.reset_direction("top")
        for i in range(10000):
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "INIT_UP", draw_skeleton=False)
            if i % 20 == 0:
                logger.info(f"  寻找上边界... iter={i}  {self.bd.status_str(mask)}")
            if self.bd.has_top_boundary(mask):
                logger.info(f"  ✓ 找到上边界 (第 {i} 步)")
                break
            self._move_up(step)
        else:
            logger.warning("未找到上边界!")

        self.bd.reset_direction("left")
        for i in range(10000):
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "INIT_LEFT", draw_skeleton=False)
            if i % 20 == 0:
                logger.info(f"  寻找左边界... iter={i}  {self.bd.status_str(mask)}")
            if self.bd.has_left_boundary(mask):
                logger.info(f"  ✓ 找到左边界 (第 {i} 步)")
                break
            self._move_left(step)
        else:
            logger.warning("未找到左边界!")

        self._record(frame, mask, "INIT_DONE", draw_skeleton=True)
        logger.info(f"  左上角校准完成  位置: x={self.cur_x_steps}, y={self.cur_y_steps}")

    def _scan_right_until_boundary(self):
        step = self.cfg.move_speed_steps
        self.bd.reset_direction("right")
        logger.info(f"  → 向右扫描 (Row {self.current_row})")
        for i in range(100000):
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "RIGHT")
            if self.bd.has_right_boundary(mask):
                logger.info(f"    ✓ 到达右边界 (iter {i})  x={self.cur_x_steps}")
                return
            self._move_right(step)
        logger.warning("向右扫描超过最大迭代次数!")

    def _scan_left_until_boundary(self):
        step = self.cfg.move_speed_steps
        self.bd.reset_direction("left")
        logger.info(f"  ← 向左扫描 (Row {self.current_row})")
        for i in range(100000):
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "LEFT")
            if self.bd.has_left_boundary(mask):
                logger.info(f"    ✓ 到达左边界 (iter {i})  x={self.cur_x_steps}")
                return
            self._move_left(step)
        logger.warning("向左扫描超过最大迭代次数!")

    def _move_down_one_frame(self) -> bool:
        total_steps = self.cfg.frame_h_steps
        step_chunk  = self.cfg.move_speed_steps
        remaining   = total_steps
        logger.info(f"  ↓ 下移 {total_steps} 步 (含 {self.cfg.row_overlap_ratio:.0%} 重叠)")
        while remaining > 0:
            move_now = min(step_chunk, remaining)
            self._move_down(move_now)
            remaining -= move_now
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "DOWN")
        self.bd.reset_direction("bottom")
        for _ in range(self.cfg.corner_confirm_count + 1):
            frame, mask = self._capture_and_segment()
            self._record(frame, mask, "DOWN_CHK")
            if self.bd.has_bottom_boundary(mask):
                logger.info(f"    ✓ 检测到下边界 y={self.cur_y_steps}")
                return False
        return True

    def run(self):
        logger.info("=" * 60)
        logger.info("蛇形扫描启动 v6.5 (干净主路径去环去分支)")
        logger.info(f"  帧尺寸: {self.cfg.frame_w} x {self.cfg.frame_h}")
        logger.info("=" * 60)
        self._init_to_top_left()
        self._scan_right_until_boundary()
        while True:
            can_continue = self._move_down_one_frame()
            if not can_continue:
                break
            self.current_row += 1
            if self.current_row % 2 == 1:
                self._scan_left_until_boundary()
            else:
                self._scan_right_until_boundary()
        logger.info("=" * 60)
        logger.info(f"扫描完成! 共 {self.current_row + 1} 行")
        logger.info("=" * 60)
        self._save_results()

    def _save_results(self):
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary = self.motion.summary()
        print(summary)
        (out / "motion_log.txt").write_text(summary, encoding="utf-8")

        logger.info(">>> 构建全局地图并提取骨架...")
        # ★ 解包 8 个返回值（新增 clean_path_mask、clean_path_points）
        (global_map_with_skeleton, global_mask, global_mask_color_with_skeleton,
         full_skeleton, longest_path_mask, longest_path_points,
         clean_path_mask, clean_path_points) = self.stitcher.build_global_map()

        # ── 原有输出（保持不变）──────────────────────────────
        cv2.imwrite(str(out / "global_map.png"), global_map_with_skeleton)
        logger.info(f"  ✓ global_map.png  ({global_map_with_skeleton.shape[1]}x{global_map_with_skeleton.shape[0]})")

        cv2.imwrite(str(out / "global_mask.png"), global_mask)
        logger.info(f"  ✓ global_mask.png")

        cv2.imwrite(str(out / "global_mask_color.png"), global_mask_color_with_skeleton)
        logger.info(f"  ✓ global_mask_color.png")

        if np.sum(full_skeleton) > 0:
            cv2.imwrite(str(out / "global_skeleton_all.png"),
                        (full_skeleton * 255).astype(np.uint8))
            logger.info(f"  ✓ global_skeleton_all.png (完整骨架)")
            cv2.imwrite(str(out / "global_skeleton_main.png"),
                        (longest_path_mask * 255).astype(np.uint8))
            logger.info(f"  ✓ global_skeleton_main.png (最长路径，含环/分支，原样保留)")

        if len(longest_path_points) > 0:
            path_data = {
                "full_skeleton_pixels": int(np.sum(full_skeleton > 0)),
                "longest_path_length": len(longest_path_points),
                "longest_path_points": [{"x": int(x), "y": int(y)} for (y, x) in longest_path_points]
            }
            (out / "global_path.json").write_text(
                json.dumps(path_data, indent=2), encoding="utf-8")
            logger.info(f"  ✓ global_path.json")

        # ── ★ 新增：干净主路径输出 ────────────────────────────
        if np.sum(clean_path_mask) > 0:
            cv2.imwrite(str(out / "global_clean_path.png"),
                        (clean_path_mask * 255).astype(np.uint8))
            logger.info(f"  ✓ global_clean_path.png（干净主路径，无环无分支）")

            clean_overlay = self.stitcher.skeleton_extractor.draw_all_skeleton(
                global_map_with_skeleton.copy(), clean_path_mask,
                color=(0, 255, 255), radius=self.cfg.skeleton_highlight_radius)
            cv2.imwrite(str(out / "global_clean_path_overlay.png"), clean_overlay)
            logger.info(f"  ✓ global_clean_path_overlay.png（黄色叠加可视化）")

            clean_data = {
                "clean_path_pixel_count": len(clean_path_points),
                "clean_path_points": [
                    {"x": int(x), "y": int(y)} for (y, x) in clean_path_points
                ],
            }
            (out / "global_clean_path.json").write_text(
                json.dumps(clean_data, indent=2), encoding="utf-8")
            logger.info(f"  ✓ global_clean_path.json")

        self.recorder.release()

        steps_data = {
            "scan_config": {
                "frame_size": f"{self.cfg.frame_w}x{self.cfg.frame_h}",
                "corner_roi_size": self.cfg.corner_roi_size,
                "corner_threshold": self.cfg.corner_ratio_threshold,
                "overlap_ratio": self.cfg.row_overlap_ratio,
                "corner_confirm": self.cfg.corner_confirm_count,
                "lock_margin": self.cfg.boundary_lock_margin_steps,
                "skeleton_radius": self.cfg.skeleton_path_radius,
                "skeleton_min_length": self.cfg.skeleton_min_length,
            },
            "physical_corners": self.bd.get_status()["physical_corners"],
            "inferred_boundaries": self.bd.get_status()["inferred_boundaries"],
            "skeleton_info": {
                "total_pixels": int(np.sum(full_skeleton > 0)),
                "longest_path_length": len(longest_path_points),
                "clean_path_length": len(clean_path_points),
            },
            "records": [{"direction": r.direction, "steps": r.steps} for r in self.motion.records],
            "net_x_steps": self.motion.total_x_steps,
            "net_y_steps": self.motion.total_y_steps,
            "total_rows": self.current_row + 1,
        }
        (out / "motion_steps.json").write_text(
            json.dumps(steps_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"所有结果已保存到: {out.resolve()}")


# ============================================================
# 9. 模拟硬件
# ============================================================
class SimulatedHardware(HardwareInterface):
    def __init__(self, cfg: ScanConfig, full_map_path: str):
        super().__init__(cfg)
        self.full_map = cv2.imread(full_map_path)
        if self.full_map is None:
            raise FileNotFoundError(f"找不到模拟地图: {full_map_path}")
        self.map_h, self.map_w = self.full_map.shape[:2]
        self.pos_x = self.map_w // 2
        self.pos_y = self.map_h // 2
        logger.info(f"模拟模式: 地图 {self.map_w}x{self.map_h}  初始位置: ({self.pos_x}, {self.pos_y})")

    def _compute_oob_mask(self) -> np.ndarray:
        h, w = self.cfg.frame_h, self.cfg.frame_w
        oob = np.zeros((h, w), dtype=bool)
        vx1, vy1 = self.pos_x, self.pos_y
        vx2, vy2 = self.pos_x + w, self.pos_y + h
        if vy1 < 0: oob[:min(h, -vy1), :] = True
        if vy2 > self.map_h: oob[max(0, h-(vy2-self.map_h)):, :] = True
        if vx1 < 0: oob[:, :min(w, -vx1)] = True
        if vx2 > self.map_w: oob[:, max(0, w-(vx2-self.map_w)):] = True
        return oob

    def grab_frame(self) -> np.ndarray:
        h, w = self.cfg.frame_h, self.cfg.frame_w
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        sx1, sy1 = self.pos_x, self.pos_y
        sx2, sy2 = self.pos_x + w, self.pos_y + h
        vx1 = max(0, sx1); vy1 = max(0, sy1)
        vx2 = min(self.map_w, sx2); vy2 = min(self.map_h, sy2)
        if vx2 > vx1 and vy2 > vy1:
            dx1, dy1 = vx1 - sx1, vy1 - sy1
            canvas[dy1:dy1+(vy2-vy1), dx1:dx1+(vx2-vx1)] = \
                self.full_map[vy1:vy2, vx1:vx2]
        return canvas

    def grab_frame_and_mask(self, seg_model: SegmentationInference) -> Tuple[np.ndarray, np.ndarray]:
        frame = self.grab_frame()
        mask = seg_model.predict(frame)
        mask[self._compute_oob_mask()] = SegClass.BOUNDARY
        return frame, mask

    def move_x(self, steps: int):
        self.pos_x += int(steps * self.cfg.pixels_per_step)

    def move_y(self, steps: int):
        self.pos_y += int(steps * self.cfg.pixels_per_step)


# ============================================================
# 主程序
# ============================================================
def main():
    cfg = ScanConfig(
        frame_w=1028, frame_h=776,
        um_per_pixel=1.0, um_per_step=1.0,
        move_speed_steps=100, row_overlap_ratio=0.2,
        corner_roi_size=80, corner_ratio_threshold=0.25,
        corner_confirm_count=4, boundary_lock_margin_steps=10,
        skeleton_path_radius=1, skeleton_highlight_radius=3,
        skeleton_min_length=100000,
        output_dir=r"F:/0_My_jupyter/0_Optical/scan_output/Small",
        video_fps=10, save_debug_frames=True,
    )

    USE_SIMULATION = True

    if USE_SIMULATION:
        hw = SimulatedHardware(
            cfg,
            full_map_path=r"F:\0_My_jupyter\0_Optical\scan_output\Small\map_small2.jpg",
        )
    else:
        hw = HardwareInterface(cfg)

    scanner = SnakeScanController(cfg, hw)
    scanner.run()


if __name__ == "__main__":
    main()
