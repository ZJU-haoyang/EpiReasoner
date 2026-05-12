# file: core/visual_explainer.py
# -*- coding: utf-8 -*-

"""
Phenotype Visual Explainer (Pro Version)
----------------------------------------
Version: 2.3.0 (Enhanced with Betweenness Centrality)
Description: 
    表型可视化解释器。
    该模块旨在通过几何图元（Geometric Primitives）直观展示高维表型指标的计算原理。
    它将抽象的数学特征（如形状熵、拓扑不变量、力学张量代理）转化为
    可被人类理解的视觉证据（Visual Evidence）。

Features:
    - Basic: Contour, Solidity, Circularity, Feret Diameter
    - Advanced: Skeleton (Topological Axis), Convexity Defects (Lobes)
    - Physics-based: Max Inscribed Circle (Mechanical Stress)
    - Network Theory: Visibility Graph, Closeness (Lobes), Betweenness (Bottlenecks) [New]
    - TDA: Euler Characteristic Scan (Topology)
    - [New P0] Curvature heatmap, Thickness heatmap, Multi-angle ECT
    - [New P1] Multi-scale lobes, Mechanical stress heatmap

Citations:
    - Visibility Graph for Shape Analysis: Vetter et al. (2019)
    - Betweenness Centrality in Spatial Networks: Barthelemy (2011)
"""

import numpy as np
import cv2
from scipy.spatial import ConvexHull, distance_matrix
from scipy.ndimage import distance_transform_edt
from skimage import morphology
import warnings
import matplotlib.cm as cm  
import scipy.ndimage as ndimage

# Optional: Shapely for robust geometric operations in Visibility Graph
try:
    from shapely.geometry import Polygon, LineString, Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    print("Warning: Shapely not found. Visibility Graph visualization will be approximated.")

# Optional: NetworkX for centrality calculation
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not found. Centrality visualization will be skipped.")

class VisualExplainer:
    """
    表型可视化解释器 (Phenotype Visual Explainer)
    
    设计理念：
    "Show, don't just tell." —— 为每个计算出的表型指标提供几何证明。
    """
    
    def __init__(self, scale_factor=1.0):
        """
        Args:
            scale_factor (float): 像素/物理单位转换系数，用于输出带有物理单位的标注。
        """
        self.scale_factor = scale_factor

    def get_contour(self, binary_mask):
        """获取用于OpenCV处理的最大连通域轮廓"""
        mask_u8 = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=len)
    
    def _get_insight(self, metric_type):
        """
        获取植物学/生物物理学见解文本库 (Scientific Insights Library)
        
        References:
        - Sapala et al. (2018) eLife: Mechanical stress in pavement cells.
        - Carter et al. (2017) Current Biology: Stomatal patterning.
        - Zahn & Roskies (1972): Fourier descriptors.
        """
        insights = {
            # --- 1. 基础形态学 (Basic Morphometrics) ---
            'Solidity': (
                "📉 [Solidity] Ratio of Area to Convex Hull.\n"
                "• Biology: Indicates the degree of interdigitation in pavement cells.\n"
                "• High (≈1): Simple shapes (e.g., young cells, stomata).\n"
                "• Low (<0.8): Highly lobed cells, suggesting active ROP-GTPase signaling."
            ),
            'Circularity': (
                "🔄 [Circularity] 4π × Area / Perimeter².\n"
                "• Biology: A measure of compactness. Stomata are typically high (>0.9).\n"
                "• Pavement Cells: Decreases as cells expand and form lobes to reduce mechanical stress."
            ),
            'Aspect Ratio': (
                "📏 [Aspect Ratio] Major Axis / Minor Axis.\n"
                "• Biology: Indicates cell elongation direction.\n"
                "• Significance: Oriented elongation often aligns with tissue growth axes or microtubule arrays."
            ),
            'Feret Diameter': (
                "📏 [Feret Diameter] The 'Max Caliper' distance.\n"
                "• Biology: Defines the maximum spatial extent of a cell.\n"
                "• Utility: More robust than 'Major Axis' for irregular, crescent-shaped guard cells."
            ),
            
            # --- 2. 边缘复杂性与拓扑 (Complexity & Topology) ---
            'Undulation Index': (
                "〰️ [Undulation Index] Perimeter / Circumference of eq. circle.\n"
                "• Biology: Directly quantifies the 'waviness' of cell walls.\n"
                "• Function: Enhances mechanical adhesion between cells, preventing tissue tearing under turgor pressure."
            ),
            'Lobeyness': (
                "🧩 [Lobeyness] Pavement Cell Shape Factor.\n"
                "• Biology: Characterizes the puzzle-piece shape.\n"
                "• Mechanism: Driven by localized auxin accumulation and actin cytoskeleton reorganization."
            ),
            'Lobe Count': (
                "🔢 [Lobe Count] Number of protrusions.\n"
                "• Biology: Correlates with cell ploidy levels (DNA content) and developmental stage in many species."
            ),
            'Skeleton': (
                "🦴 [Skeleton] Topological Medial Axis.\n"
                "• Biology: Represents the true 'growth path' or branching structure of complex cells better than simple width/length."
            ),
            'Convexity Defects (Lobes)': (
                "🧬 [Convexity Defects] Deep invaginations (Necks).\n"
                "• Mechanics: These are regions of high mechanical stress accumulation, often reinforced by microtubule bundles."
            ),

            # --- 3. 空间分布与生态 (Spatial Ecology) ---
            'Voronoi Entropy': (
                "🎲 [Voronoi Entropy] Disorder of spatial arrangement.\n"
                "• Biology: Measures the regularity of stomatal spacing.\n"
                "• Insight: Low entropy implies a strict 'one-cell-spacing' rule; High entropy suggests random distribution or clustering mutants."
            ),
            'Ripley L_max': (
                "📍 [Ripley's L] Clustering Deviation.\n"
                "• Ecology: Quantifies deviation from Random (CSR).\n"
                "• Positive: Clustered (e.g., clustering mutants).\n"
                "• Negative: Dispersed/Regular (typical wild-type stomata)."
            ),
            'LEC Coverage': (
                "💥 [LEC Coverage] Largest Empty Circle / Area.\n"
                "• Mechanics: Proxy for maximum cell wall stress.\n"
                "• Insight: Large open areas (high LEC) are mechanically vulnerable; lobes reduce this value."
            ),

            # --- 4. 网络与图论 (Network Theory) ---
            'Visibility Graph': (
                "🕸️ [Visibility Graph] Shape connectivity.\n"
                "• Math: Maps shape boundary to a network.\n"
                "• Insight: Dense networks correlate with fractal boundaries and high surface-area-to-volume ratios."
            ),
            'VG Betweenness': (
                "🚦 [Betweenness Centrality] Network bottlenecks.\n"
                "• Insight: Highlights 'Neck' regions that act as transport bottlenecks or structural hinges in the cell shape."
            ),
            'VG Closeness': (
                "🎯 [Closeness Centrality] Global visibility.\n"
                "• Insight: Identifies 'Lobe Tips' (high values) which have the most direct line-of-sight to the rest of the cell."
            ),

            # --- 5. 新指标 (New Metrics) ---
            'Curvature Mean': (
                "↩️ [Curvature] Rate of direction change.\n"
                "• Biology: High curvature peaks mark sites of active lobe initiation or outgrowth."
            ),
            'Thickness Mean': (
                "↔️ [Thickness] Local width.\n"
                "• Biology: Uniform thickness is crucial for guard cell function; variation indicates pavement cell heterogeneity."
            ),
            'ECT Complexity': (
                "📶 [ECT] Euler Characteristic Transform.\n"
                "• Math: A topological fingerprint.\n"
                "• Utility: Captures subtle shape features lost by scalar metrics, useful for distinguishing mutant phenotypes."
            )
        }
        
        # 处理别名查找
        key_map = {
            "Lobe Count (Basic)": "Lobe Count",
            "Lobe Count (Multi-Scale)": "Lobe Count",
            "VG Lobe Count": "Lobe Count",
            "Convexity": "Solidity", # Similar concept
            "Mechanical Stress Map": "LEC Coverage", # Related
            "Curvature Heatmap": "Curvature Mean",
            "Thickness Heatmap": "Thickness Mean",
            "Centrality Profile": "VG Closeness",
            "Visibility Graph Betweenness": "VG Betweenness",
            "Multi-Scale Lobes": "Convexity Defects (Lobes)"
        }
        
        search_key = key_map.get(metric_type, metric_type)
        return insights.get(search_key, f"🔍 Visualizing {metric_type} geometry.")
    # ==================== 基础形态学 (Basic Morphometrics) ====================

    def explain_solidity(self, binary_mask):
        """
        解释实心度 (Solidity)
        原理：可视化对象与其凸包 (Convex Hull) 的差异。
        凸包类似于"橡皮筋"包裹的效果，凹陷区域即为非实心区域。
        """
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        
        hull = cv2.convexHull(cnt)
        
        return {
            'type': 'polygon_comparison',
            'contour': cnt.reshape(-1, 2), # 原始形状
            'hull': hull.reshape(-1, 2),   # 凸包形状
            'label': 'Solidity: Object Area vs Convex Area',
            'insight': self._get_insight('Solidity')
        }

    def explain_circularity(self, binary_mask):
        """
        解释圆度 (Circularity) 与 波动指数 (Undulation)
        原理：叠加一个与对象"等面积"的完美圆。
        铺板细胞边缘超出该圆的部分越长、越复杂，圆度越低，波动指数越高。
        """
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        
        M = cv2.moments(cnt)
        if M['m00'] == 0: return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        area = cv2.contourArea(cnt)
        radius = np.sqrt(area / np.pi)
        
        return {
            'type': 'circle_overlay',
            'center': (cx, cy),
            'radius': radius,
            'contour': cnt.reshape(-1, 2),
            'label': 'Circularity: Deviation from Equivalent Circle',
            'insight': self._get_insight('Circularity')
        }

    def explain_feret_diameter(self, binary_mask):
        """
        解释 Feret 直径 (Max Caliper Diameter)
        原理：基于旋转卡尺法或凸包顶点距离，找到轮廓上欧氏距离最远的两点。
        这是植物学中定义"细胞长度"的标准方法。
        """
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        
        hull = cv2.convexHull(cnt).reshape(-1, 2)
        if len(hull) < 2: return None
        
        dists = distance_matrix(hull, hull)
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        
        p1, p2 = hull[i], hull[j]
        max_dist = dists[i, j]
        
        return {
            'type': 'line',
            'start': p1, 
            'end': p2,   
            'value': max_dist / self.scale_factor,
            'label': 'Feret Diameter (Max Caliper)',
            'insight': self._get_insight('Feret Diameter')
        }

    # ==================== 复杂形态与拓扑 (Complexity & Topology) ====================

    def explain_visibility_graph(self, binary_mask, num_nodes=50):
        """
        解释可见性图 (Visibility Graph)
        """
        if not HAS_SHAPELY:
            return None

        cnt = self.get_contour(binary_mask)
        if cnt is None: return None

        try:
            # 1. 轮廓重采样
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2)
            
            if len(points) > num_nodes:
                indices = np.linspace(0, len(points)-1, num_nodes, dtype=int)
                points = points[indices]
            
            poly_geom = Polygon(points)
            if not poly_geom.is_valid:
                poly_geom = poly_geom.buffer(0)
            
            valid_edges = []
            n = len(points)
            
            # 2. 构建可见边
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(i - j) == 1 or abs(i - j) == n - 1:
                        continue
                        
                    p1 = points[i]
                    p2 = points[j]
                    
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    mid_point = Point(mid_x, mid_y)
                    
                    if poly_geom.contains(mid_point):
                        valid_edges.append((p1, p2))
            
            return {
                'type': 'network_overlay',
                'nodes': points,
                'edges': valid_edges,
                'color': (0.0, 1.0, 1.0, 0.4), 
                'label': f'Visibility Graph (n={n})'
            }
        except Exception as e:
            print(f"VG Visual error: {e}")
            return None

    def explain_convexity_defects(self, binary_mask):
        """
        解释裂片与凹陷 (Lobes & Bays)
        """
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        if hull_indices is None or len(hull_indices) < 3: return None
        
        try:
            defects = cv2.convexityDefects(cnt, hull_indices)
            visual_points = []
            
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(cnt[f][0])
                    depth = d / 256.0
                    if depth > 2.0:
                        visual_points.append(far)
            
            return {
                'type': 'points',
                'points': visual_points,
                'hull_contour': self.get_contour(binary_mask).reshape(-1,2),
                'marker': 'x',
                'color': 'red',
                'label': 'Convexity Defects (Necks)',
                'insight': self._get_insight('Convexity Defects (Lobes)')
            }
        except:
            return None

    def explain_centrality_lobes(self, binary_mask, num_nodes=60):
        """
        可视化接近中心性 (Closeness Centrality)
        原理：接近中心性高的节点（红色）代表"Lobe Tips"（能看到整个细胞），
        低值区域通常是凹陷处。
        """
        if not HAS_SHAPELY or not HAS_NETWORKX:
            return None

        # 复用通用的图构建逻辑，避免代码重复
        G, points = self._build_visibility_graph_nx(binary_mask, num_nodes)
        if G is None: return None

        try:
            # 计算接近中心性
            closeness = nx.closeness_centrality(G)
            centrality_values = np.array([closeness[i] for i in range(len(points))])
            
            # 检测 Lobes (局部极大值 - 注意：这里修正了逻辑，Lobe tip 通常视野最好，Closeness高)
            # 或者在凹陷处 Closeness 可能受阻。通常 Lobe Tip 的 Closeness 较高。
            # 为了可视化，我们直接用颜色映射 Closeness 值。
            
            # 颜色映射：Viridis (Blue->Yellow)
            norm_values = centrality_values / np.max(centrality_values) if np.max(centrality_values) > 0 else centrality_values
            colormap = cm.get_cmap('viridis')
            colors = colormap(norm_values)[:, :3]

            return {
                'type': 'weighted_graph_nodes', # 前端需支持此类型：画点，颜色随值变化
                'points': points,
                'colors': colors,
                'values': centrality_values,
                'label': 'Closeness Centrality (Yellow=Global Vis.)'
            }
        except Exception as e:
            return None

    def explain_betweenness_bottlenecks(self, binary_mask, num_nodes=60):
        """
        [NEW] 可视化介数中心性 (Betweenness Centrality) 以识别形状瓶颈
        
        原理：
            介数中心性衡量一个节点作为其他两个节点之间最短路径桥梁的频率。
            在细胞形态的可见性图中，位于"颈部"（Neck）或狭窄连接处的节点，
            往往是连接两个较大区域（Lobes）的必经之路，因此具有极高的介数。
            
        生物学/物理学意义：
            1. 拓扑瓶颈：识别形状中最脆弱的连接点。
            2. 运输限制：模拟细胞质流动或信号分子扩散的拥堵点。
            3. 应力集中：通常与机械应力集中区域高度重合。
        """
        if not HAS_SHAPELY or not HAS_NETWORKX:
            return None

        G, points = self._build_visibility_graph_nx(binary_mask, num_nodes)
        if G is None: return None

        try:
            # 计算介数中心性 (normalized=True 使值在 0-1 之间)
            betweenness = nx.betweenness_centrality(G, normalized=True)
            bet_values = np.array([betweenness[i] for i in range(len(points))])
            
            # 归一化用于热图显示
            bet_norm = bet_values / np.max(bet_values) if np.max(bet_values) > 0 else bet_values
            
            # 使用 'magma' 色图 (Black->Purple->Orange->Yellow)
            # 黄色高亮显示"瓶颈"，黑色显示"Lobe Tips"（非桥梁）
            colormap = cm.get_cmap('magma')
            colors = colormap(bet_norm)[:, :3] # (N, 3) RGB

            # 识别显著的瓶颈点 (Top 10%)
            threshold = np.percentile(bet_values, 90) if len(bet_values) > 0 else 0
            bottleneck_indices = [i for i, v in enumerate(bet_values) if v >= threshold]
            bottleneck_points = points[bottleneck_indices]

            return {
                'type': 'weighted_graph_nodes',
                'points': points,
                'colors': colors,
                'values': bet_values,
                'highlight_points': bottleneck_points,
                'label': 'Betweenness Centrality (Yellow=Bottlenecks)',
                'insight': self._get_insight('Visibility Graph Betweenness')
            }

        except Exception as e:
            print(f"Betweenness visual error: {e}")
            return None

    def _build_visibility_graph_nx(self, binary_mask, num_nodes):
        """辅助函数：构建 NetworkX 可见性图"""
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None, None

        # 1. 重采样
        epsilon = 0.003 * cv2.arcLength(cnt, True) # 更精细的采样
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        points = approx.reshape(-1, 2)
        
        if len(points) > num_nodes:
            indices = np.linspace(0, len(points)-1, num_nodes, dtype=int)
            points = points[indices]
        
        poly_geom = Polygon(points)
        if not poly_geom.is_valid:
            poly_geom = poly_geom.buffer(0)

        # 2. 构建图
        G = nx.Graph()
        n = len(points)
        for i in range(n):
            G.add_node(i, pos=points[i])
        
        # 3. 连边
        for i in range(n):
            next_i = (i + 1) % n
            G.add_edge(i, next_i) 
            
            for j in range(i + 2, n):
                if i == 0 and j == n - 1: continue
                
                p1 = points[i]
                p2 = points[j]
                
                # 几何检测：连线中点是否在多边形内
                mid_point = Point((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                if poly_geom.contains(mid_point):
                        G.add_edge(i, j)
        return G, points

    def explain_ect_scan(self, binary_mask, angle=45):
        """
        解释欧拉特征变换 (ECT Scan)
        """
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        
        try:
            h, w = binary_mask.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_mask = cv2.warpAffine(binary_mask.astype(np.uint8), M, (w, h))
            
            scan_lines = []
            grid_spacing = 15
            
            y_idxs, x_idxs = np.where(rotated_mask > 0)
            if len(y_idxs) == 0: return None
            min_y, max_y = np.min(y_idxs), np.max(y_idxs)
            
            for y in range(min_y, max_y, grid_spacing):
                row = rotated_mask[y, :]
                padded = np.pad(row, (1, 1), mode='constant')
                diff = np.diff(padded.astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                for s, e in zip(starts, ends):
                    p1_rot = np.array([s, y])
                    p2_rot = np.array([e, y])
                    scan_lines.append((p1_rot, p2_rot))

            inv_M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            segments_original_space = []
            
            if scan_lines:
                points_rot = np.array(scan_lines).reshape(-1, 1, 2)
                points_original = cv2.transform(points_rot.astype(np.float32), inv_M)
                points_flat = points_original.reshape(-1, 2)
                for i in range(0, len(points_flat), 2):
                    segments_original_space.append((points_flat[i], points_flat[i+1]))

            return {
                'type': 'multi_line',
                'lines': segments_original_space,
                'color': (1.0, 0.65, 0.0, 0.8),
                'label': f'ECT Scan @ {angle}°'
            }
        except Exception as e:
            return None

    # ==================== 新增可视化：曲率热图 [P0] ====================

    def explain_curvature_heatmap(self, binary_mask):
        """可视化轮廓曲率分布（颜色映射）"""
        if cv2 is None: return None
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours: return None
            
            cnt = max(contours, key=len).squeeze()
            if len(cnt) < 5: return None
            
            curvatures = []
            for i in range(len(cnt)):
                p1 = cnt[i - 2]
                p2 = cnt[i - 1]
                p3 = cnt[i % len(cnt)]
                v1 = p2 - p1
                v2 = p3 - p2
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                curvatures.append(np.abs(angle))
            
            curv_norm = np.array(curvatures) / np.max(curvatures) if np.max(curvatures) > 0 else np.zeros_like(curvatures)
            colormap = cm.get_cmap('jet')
            colors = colormap(curv_norm)[:, :3]
            
            return {
                'type': 'colored_contour',
                'points': cnt,
                'colors': colors,
                'label': 'Curvature Heatmap (Red=High, Blue=Low)'
            }
        except: return None

    # ==================== 新增可视化：厚度热图 [P0] ====================

    def explain_thickness_heatmap(self, binary_mask):
        """可视化内部厚度分布（距离变换热图）"""
        try:
            dist_map = distance_transform_edt(binary_mask)
            dist_norm = dist_map / np.max(dist_map) if np.max(dist_map) > 0 else dist_map
            colormap = cm.get_cmap('viridis')
            heatmap = colormap(dist_norm)[:, :, :3]
            
            return {
                'type': 'heatmap_overlay',
                'heatmap': heatmap,
                'label': 'Thickness Heatmap (Yellow=Thick, Purple=Thin)'
            }
        except: return None

    # ==================== 新增可视化：多角度 ECT 指纹图 [P0] ====================

    def explain_full_ect_fingerprint(self, binary_mask, num_angles=36):
        """可视化多角度 ECT 矩阵作为热图"""
        if cv2 is None: return None
        try:
            ect_curves = []
            angles = np.linspace(0, 180, num_angles, endpoint=False)
            img_uint8 = binary_mask.astype(np.uint8)
            
            for angle in angles:
                M = cv2.getRotationMatrix2D((img_uint8.shape[1]/2, img_uint8.shape[0]/2), angle, 1)
                rotated = cv2.warpAffine(img_uint8, M, (img_uint8.shape[1], img_uint8.shape[0]), flags=cv2.INTER_NEAREST)
                ec_profile = []
                for row in rotated:
                    runs = np.count_nonzero(np.diff(np.concatenate(([0], row, [0]))) != 0) // 2
                    ec_profile.append(runs)
                ect_curves.append(ec_profile)
            
            max_len = max(len(p) for p in ect_curves)
            ect_matrix = np.array([p + [0] * (max_len - len(p)) for p in ect_curves])
            
            ect_norm = ect_matrix / np.max(ect_matrix) if np.max(ect_matrix) > 0 else ect_matrix
            colormap = cm.get_cmap('plasma')
            fingerprint = colormap(ect_norm)[:, :, :3]
            
            return {
                'type': 'fingerprint_heatmap',
                'matrix': fingerprint,
                'axes': {'x': 'Scan Position', 'y': 'Angle (deg)'},
                'label': 'ECT Fingerprint (High χ = Complex)'
            }
        except: return None

    # ==================== 新增可视化：多尺度裂片 [P1] ====================

    def explain_multi_scale_lobes(self, binary_mask, thresholds=[2, 4, 8]):
        """可视化多阈值裂片检测"""
        if cv2 is None: return None
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            
            cnt = max(contours, key=len)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3: return None
            
            defects = cv2.convexityDefects(cnt, hull_indices)
            lobe_points_multi = {thresh: [] for thresh in thresholds}
            
            if defects is not None:
                for thresh in thresholds:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth > thresh:
                            far = tuple(cnt[f][0])
                            lobe_points_multi[thresh].append(far)
            
            return {
                'type': 'multi_points',
                'points_dict': lobe_points_multi,
                'colors': {2: 'green', 4: 'yellow', 8: 'red'},
                'label': 'Multi-Scale Lobes (Green=Shallow, Red=Deep)',
                'insight': self._get_insight('Multi-Scale Lobes')
            }
        except: return None

    # ==================== 新增可视化：机械应力热图 [P1] ====================

    def explain_mechanical_stress_heatmap(self, binary_mask):
        """可视化机械应力代理热图（曲率 × 1/厚度）"""
        try:
            dist_map = distance_transform_edt(binary_mask)
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours: return None
            cnt = max(contours, key=len).squeeze()
            curvatures = []
            for i in range(len(cnt)):
                p1 = cnt[i - 2]
                p2 = cnt[i - 1]
                p3 = cnt[i % len(cnt)]
                v1 = p2 - p1
                v2 = p3 - p2
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                curvatures.append(np.abs(angle))
            
            stress_map = np.zeros_like(dist_map)
            for idx, point in enumerate(cnt):
                x, y = point
                thickness = dist_map[y, x] if dist_map[y, x] > 0 else 1
                stress = curvatures[idx] / thickness
                stress_map[y, x] = stress
            
            stress_map = ndimage.maximum_filter(stress_map, size=3)
            stress_norm = stress_map / np.max(stress_map) if np.max(stress_map) > 0 else stress_map
            colormap = cm.get_cmap('hot')
            heatmap = colormap(stress_norm)[:, :, :3]
            
            return {
                'type': 'heatmap_overlay',
                'heatmap': heatmap,
                'label': 'Stress Heatmap (Red=High Stress)',
                'insight': self._get_insight('Mechanical Stress Map')
            }
        except: return None

    # ==================== 力学与骨架 (Mechanics & Skeleton) ====================

    def explain_skeleton(self, binary_mask):
        """解释拓扑骨架"""
        try:
            skeleton = morphology.skeletonize(binary_mask)
            y_idxs, x_idxs = np.where(skeleton)
            if len(x_idxs) == 0: return None
            points = np.column_stack((x_idxs, y_idxs))
            
            return {
                'type': 'points_cloud',
                'points': points,
                'color': 'magenta',
                'marker_size': 1,
                'label': 'Skeleton: Topological Structure'
            }
        except: return None

    def explain_inscribed_circle(self, binary_mask):
        """解释最大内切圆 (LEC)"""
        try:
            dist_map = distance_transform_edt(binary_mask)
            max_dist = np.max(dist_map)
            if max_dist <= 0: return None
            cy, cx = np.unravel_index(np.argmax(dist_map), dist_map.shape)
            
            return {
                'type': 'circle_overlay',
                'center': (cx, cy),
                'radius': max_dist,
                'color': 'yellow',
                'linestyle': 'dashed',
                'label': 'Max Inscribed Circle (Stress Hotspot)',
                'insight': self._get_insight('Inscribed Circle')
            }
        except: return None

    def explain_rectangularity(self, binary_mask):
        """解释矩形度"""
        cnt = self.get_contour(binary_mask)
        if cnt is None: return None
        try:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            return {
                'type': 'polygon_overlay',
                'points': box,
                'color': 'cyan',
                'label': 'Min Area Rectangle'
            }
        except: return None