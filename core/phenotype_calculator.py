# file: core/phenotype_calculator.py
# -*- coding: utf-8 -*-

"""
Advanced Plant Phenotype Calculator (Pro Version)
-------------------------------------------------
Version: 3.3.0
Description: 
    这是一个用于植物表型组学的高级计算模块。它接受分割掩码（Segmentation Masks），
    并提取跨越多个维度的表型特征，旨在实现"尽可能完善"的表型解析。
    
    涵盖维度：
    1. 基础形态学 (Basic Morphometrics): 面积, 周长, 轴长, Feret直径
    2. 生理功能潜力 (Physiological Potential): g_smax (理论最大气孔导度)
    3. 空间生态分布 (Spatial Ecology): Voronoi 熵 (S_vor), 互作域
    4. 拓扑与形状复杂性 (Topology & Complexity): 可见性图 (Visibility Graph), 骨架, 波动指数
    5. 机械力学 (Mechanics): 最大内切圆覆盖率 (LEC Coverage)
    6. NEW: 扩展可见性图指标（节点度统计、接近中心性统计、lobes/necks检测）
    7. [New P0] 曲率分布统计、厚度分布统计、多角度ECT指纹
    8. [New P1] Ripley’s K/PCF、Lobe多尺度分割

Citations:
    - [cite_start]g_smax: Franks & Beerling (2009) [cite: 11]
    - Visibility Graph: Vetter et al. (2019)[cite_start], GraVis [cite: 104]
    - [cite_start]Voronoi Entropy: 空间热力学与自组织理论 [cite: 49]
    - [cite_start]LEC Coverage: 机械热点假说 [cite: 88, 91]
    - [New] Curvature: Zahn & Roskies (1972)
    - [New] Ripley’s K: Ripley (1977)
"""

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, distance_matrix, KDTree
from scipy import ndimage
from skimage import measure, morphology
from scipy.ndimage import distance_transform_edt
import math
import os
import warnings
import traceback
from scipy import signal  # [New P0] For peak finding in curvature

# Optional dependencies check
try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV (cv2) not found. Advanced shape analysis will be limited.")

try:
    import networkx as nx
except ImportError:
    nx = None
    print("Warning: NetworkX not found. Visibility Graph metrics will be skipped.")

try:
    from shapely.geometry import Polygon, LineString, Point
except ImportError:
    Polygon = None
    print("Warning: Shapely not found. Advanced topology metrics will be skipped.")

# Filter generic warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class PhenotypeCalculator:
    def __init__(self, scale_factor_px_per_um=1.0):
        """
        初始化表型计算器
        Args:
            scale_factor_px_per_um (float): 像素到微米的转换系数 (px/μm)
        """
        self.scale_factor = float(scale_factor_px_per_um)
        # pixel_ratio: 1个像素代表多少微米 (um/px)
        self.pixel_ratio = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0
        self.pixel_area_ratio = self.pixel_ratio ** 2
        
        # --- 物理常数 (25°C, 101.325 kPa) ---
        self.DIFFUSIVITY_H2O = 2.49e-5  # m^2/s
        self.MOLAR_VOLUME_AIR = 0.0244  # m^3/mol

    def calculate_basic_properties(self, masks, boxes=None, confidences=None):
        """
        计算综合表型属性 (Master Method)
        """
        results = []
        centroids = [] # 用于空间分析 [x, y]
        
        if len(masks) == 0:
            return pd.DataFrame()

        # 1. 逐实例计算形态学特征
        for i, mask in enumerate(masks):
            # 处理 Tensor 或 Numpy
            if hasattr(mask, 'cpu'): 
                mask = mask.cpu().numpy()
            
            # 确保是二值掩码
            binary_mask = mask > 0.5
            if not binary_mask.any():
                continue
            
            # --- A. 基础几何 (Basic Geometry) ---
            area_px = np.sum(binary_mask)
            area_um2 = area_px * self.pixel_area_ratio
            
            perimeter_px = self._calculate_perimeter_robust(binary_mask)
            perimeter_um = perimeter_px * self.pixel_ratio
            
            # [Fixed] 计算 Feret 直径 (Max Caliper)
            feret_diameter_px = self._calculate_feret_diameter_px(binary_mask)
            feret_diameter_um = feret_diameter_px * self.pixel_ratio
            
            y_indices, x_indices = np.where(binary_mask)
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            centroids.append([center_x, center_y])
            
            # 边界框
            width_um, height_um = 0, 0
            if boxes is not None and i < len(boxes) and len(boxes[i]) >= 4:
                x0, y0, x1, y1 = boxes[i][:4]
                width_um = abs(x1 - x0) * self.pixel_ratio
                height_um = abs(y1 - y0) * self.pixel_ratio
            
            # --- B. 形状描述符 ---
            # PCA 主轴
            major_axis_px, minor_axis_px, pca_angle = self._calculate_principal_axes_pca(binary_mask)
            major_axis_um = major_axis_px * self.pixel_ratio
            minor_axis_um = minor_axis_px * self.pixel_ratio
            aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else 0
            
            circularity = (4 * np.pi * area_um2) / (perimeter_um ** 2) if perimeter_um > 0 else 0
            solidity = self._calculate_solidity(binary_mask)
            convexity = self._calculate_convexity(binary_mask)
            rectangularity = self._calculate_rectangularity(binary_mask, area_px)

            # --- C. 高级力学与拓扑 (Advanced Mechanics & Topology) ---
            # 1. 骨架长度
            skeleton_len_px = self._calculate_skeleton_length(binary_mask)
            skeleton_len_um = skeleton_len_px * self.pixel_ratio

            # 2. 最大内切圆 (LEC) 与 覆盖率 (Coverage)
            # [cite: 88, 91] LEC 是机械应力热点的代理
            max_inscribed_radius_px = self._calculate_inscribed_circle_radius(binary_mask)
            max_inscribed_width_um = (max_inscribed_radius_px * 2) * self.pixel_ratio
            
            lec_area_px = np.pi * (max_inscribed_radius_px ** 2)
            lec_coverage = lec_area_px / area_px if area_px > 0 else 0 # LEC Area / Cell Area

            # [cite_start]3. 波动指数 (Undulation Index) [cite: 77, 79]
            equiv_circle_circumference = 2 * np.pi * np.sqrt(area_um2 / np.pi)
            undulation_index = perimeter_um / equiv_circle_circumference if equiv_circle_circumference > 0 else 1.0
            
            # [cite_start]4. 裂片度 (Lobeyness) [cite: 101]
            lobeyness = 1.0
            lobe_count = 0
            if cv2 is not None:
                lobeyness, lobe_count = self._calculate_lobeyness_cv2(binary_mask, perimeter_um)
            
            # [New P1] 多尺度 Lobe 分割与描述符
            lobe_metrics = self._calculate_multi_scale_lobes(binary_mask)
            lobe_count_multi = lobe_metrics['lobe_count_multi']
            lobe_area_mean = lobe_metrics['lobe_area_mean']
            lobe_width_mean = lobe_metrics['lobe_width_mean']
            lobe_autocorr = lobe_metrics['lobe_autocorr']
            
            # [cite_start]5. 扩展的可见性图指标 (Extended Visibility Graph Metrics) [cite: 104, 111]
            # 仅对足够大的对象计算以节省时间 (如 > 50px)
            vg_avg_degree = np.nan
            vg_density = np.nan
            vg_degree_skew = np.nan
            vg_degree_kurt = np.nan
            vg_closeness_mean = np.nan
            vg_closeness_skew = np.nan
            vg_closeness_kurt = np.nan
            lobe_count_vg = 0
            neck_count_vg = 0
            # [New P0] Betweenness & Clustering
            vg_betweenness_mean = np.nan
            vg_clustering_mean = np.nan
            
            if nx is not None and Polygon is not None and area_px > 50:
                vg_metrics = self._analyze_visibility_graph_extended(binary_mask)
                vg_avg_degree = vg_metrics['vg_avg_degree']
                vg_density = vg_metrics['vg_density']
                vg_degree_skew = vg_metrics['vg_degree_skew']
                vg_degree_kurt = vg_metrics['vg_degree_kurt']
                vg_closeness_mean = vg_metrics['vg_closeness_mean']
                vg_closeness_skew = vg_metrics['vg_closeness_skew']
                vg_closeness_kurt = vg_metrics['vg_closeness_kurt']
                lobe_count_vg = vg_metrics['lobe_count_vg']
                neck_count_vg = vg_metrics['neck_count_vg']
                vg_betweenness_mean = vg_metrics['vg_betweenness_mean']
                vg_clustering_mean = vg_metrics['vg_clustering_mean']

            # 6. ECT 复杂度 (升级为多角度指纹)
            # [New P0] Multi-angle ECT
            ect_metrics = self._calculate_full_ect(binary_mask)
            ect_mean = ect_metrics['ect_mean']
            ect_var = ect_metrics['ect_var']
            ect_max_amp = ect_metrics['ect_max_amp']
            ect_fourier_low = ect_metrics['ect_fourier'][0] if 'ect_fourier' in ect_metrics else np.nan

            # [New P0] 厚度分布统计
            thickness_metrics = self._calculate_thickness_distribution(binary_mask)
            thickness_mean_um = thickness_metrics['mean'] * self.pixel_ratio * 2  # Diameter
            thickness_std_um = thickness_metrics['std'] * self.pixel_ratio * 2
            thickness_cv = thickness_metrics['cv']

            # [New P0] 曲率分布统计
            curvature_metrics = self._calculate_curvature_distribution(binary_mask)
            curvature_mean = curvature_metrics['mean']
            curvature_std = curvature_metrics['std']
            curvature_skew = curvature_metrics['skew']
            curvature_kurt = curvature_metrics['kurt']

            # --- D. 气孔生理潜力 (g_smax) ---
            # [cite_start]仅当对象为气孔时有效 [cite: 11, 12]
            pore_depth_m = (minor_axis_um * 1e-6) # Proxy: Guard cell width
            # 假设最大开口为椭圆，长轴为保卫细胞长轴的一半
            a_max_m2 = (np.pi * (major_axis_um/2 * 1e-6) * (major_axis_um/4 * 1e-6))
            
            gs_capacity_mol_s = 0
            if a_max_m2 > 0:
                end_correction = (np.pi / 2) * np.sqrt(a_max_m2 / np.pi)
                gs_capacity_mol_s = (self.DIFFUSIVITY_H2O * a_max_m2) / \
                                    (self.MOLAR_VOLUME_AIR * (pore_depth_m + end_correction))

            # 收集数据
            instance_data = {
                'instance_id': i + 1,
                'center_x': center_x,
                'center_y': center_y,
                'area_px': area_px,
                'area_um2': area_um2,
                'perimeter_um': perimeter_um,
                'feret_diameter_um': feret_diameter_um, # [Fixed] 确保加入字典
                'major_axis_um': major_axis_um,
                'minor_axis_um': minor_axis_um,
                'aspect_ratio': aspect_ratio,
                'pca_angle': pca_angle,
                'circularity': circularity,
                'solidity': solidity, # [cite: 83]
                'convexity': convexity,
                'rectangularity': rectangularity,
                'max_inscribed_width_um': max_inscribed_width_um,
                'lec_coverage': lec_coverage, 
                'skeleton_len_um': skeleton_len_um,
                'undulation_index': undulation_index, # [cite: 79]
                'lobeyness': lobeyness, # [cite: 101]
                'lobe_count': lobe_count,
                'vg_avg_degree': vg_avg_degree, # [cite: 112]
                'vg_density': vg_density, # [cite: 112]
                # 新增可见性图扩展指标
                'vg_degree_skew': vg_degree_skew,
                'vg_degree_kurt': vg_degree_kurt,
                'vg_closeness_mean': vg_closeness_mean,
                'vg_closeness_skew': vg_closeness_skew,
                'vg_closeness_kurt': vg_closeness_kurt,
                'lobe_count_vg': lobe_count_vg,  # 基于接近中心性检测
                'neck_count_vg': neck_count_vg,  # 基于接近中心性检测
                'vg_betweenness_mean': vg_betweenness_mean,
                'vg_clustering_mean': vg_clustering_mean,
                'ect_mean': ect_mean,
                'ect_var': ect_var,
                'ect_max_amp': ect_max_amp,
                'ect_fourier_low': ect_fourier_low,
                'thickness_mean_um': thickness_mean_um,
                'thickness_std_um': thickness_std_um,
                'thickness_cv': thickness_cv,
                'curvature_mean': curvature_mean,
                'curvature_std': curvature_std,
                'curvature_skew': curvature_skew,
                'curvature_kurt': curvature_kurt,
                'lobe_count_multi': lobe_count_multi,
                'lobe_area_mean': lobe_area_mean,
                'lobe_width_mean': lobe_width_mean,
                'lobe_autocorr': lobe_autocorr,
                'gs_capacity_mol_s': gs_capacity_mol_s,
                'confidence': confidences[i] if confidences is not None and i < len(confidences) else 1.0
            }
            results.append(instance_data)

        # 2. 空间分布分析 (Population Level)
        df = pd.DataFrame(results)
        if not df.empty and len(df) > 3:
            img_shape = masks[0].shape if hasattr(masks[0], 'shape') else (1024, 1024)
            
            # Voronoi Analysis (Entropy & Areas)
            voronoi_metrics = self._analyze_spatial_voronoi(centroids, img_shape)
            if voronoi_metrics['areas'] is not None:
                df['voronoi_area_um2'] = voronoi_metrics['areas']
                # [Fixed] 将群体熵值赋予每一个个体，以便 GUI 标签读取
                df['voronoi_entropy'] = voronoi_metrics['entropy'] 
            else:
                df['voronoi_area_um2'] = np.nan
                df['voronoi_entropy'] = np.nan
            
            # [New P1] Ripley’s K + PCF
            spatial_metrics = self._analyze_spatial_ripley_pcf(centroids, img_shape)
            df['ripley_L_max'] = spatial_metrics['ripley_L_max']  # 群体指标重复赋值
            df['pcf_g_max'] = spatial_metrics['pcf_g_max']
            df['csr_deviation'] = spatial_metrics['csr_deviation']

        return df

    # ==================== 扩展的可见性图分析 ====================

    def _analyze_visibility_graph_extended(self, binary_mask, num_nodes=50):
        """
        构建可见性图并计算扩展的图论指标
        [cite_start]Ref: GraVis [cite: 104, 105]
        返回：包含节点度分布统计、相对完整度、中心性指标的字典
        """
        if cv2 is None or nx is None or Polygon is None:
            return self._empty_vg_metrics()

        try:
            # 1. 获取轮廓并重采样
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: 
                return self._empty_vg_metrics()
            
            cnt = max(contours, key=len)
            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            points = approx.reshape(-1, 2)
            if len(points) > num_nodes:
                indices = np.linspace(0, len(points)-1, num_nodes, dtype=int)
                points = points[indices]
            elif len(points) < 4:
                # 对于简单凸多边形（如矩形），返回默认值
                return self._simple_convex_vg_metrics(len(points))

            # 2. 构建 Shapely 多边形
            poly_geom = Polygon(points)
            if not poly_geom.is_valid:
                poly_geom = poly_geom.buffer(0)

            # 3. 构建 networkx 图
            G = nx.Graph()
            n = len(points)
            G.add_nodes_from(range(n))
            
            # 4. 连边检测
            for i in range(n):
                next_i = (i + 1) % n
                G.add_edge(i, next_i)  # 相邻点必连
                
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1: 
                        continue 
                    
                    p1 = points[i]
                    p2 = points[j]
                    line = LineString([p1, p2])
                    
                    # 简化判断：中点在内部即视为可见
                    mid_point = Point((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    if poly_geom.contains(mid_point):
                         G.add_edge(i, j)

            # 5. 计算基础图指标
            m = G.number_of_edges()
            max_possible_edges = n * (n - 1) / 2 if n > 1 else 0
            density = m / max_possible_edges if max_possible_edges > 0 else 0
            
            # 6. 节点度分布统计
            degrees = list(dict(G.degree()).values())
            degree_series = pd.Series(degrees)
            
            degree_skew = degree_series.skew()
            degree_kurt = degree_series.kurtosis()
            
            # 7. 接近中心性统计与 lobes/necks 检测
            closeness = list(nx.closeness_centrality(G).values())
            closeness_series = pd.Series(closeness)
            
            # 检测 lobes（局部最小值）和 necks（局部最大值）
            lobe_count = 0
            neck_count = 0
            
            # 使用滑动窗口检测局部极值（窗口大小=3）
            for i in range(1, len(closeness)-1):
                if closeness[i] < closeness[i-1] and closeness[i] < closeness[i+1]:
                    lobe_count += 1  # 局部最小值 → lobe
                elif closeness[i] > closeness[i-1] and closeness[i] > closeness[i+1]:
                    neck_count += 1  # 局部最大值 → neck
            
            # [New P0] Betweenness centrality & Clustering coefficient
            betweenness = list(nx.betweenness_centrality(G).values())
            betweenness_mean = np.mean(betweenness) if betweenness else 0
            
            clustering = list(nx.clustering(G).values())
            clustering_mean = np.mean(clustering) if clustering else 0
            
            # 8. 返回所有指标
            return {
                'vg_avg_degree': np.mean(degrees) if degrees else 0,
                'vg_density': density,  # 即 relative completeness
                'vg_degree_skew': degree_skew,
                'vg_degree_kurt': degree_kurt,
                'vg_closeness_mean': np.mean(closeness) if closeness else 0,
                'vg_closeness_skew': closeness_series.skew(),
                'vg_closeness_kurt': closeness_series.kurtosis(),
                'lobe_count_vg': lobe_count,   # 通过接近中心性检测
                'neck_count_vg': neck_count,    # 通过接近中心性检测
                # [New]
                'vg_betweenness_mean': betweenness_mean,
                'vg_clustering_mean': clustering_mean
            }

        except Exception as e:
            print(f"Extended visibility graph analysis error: {e}")
            return self._empty_vg_metrics()

    def _empty_vg_metrics(self):
        """返回空值占位字典"""
        return {
            'vg_avg_degree': np.nan,
            'vg_density': np.nan,
            'vg_degree_skew': np.nan,
            'vg_degree_kurt': np.nan,
            'vg_closeness_mean': np.nan,
            'vg_closeness_skew': np.nan,
            'vg_closeness_kurt': np.nan,
            'lobe_count_vg': 0,
            'neck_count_vg': 0,
            'vg_betweenness_mean': np.nan,  # [New]
            'vg_clustering_mean': np.nan    # [New]
        }

    def _simple_convex_vg_metrics(self, n_nodes):
        """对于简单凸多边形（如矩形、圆形）的默认指标"""
        return {
            'vg_avg_degree': n_nodes - 1,  # 完全图
            'vg_density': 1.0,              # 密度为1
            'vg_degree_skew': 0.0,          # 对称分布
            'vg_degree_kurt': 0.0,          # 正态峰度
            'vg_closeness_mean': 1.0,       # 所有节点完全连通
            'vg_closeness_skew': 0.0,
            'vg_closeness_kurt': 0.0,
            'lobe_count_vg': 0,             # 凸形无 lobes
            'neck_count_vg': 0,             # 凸形无 necks
            'vg_betweenness_mean': 0.0,     # [New] 凸形无中介
            'vg_clustering_mean': 1.0       # [New] 全聚类
        }

    # ==================== 新增方法：曲率分布统计 [P0] ====================

    def _calculate_curvature_distribution(self, binary_mask):
        """计算轮廓曲率分布统计 [cite: Zahn & Roskies, 1972]"""
        if cv2 is None:
            return {'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurt': np.nan}
        
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                return {'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurt': np.nan}
            
            cnt = max(contours, key=len).squeeze()
            if len(cnt) < 5:
                return {'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurt': np.nan}
            
            # 计算曲率：使用三点法
            curvatures = []
            for i in range(len(cnt)):
                p1 = cnt[i - 2]
                p2 = cnt[i - 1]
                p3 = cnt[i % len(cnt)]
                v1 = p2 - p1
                v2 = p3 - p2
                angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                curvatures.append(np.abs(angle))
            
            curv_series = pd.Series(curvatures)
            return {
                'mean': curv_series.mean(),
                'std': curv_series.std(),
                'skew': curv_series.skew(),
                'kurt': curv_series.kurtosis()
            }
        except:
            return {'mean': np.nan, 'std': np.nan, 'skew': np.nan, 'kurt': np.nan}

    # ==================== 新增方法：厚度分布统计 [P0] ====================

    def _calculate_thickness_distribution(self, binary_mask):
        """计算内部厚度（距离变换 × 2）分布"""
        try:
            dist_map = distance_transform_edt(binary_mask)
            thicknesses = dist_map[dist_map > 0]  # 只取内部点
            if len(thicknesses) == 0:
                return {'mean': 0, 'std': 0, 'cv': 0}
            
            mean = np.mean(thicknesses)
            std = np.std(thicknesses)
            cv = std / mean if mean > 0 else 0
            return {'mean': mean, 'std': std, 'cv': cv}
        except:
            return {'mean': 0, 'std': 0, 'cv': 0}

    # ==================== 新增方法：多角度 ECT 指纹 [P0] ====================

    def _calculate_full_ect(self, binary_mask, num_angles=36):
        """多角度 ECT 计算 [Enhanced from simple_ect]"""
        if cv2 is None:
            return {'ect_mean': np.nan, 'ect_var': np.nan, 'ect_max_amp': np.nan, 'ect_fourier': []}
        
        try:
            ect_curves = []
            angles = np.linspace(0, 180, num_angles, endpoint=False)
            img_uint8 = binary_mask.astype(np.uint8)
            
            for angle in angles:
                M = cv2.getRotationMatrix2D((img_uint8.shape[1]/2, img_uint8.shape[0]/2), angle, 1)
                rotated = cv2.warpAffine(img_uint8, M, (img_uint8.shape[1], img_uint8.shape[0]), flags=cv2.INTER_NEAREST)
                ec_profile = []
                for row in rotated:
                    runs = np.count_nonzero(np.diff(np.concatenate(([0], row, [0]))) != 0) // 2  # 连通分量
                    # 简化孔洞估计：假设无孔（植物细胞常见）
                    chi = runs  # 近似 χ = C (无孔)
                    ec_profile.append(chi)
                ect_curves.append(ec_profile)
            
            # 填充到相同长度
            max_len = max(len(p) for p in ect_curves)
            ect_matrix = np.array([p + [0] * (max_len - len(p)) for p in ect_curves])
            
            # 统计特征
            ect_mean = np.mean(ect_matrix)
            ect_var = np.var(ect_matrix)
            ect_max_amp = np.max(ect_matrix) - np.min(ect_matrix)
            # 傅里叶分量（平均曲线）
            avg_curve = np.mean(ect_matrix, axis=0)
            fourier = np.abs(np.fft.fft(avg_curve))[:10]  # 前10个低频分量
            
            return {
                'ect_mean': ect_mean,
                'ect_var': ect_var,
                'ect_max_amp': ect_max_amp,
                'ect_fourier': fourier
            }
        except:
            return {'ect_mean': np.nan, 'ect_var': np.nan, 'ect_max_amp': np.nan, 'ect_fourier': []}

    # ==================== 新增方法：多尺度 Lobe 分割 [P1] ====================

    def _calculate_multi_scale_lobes(self, binary_mask, thresholds=[2, 4, 8]):
        """多阈值裂片检测与描述符"""
        if cv2 is None:
            return {'lobe_count_multi': 0, 'lobe_area_mean': np.nan, 'lobe_width_mean': np.nan, 'lobe_autocorr': np.nan}
        
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {'lobe_count_multi': 0, 'lobe_area_mean': np.nan, 'lobe_width_mean': np.nan, 'lobe_autocorr': np.nan}
            
            cnt = max(contours, key=len)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3:
                return {'lobe_count_multi': 0, 'lobe_area_mean': np.nan, 'lobe_width_mean': np.nan, 'lobe_autocorr': np.nan}
            
            defects = cv2.convexityDefects(cnt, hull_indices)
            lobe_counts = []
            lobe_areas = []
            lobe_widths = []
            lobe_positions = []  # 用于自相关
            
            if defects is not None:
                for thresh in thresholds:
                    count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        depth = d / 256.0
                        if depth > thresh:
                            count += 1
                            # 近似 lobe 面积/宽度（简化：缺陷深度 × 距离）
                            far = cnt[f][0]
                            start = cnt[s][0]
                            end = cnt[e][0]
                            width = np.linalg.norm(start - end)
                            area_approx = depth * width / 2  # 三角形近似
                            lobe_areas.append(area_approx)
                            lobe_widths.append(width)
                            lobe_positions.append(np.mean([start, end, far], axis=0))  # 中心位置
                    
                    lobe_counts.append(count)
            
            lobe_count_multi = np.mean(lobe_counts) if lobe_counts else 0
            lobe_area_mean = np.mean(lobe_areas) * self.pixel_area_ratio if lobe_areas else np.nan
            lobe_width_mean = np.mean(lobe_widths) * self.pixel_ratio if lobe_widths else np.nan
            
            # Lobe 空间自相关（Moran's I 简化版）
            if len(lobe_positions) > 5:
                pos = np.array(lobe_positions)
                dists = distance_matrix(pos, pos)
                np.fill_diagonal(dists, np.inf)
                inv_dists = 1 / dists
                autocorr = np.mean(inv_dists) / np.mean(1 / np.arange(1, len(pos) + 1))  # 归一化
            else:
                autocorr = np.nan
            
            return {
                'lobe_count_multi': lobe_count_multi,
                'lobe_area_mean': lobe_area_mean,
                'lobe_width_mean': lobe_width_mean,
                'lobe_autocorr': autocorr
            }
        except:
            return {'lobe_count_multi': 0, 'lobe_area_mean': np.nan, 'lobe_width_mean': np.nan, 'lobe_autocorr': np.nan}

    # ==================== 新增方法：Ripley’s K + PCF [P1] ====================

    def _analyze_spatial_ripley_pcf(self, points, shape, num_r=20):
        """Ripley’s K, L(r) 和 PCF g(r) 计算 [cite: Ripley, 1977]"""
        if len(points) < 10:
            return {'ripley_L_max': 0, 'pcf_g_max': 1, 'csr_deviation': 0}
        
        try:
            points_um = np.array(points) * self.pixel_ratio
            area_um2 = shape[0] * shape[1] * self.pixel_area_ratio
            n = len(points_um)
            max_dist = min(shape) * self.pixel_ratio / 2  # 合理半径
            radii = np.linspace(1, max_dist, num_r)
            
            # Ripley’s K
            dists = distance_matrix(points_um, points_um)
            k_values = []
            for r in radii:
                count = np.sum((dists > 0) & (dists < r))
                k = (area_um2 / (n ** 2)) * count
                k_values.append(k)
            
            l_values = np.sqrt(np.array(k_values) / np.pi) - radii
            ripley_L_max = np.max(l_values)
            
            # PCF g(r) ≈ K'(r) / (2πr)
            dk_dr = np.diff(k_values) / np.diff(radii)
            r_mid = (radii[:-1] + radii[1:]) / 2
            g_values = dk_dr / (2 * np.pi * r_mid)
            pcf_g_max = np.max(g_values) if len(g_values) > 0 else 1
            
            # CSR 偏差（与理论 L(r)=0 的均方差）
            csr_dev = np.mean(l_values ** 2)
            
            return {'ripley_L_max': ripley_L_max, 'pcf_g_max': pcf_g_max, 'csr_deviation': csr_dev}
        except:
            return {'ripley_L_max': 0, 'pcf_g_max': 1, 'csr_deviation': 0}

    # ==================== 几何计算辅助方法 (Optimized) ====================

    def _calculate_feret_diameter_px(self, binary_mask):
        """计算 Feret 直径 (最大卡尺距离)"""
        try:
            y_idxs, x_idxs = np.where(binary_mask)
            if len(x_idxs) == 0: return 0
            
            # 使用凸包加速
            points = np.column_stack((x_idxs, y_idxs))
            if len(points) < 3: return 0
            
            hull = morphology.convex_hull_image(binary_mask)
            y_hull, x_hull = np.where(hull)
            hull_points = np.column_stack((x_hull, y_hull))
            
            # 降采样以提高速度 (只取轮廓上的点计算最大距离)
            if len(hull_points) > 200: 
                indices = np.linspace(0, len(hull_points)-1, 200, dtype=int)
                hull_points = hull_points[indices]

            dists = distance_matrix(hull_points, hull_points)
            return np.max(dists)
        except:
            return 0

    def _calculate_perimeter_robust(self, binary_mask):
        """Robust perimeter calculation"""
        if cv2 is not None:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                return cv2.arcLength(max(contours, key=len), True)
        eroded = ndimage.binary_erosion(binary_mask)
        perimeter = np.logical_xor(binary_mask, eroded)
        return np.sum(perimeter)

    def _calculate_principal_axes_pca(self, binary_mask):
        y_indices, x_indices = np.where(binary_mask)
        if len(x_indices) < 3: return 0, 0, 0
        coords = np.column_stack((x_indices, y_indices))
        try:
            cov = np.cov(coords.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return 0, 0, 0
        eigenvalues = np.maximum(eigenvalues, 0)
        major_len = 4 * np.sqrt(eigenvalues[1])
        minor_len = 4 * np.sqrt(eigenvalues[0])
        v_major = eigenvectors[:, 1]
        angle = np.degrees(np.arctan2(v_major[1], v_major[0]))
        return major_len, minor_len, angle

    def _calculate_solidity(self, binary_mask):
        try:
            convex_hull = morphology.convex_hull_image(binary_mask)
            convex_area = np.sum(convex_hull)
            return np.sum(binary_mask) / convex_area if convex_area > 0 else 0
        except: return 0

    def _calculate_convexity(self, binary_mask):
        try:
            perimeter = self._calculate_perimeter_robust(binary_mask)
            convex_hull = morphology.convex_hull_image(binary_mask)
            convex_perimeter = self._calculate_perimeter_robust(convex_hull)
            return convex_perimeter / perimeter if perimeter > 0 else 0
        except: return 0

    def _calculate_rectangularity(self, binary_mask, area_px):
        if cv2 is None: return 0
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 0
            cnt = max(contours, key=len)
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            rect_area = w * h
            return area_px / rect_area if rect_area > 0 else 0
        except: return 0

    def _calculate_inscribed_circle_radius(self, binary_mask):
        try:
            dist_map = distance_transform_edt(binary_mask)
            return np.max(dist_map)
        except: return 0

    def _calculate_skeleton_length(self, binary_mask):
        try:
            skeleton = morphology.skeletonize(binary_mask)
            return np.sum(skeleton)
        except: return 0

    def _calculate_lobeyness_cv2(self, binary_mask, perimeter_um):
        try:
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return 1.0, 0
            cnt = max(contours, key=len)
            hull = cv2.convexHull(cnt)
            hull_perimeter_px = cv2.arcLength(hull, True)
            hull_perimeter_um = hull_perimeter_px * self.pixel_ratio
            lobeyness = perimeter_um / hull_perimeter_um if hull_perimeter_um > 0 else 1.0
            
            lobe_count = 0
            if len(cnt) > 5:
                hull_indices = cv2.convexHull(cnt, returnPoints=False)
                if hull_indices is not None and len(hull_indices) > 3:
                    defects = cv2.convexityDefects(cnt, hull_indices)
                    if defects is not None:
                        depth_thresh = 2.0 
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            if (d / 256.0) > depth_thresh:
                                lobe_count += 1
            return lobeyness, lobe_count
        except: return 1.0, 0

    # ==================== 空间分析辅助方法 (Updated) ====================

    def _analyze_spatial_voronoi(self, points, shape):
        """
        [cite_start]Voronoi 分析：增加熵 (Entropy) 计算 [cite: 49]
        """
        if len(points) < 4:
            return {'entropy': None, 'cv': None, 'areas': None}
        
        try:
            h, w = shape[:2]
            points_np = np.array(points)
            vor = Voronoi(points_np)
            regions_areas = []
            neighbor_counts = [] # 用于计算熵
            
            # 使用 Shapely 计算精确的截断面积
            if Polygon:
                img_box = Polygon([(0,0), (0, h), (w, h), (w, 0)])
                for i, region_idx in enumerate(vor.point_region):
                    region = vor.regions[region_idx]
                    
                    # 1. 面积与有效性检查
                    if -1 in region or len(region) == 0:
                        regions_areas.append(np.nan)
                        neighbor_counts.append(np.nan)
                        continue
                    
                    vertices = vor.vertices[region]
                    poly = Polygon(vertices)
                    
                    if poly.is_valid:
                        # 2. 统计几何邻居数 (即多边形边数)
                        intersection = poly.intersection(img_box)
                        area_um2 = intersection.area * self.pixel_area_ratio
                        regions_areas.append(area_um2)
                        
                        # 如果多边形完全在图像内，则顶点数=邻居数
                        if poly.within(img_box):
                            neighbor_counts.append(len(region))
                        else:
                            neighbor_counts.append(np.nan) 
                    else:
                        regions_areas.append(np.nan)
                        neighbor_counts.append(np.nan)
            else:
                # Fallback
                regions_areas = [np.nan] * len(points)
                neighbor_counts = [np.nan] * len(points)
                
            # --- 计算指标 ---
            # CV of Area
            valid_areas = [a for a in regions_areas if not np.isnan(a) and a > 0]
            cv = np.std(valid_areas) / np.mean(valid_areas) if valid_areas else None
            
            # Voronoi Entropy (S_vor)
            # [cite_start]Ref: [cite: 51] S = - sum(P_n * ln(P_n))
            valid_neighbors = [n for n in neighbor_counts if not np.isnan(n)]
            entropy = None
            if len(valid_neighbors) > 5: # 适当放宽样本量限制
                counts = pd.Series(valid_neighbors).value_counts(normalize=True) # P_n
                entropy = -np.sum(counts * np.log(counts))
                
            return {'entropy': entropy, 'cv': cv, 'areas': regions_areas}
            
        except Exception as e:
            return {'entropy': None, 'cv': None, 'areas': None}

    # ==================== 报告与导出 ====================

    def calculate_stomatal_indices(self, stoma_df, pavement_cell_df, image_area_um2=0):
        results = {}
        n_stoma = len(stoma_df)
        n_pavement = len(pavement_cell_df)
        
        results['stomatal_count'] = n_stoma
        results['pavement_cell_count'] = n_pavement
        
        if image_area_um2 > 0:
            results['stomatal_density_per_mm2'] = (n_stoma / image_area_um2) * 1e6
        
        if n_stoma + n_pavement > 0:
            results['stomatal_index_percent'] = (n_stoma / (n_stoma + n_pavement)) * 100
            
        def safe_mean(df, col):
            return df[col].mean() if col in df.columns else np.nan

        if n_stoma > 0:
            results.update({
                'stomatal_area_mean_um2': safe_mean(stoma_df, 'area_um2'),
                'stomatal_length_mean_um': safe_mean(stoma_df, 'major_axis_um'),
                'stomatal_width_mean_um': safe_mean(stoma_df, 'minor_axis_um'),
                'stomatal_feret_diameter_mean_um': safe_mean(stoma_df, 'feret_diameter_um'), 
                'stomatal_solidity_mean': safe_mean(stoma_df, 'solidity'),
                'stomatal_circularity_mean': safe_mean(stoma_df, 'circularity'),
                'stomatal_lec_coverage_mean': safe_mean(stoma_df, 'lec_coverage'),
                'stomatal_voronoi_entropy': safe_mean(stoma_df, 'voronoi_entropy'),
                # 新增可见性图指标
                'stomatal_vg_density_mean': safe_mean(stoma_df, 'vg_density'),
                'stomatal_vg_degree_skew_mean': safe_mean(stoma_df, 'vg_degree_skew'),
                # 扩展指标
                'stomatal_curvature_mean': safe_mean(stoma_df, 'curvature_mean'),
                'stomatal_thickness_cv': safe_mean(stoma_df, 'thickness_cv'),
                'stomatal_ect_var': safe_mean(stoma_df, 'ect_var'),
            })
            
            if 'gs_capacity_mol_s' in stoma_df.columns:
                total_conductance = stoma_df['gs_capacity_mol_s'].sum()
                image_area_m2 = image_area_um2 * 1e-12
                if image_area_m2 > 0:
                    results['theoretical_gsmax_mol_m2_s'] = total_conductance / image_area_m2
        
        if n_pavement > 0:
            results.update({
                'pavement_area_mean_um2': safe_mean(pavement_cell_df, 'area_um2'),
                'pavement_undulation_index_mean': safe_mean(pavement_cell_df, 'undulation_index'),
                'pavement_lobeyness_mean': safe_mean(pavement_cell_df, 'lobeyness'),
                'pavement_lobe_count_vg_mean': safe_mean(pavement_cell_df, 'lobe_count_vg'),  # 新增
                'pavement_neck_count_vg_mean': safe_mean(pavement_cell_df, 'neck_count_vg'),  # 新增
                'pavement_vg_avg_degree_mean': safe_mean(pavement_cell_df, 'vg_avg_degree'),
                'pavement_vg_density_mean': safe_mean(pavement_cell_df, 'vg_density'),
                'pavement_vg_degree_skew_mean': safe_mean(pavement_cell_df, 'vg_degree_skew'),  # 新增
                'pavement_vg_closeness_skew_mean': safe_mean(pavement_cell_df, 'vg_closeness_skew'),  # 新增
                # [New] 扩展指标
                'pavement_curvature_skew_mean': safe_mean(pavement_cell_df, 'curvature_skew'),
                'pavement_lobe_count_multi_mean': safe_mean(pavement_cell_df, 'lobe_count_multi'),
                'pavement_ripley_L_max': safe_mean(pavement_cell_df, 'ripley_L_max'),
            })

        if n_stoma > 1 and 'center_x' in stoma_df.columns:
            results['stomatal_distribution_uniformity'] = self.calculate_distribution_uniformity(
                stoma_df[['center_x', 'center_y']].values
            )
            
        return results
        
    def calculate_stomatal_aperture_indices(self, aperture_df, stoma_df):
        results = {}
        n_apertures = len(aperture_df)
        results['aperture_count'] = n_apertures
        if n_apertures > 0:
            results['aperture_area_mean_um2'] = aperture_df['area_um2'].mean()
        
        if n_apertures > 0 and not stoma_df.empty:
            try:
                ap_centers = aperture_df[['center_x', 'center_y']].values
                st_centers = stoma_df[['center_x', 'center_y']].values
                dists = distance_matrix(ap_centers, st_centers)
                nearest_stoma_indices = np.argmin(dists, axis=1)
                
                ratios = []
                for i, stoma_idx in enumerate(nearest_stoma_indices):
                    if dists[i, stoma_idx] * self.pixel_ratio < 50:
                        ap_area = aperture_df.iloc[i]['area_um2']
                        st_area = stoma_df.iloc[stoma_idx]['area_um2']
                        if st_area > 0:
                            ratios.append(ap_area / st_area)
                if ratios:
                    results['aperture_ratio_mean'] = np.mean(ratios)
            except: pass
        return results
    
    def calculate_distribution_uniformity(self, centers):
        if len(centers) < 2: return 1.0
        try:
            tree = KDTree(centers)
            dists, _ = tree.query(centers, k=2) 
            nearest_dists = dists[:, 1]
            if len(nearest_dists) > 0 and np.mean(nearest_dists) > 0:
                cv = np.std(nearest_dists) / np.mean(nearest_dists)
                return 1.0 / (1.0 + cv)
        except: pass
        return 0.0

    def generate_comprehensive_report(self, stoma_df, pavement_df, aperture_df=None, 
                                     image_area_um2=0, image_name="", image_path=""):
        report = {
            'image_info': {
                'image_name': image_name,
                'image_path': image_path,
                'image_area_um2': image_area_um2,
                'scale_factor_px_per_um': self.scale_factor
            },
            'basic_counts': {
                'stomatal_count': len(stoma_df),
                'pavement_cell_count': len(pavement_df),
                'aperture_count': len(aperture_df) if aperture_df is not None else 0
            },
            'composite_indices': self.calculate_stomatal_indices(stoma_df, pavement_df, image_area_um2)
        }
        if aperture_df is not None and not aperture_df.empty:
            ap_indices = self.calculate_stomatal_aperture_indices(aperture_df, stoma_df)
            report['composite_indices'].update(ap_indices)
        return report

    def export_to_csv(self, stoma_df, pavement_cell_df, aperture_df=None, 
                     report=None, output_path="phenotype_results.xlsx"):
        if output_path.endswith('.csv') or output_path.endswith('.txt'):
            output_path = os.path.splitext(output_path)[0] + '.xlsx'
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if not stoma_df.empty:
                    stoma_df.to_excel(writer, sheet_name='Stomata', index=False)
                if not pavement_cell_df.empty:
                    pavement_cell_df.to_excel(writer, sheet_name='Pavement_Cells', index=False)
                if aperture_df is not None and not aperture_df.empty:
                    aperture_df.to_excel(writer, sheet_name='Apertures', index=False)
                if report:
                    flat_data = []
                    self._flatten_dict(report, flat_data)
                    pd.DataFrame(flat_data, columns=['Key', 'Value', 'Raw_Value']).to_excel(
                        writer, sheet_name='Summary_Report', index=False)
            return output_path
        except Exception as e:
            print(f"Export error: {e}")
            traceback.print_exc()
            return None

    def _flatten_dict(self, d, result_list, parent_key=''):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                self._flatten_dict(v, result_list, new_key)
            else:
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                result_list.append([new_key, val_str, v])

if __name__ == "__main__":
    pass