import numpy as np
from PIL import Image
import torch
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QCoreApplication
import json
import os
from datetime import datetime
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.ndimage import label
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import cv2
import gc
import networkx as nx
from shapely.geometry import Polygon
from matplotlib.path import Path as MplPath
import colorsys
import matplotlib.pyplot as plt
import psutil
import re
import pandas as pd
from core.phenotype_calculator import PhenotypeCalculator

class TileProcessor(QThread):
    """处理单个分块的线程"""
    finished = pyqtSignal(dict)  # tile_index, tile_image, result
    progress = pyqtSignal(int, str)  # current, total
    error = pyqtSignal(str)
    
    def __init__(self, parent, tile_index, tile_image, tile_position, prompt, processor, state):
        super().__init__()
        self.parent = parent
        self.tile_index = tile_index
        self.tile_image = tile_image
        self.tile_position = tile_position  # (x_start, y_start, x_end, y_end)
        self.prompt = prompt
        self.processor = processor
        self.state = state
    
    def run(self):
        try:
            # 设置当前分块图像到 processor
            tile_state = self.processor.set_image(self.tile_image)
            
            # 使用文本提示进行分割
            self.processor.reset_all_prompts(tile_state)
            tile_state = self.processor.set_text_prompt(self.prompt, tile_state)
            
            # 提取结果
            masks = tile_state.get("masks", [])
            boxes = tile_state.get("boxes", [])
            scores = tile_state.get("scores", [])
            
            # 转换结果为numpy数组
            result_masks = []
            result_boxes = []
            result_scores = []
            
            for mask, box, score in zip(masks, boxes, scores):
                # 处理mask - 转换为uint8以节省内存
                mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                mask_np = (mask_np * 255).astype(np.uint8)  # 转换为uint8节省内存
                result_masks.append(mask_np)
                
                # 处理box
                if box is not None:
                    box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                    # 转换为绝对坐标
                    x_start, y_start, x_end, y_end = self.tile_position
                    box_list[0] = box_list[0] * (x_end - x_start) + x_start
                    box_list[1] = box_list[1] * (y_end - y_start) + y_start
                    box_list[2] = box_list[2] * (x_end - x_start)
                    box_list[3] = box_list[3] * (y_end - y_start)
                    result_boxes.append(box_list)
                else:
                    result_boxes.append([])
                
                # 处理score
                score_value = score.item() if torch.is_tensor(score) else score
                result_scores.append(score_value)
            
            result = {
                'tile_index': self.tile_index,
                'tile_position': self.tile_position,
                'masks': result_masks,
                'boxes': result_boxes,
                'scores': result_scores,
                'prompt': self.prompt
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Tile {self.tile_index} processing error: {str(e)}")


class TileMultiPromptProcessor(QThread):
    """处理单个分块的多提示分割线程"""
    finished = pyqtSignal(dict)  # tile_index, results dict
    progress = pyqtSignal(int, str)  # current, total
    error = pyqtSignal(str)
    
    def __init__(self, parent, tile_index, tile_image, tile_position, prompts, processor):
        super().__init__()
        self.parent = parent
        self.tile_index = tile_index
        self.tile_image = tile_image
        self.tile_position = tile_position  # (x_start, y_start, x_end, y_end)
        self.prompts = prompts  # 多提示列表，如 ["stoma", "pavement-cell", "area"]
        self.processor = processor
    
    def run(self):
        try:
            results = {}
            
            for i, prompt in enumerate(self.prompts):
                try:
                    # 更新进度
                    self.progress.emit(i, f"Processing '{prompt}'...")
                    
                    # 设置当前分块图像到 processor
                    tile_state = self.processor.set_image(self.tile_image)
                    
                    # 使用文本提示进行分割
                    self.processor.reset_all_prompts(tile_state)
                    tile_state = self.processor.set_text_prompt(prompt, tile_state)
                    
                    # 提取结果
                    masks = tile_state.get("masks", [])
                    boxes = tile_state.get("boxes", [])
                    scores = tile_state.get("scores", [])
                    
                    # 转换结果为numpy数组
                    result_masks = []
                    result_boxes = []
                    result_scores = []
                    
                    for mask, box, score in zip(masks, boxes, scores):
                        # 处理mask - 转换为uint8以节省内存
                        mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                        mask_np = (mask_np * 255).astype(np.uint8)  # 转换为uint8节省内存
                        result_masks.append(mask_np)
                        
                        # 处理box
                        if box is not None:
                            box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                            # 转换为绝对坐标
                            x_start, y_start, x_end, y_end = self.tile_position
                            box_list[0] = box_list[0] * (x_end - x_start) + x_start
                            box_list[1] = box_list[1] * (y_end - y_start) + y_start
                            box_list[2] = box_list[2] * (x_end - x_start)
                            box_list[3] = box_list[3] * (y_end - y_start)
                            result_boxes.append(box_list)
                        else:
                            result_boxes.append([])
                        
                        # 处理score
                        score_value = score.item() if torch.is_tensor(score) else score
                        result_scores.append(score_value)
                    
                    # 存储该提示的结果
                    results[prompt] = {
                        'masks': result_masks,
                        'boxes': result_boxes,
                        'scores': result_scores
                    }
                    
                except Exception as e:
                    print(f"Error processing prompt '{prompt}' for tile {self.tile_index}: {e}")
                    # 继续处理其他提示
                    results[prompt] = {
                        'masks': [],
                        'boxes': [],
                        'scores': []
                    }
            
            # 发送完整结果
            final_result = {
                'tile_index': self.tile_index,
                'tile_position': self.tile_position,
                'results': results,
                'prompts': self.prompts
            }
            
            self.finished.emit(final_result)
            
        except Exception as e:
            self.error.emit(f"Tile {self.tile_index} multi-prompt processing error: {str(e)}")


class FOVManager:
    def __init__(self, parent):
        self.parent = parent
        self.tiles = []
        self.current_tile_index = 0
        self.tile_results = {}
        self.tile_multi_results = {}  # 新增：多提示分割结果
        self.original_image = None
        self.original_array = None
        self.output_dir = None
        self.project_name = None
        self.is_processing = False
        self.merged_instance_counter = 0  # 合并后的实例计数器
        
        self.current_fov_prompt = ""

        # 新增：统一颜色映射
        self.unified_colors = []
        self._create_unified_color_map(100)  # 创建100种颜色
        
        # 新增：统一ID管理
        self.unified_cell_ids = {}
        self.next_unified_id = 0
        
        # 新增：内存管理
        self.memory_threshold = 0.8  # 80%内存使用率阈值
        self.max_cells_per_batch = 100  # 每批次最大细胞数
        
        # 新增：合并阈值调整
        self.merge_iou_threshold = 0.15  # 降低IOU阈值，用于大细胞
        self.merge_distance_threshold_ratio = 0.25  # 增加距离阈值比例，用于大细胞
        self.merge_total_score_threshold = 0.05  # 降低总分数阈值，用于大细胞
        
        # 新增：掩码存储格式选项
        self.mask_storage_format = 'sparse'  # 'sparse', 'rle', 或 'dense'
        
        # 新增：数据类型优化配置
        self.dtype_optimization = {
            'mask_dtype': np.bool_,  # 使用bool而不是float32
            'temp_dtype': np.uint8,  # 临时计算使用uint8
            'min_memory_mode': False  # 最小内存模式标志
        }
        
        # 新增：表型计算器
        self.phenotype_calculator = PhenotypeCalculator(parent.scale_factor if hasattr(parent, 'scale_factor') else 1.0)
        
        # 新增：多提示配置
        self.multi_prompts = ["stoma", "pavement-cell", "area"]  # 默认的多提示
        self.merged_multi_results = {}  # 合并后的多提示结果
        self.comprehensive_report = None  # 综合表型报告
    def save_fov_prompt(self, prompt_text):
        """保存 FOV 提示词并更新界面标签"""
        self.current_fov_prompt = prompt_text.strip()
        
        # 更新 UI 上的标签显示
        if hasattr(self.parent, 'current_fov_prompt_label'):
            display_text = self.current_fov_prompt if self.current_fov_prompt else "None"
            self.parent.current_fov_prompt_label.setText(f"Current prompt: {display_text}")
        
        # 反馈状态
        self.parent.status_label.setText(f"💾 FOV Prompt saved: '{display_text}'")

    def _create_unified_color_map(self, n_colors):
        """创建统一的颜色映射表"""
        hues = np.linspace(0, 1, n_colors, endpoint=False)
        colors = []
        
        for hue in hues:
            # 使用HSL颜色空间，固定饱和度和亮度
            rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)  # 中等亮度，较高饱和度
            colors.append(rgb)
        
        self.unified_colors = colors
        return colors
    
    def get_unified_color(self, cell_id):
        """根据细胞ID获取统一颜色"""
        if not self.unified_colors:
            self._create_unified_color_map(100)
        
        color_idx = cell_id % len(self.unified_colors)
        return self.unified_colors[color_idx]
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        memory_info = psutil.virtual_memory()
        return memory_info.percent / 100.0
    
    def _create_memory_efficient_overlay(self, img_h, img_w, cells):
        """创建内存高效的叠加层"""
        try:
            overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            
            # 按统一ID分组细胞
            unified_groups = {}
            for cell in cells:
                unified_id = cell.get('unified_id', cell['id'])
                if unified_id not in unified_groups:
                    unified_groups[unified_id] = []
                unified_groups[unified_id].append(cell)
            
            # 分批处理，避免一次性占用过多内存
            group_ids = list(unified_groups.keys())
            batch_size = max(1, len(group_ids) // 10)  # 分成10批
            
            for batch_start in range(0, len(group_ids), batch_size):
                batch_ids = group_ids[batch_start:batch_start + batch_size]
                
                # 创建批次掩码
                batch_mask = np.zeros((img_h, img_w), dtype=bool)
                
                for unified_id in batch_ids:
                    cell_group = unified_groups[unified_id]
                    for cell in cell_group:
                        mask = cell['mask']
                        if mask.any():
                            batch_mask = np.logical_or(batch_mask, mask)
                
                # 应用颜色到批次
                if batch_mask.any():
                    # 为批次中的每个组应用颜色
                    for unified_id in batch_ids:
                        color = self.get_unified_color(unified_id)
                        color_rgb = np.array(color) * 255
                        
                        cell_group = unified_groups[unified_id]
                        for cell in cell_group:
                            mask = cell['mask']
                            if mask.any():
                                # 应用颜色（uint8格式）
                                overlay[mask, :3] = color_rgb.astype(np.uint8)
                                overlay[mask, 3] = 102  # 0.4 * 255 ≈ 102
                
                # 清理临时变量
                del batch_mask
                if self._check_memory_usage() > self.memory_threshold:
                    gc.collect()
            
            return overlay
            
        except MemoryError:
            # 如果内存不足，尝试降采样显示
            print("Memory error in overlay creation, using downsampled display")
            return self._create_downsampled_overlay(img_h, img_w, cells)
    
    def _create_downsampled_overlay(self, img_h, img_w, cells, scale=0.25):
        """创建降采样的叠加层以节省内存"""
        # 计算降采样尺寸
        small_h = int(img_h * scale)
        small_w = int(img_w * scale)
        
        # 创建小尺寸叠加层
        overlay_small = np.zeros((small_h, small_w, 4), dtype=np.uint8)
        
        # 按统一ID分组细胞
        unified_groups = {}
        for cell in cells:
            unified_id = cell.get('unified_id', cell['id'])
            if unified_id not in unified_groups:
                unified_groups[unified_id] = []
            unified_groups[unified_id].append(cell)
        
        for unified_id, cell_group in unified_groups.items():
            color = self.get_unified_color(unified_id)
            color_rgb = np.array(color) * 255
            
            for cell in cell_group:
                mask = cell['mask']
                if mask.any():
                    # 降采样掩码
                    mask_small = cv2.resize(mask.astype(np.uint8), 
                                           (small_w, small_h),
                                           interpolation=cv2.INTER_NEAREST) > 0
                    
                    # 应用颜色
                    overlay_small[mask_small, :3] = color_rgb.astype(np.uint8)
                    overlay_small[mask_small, 3] = 102
        
        return overlay_small, scale
    
    def mask_to_sparse(self, mask):
        """将掩码转换为稀疏坐标格式"""
        if mask.dtype != bool:
            binary_mask = mask > 127
        else:
            binary_mask = mask
            
        y_indices, x_indices = np.where(binary_mask)
        if len(y_indices) == 0:
            return np.empty((0, 2), dtype=np.uint16), mask.shape
        return np.column_stack((y_indices, x_indices)).astype(np.uint16), mask.shape
    
    def sparse_to_mask(self, sparse_coords, shape):
        """从稀疏坐标重建掩码"""
        if len(sparse_coords) == 0:
            return np.zeros(shape, dtype=bool)
        
        mask = np.zeros(shape, dtype=bool)
        y_indices = sparse_coords[:, 0].astype(int)
        x_indices = sparse_coords[:, 1].astype(int)
        
        # 确保坐标在范围内
        valid_idx = (y_indices >= 0) & (y_indices < shape[0]) & (x_indices >= 0) & (x_indices < shape[1])
        mask[y_indices[valid_idx], x_indices[valid_idx]] = True
        
        return mask
    
    def mask_to_rle(self, mask):
        """将掩码转换为RLE编码"""
        if mask.dtype != bool:
            binary_mask = mask > 127
        else:
            binary_mask = mask
            
        pixels = binary_mask.flatten()
        rle = []
        count = 0
        prev = False
        
        for pixel in pixels:
            if pixel != prev:
                if count > 0:
                    rle.append(count)
                count = 1
                prev = pixel
            else:
                count += 1
        
        if count > 0:
            rle.append(count)
        
        return rle, mask.shape
    
    def rle_to_mask(self, rle, shape):
        """从RLE编码重建掩码"""
        mask = np.zeros(shape[0] * shape[1], dtype=bool)
        pos = 0
        value = False
        
        for count in rle:
            mask[pos:pos+count] = value
            pos += count
            value = not value
        
        return mask.reshape(shape)
    
    def mask_to_contour(self, mask):
        """将掩码转换为轮廓多边形（简化）"""
        if not np.any(mask):
            return []
        
        if mask.dtype != np.uint8:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask
            
        # 使用形态学操作平滑边界
        kernel = np.ones((3, 3), np.uint8)
        mask_smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 取最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 简化轮廓（减少点数）
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return simplified.squeeze().tolist() if len(simplified) > 0 else []
    
    def preview_tiles(self):
        """预览分块网格"""
        if self.parent.current_image is None:
            QMessageBox.warning(self.parent, "Warning", "Please upload an image first")
            return
        
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        
        # 获取分块参数
        tile_width = self.parent.tile_width_spinbox.value()
        tile_height = self.parent.tile_height_spinbox.value()
        overlap_x = self.parent.overlap_x_spinbox.value() / 100.0
        overlap_y = self.parent.overlap_y_spinbox.value() / 100.0
        
        # 保存原始图像
        self.original_image = self.parent.current_image
        self.original_array = self.parent.current_image_array
        
        # 计算自适应分块
        self.tiles = self.calculate_adaptive_tiles(tile_width, tile_height, overlap_x, overlap_y)
        
        if not self.tiles:
            QMessageBox.warning(self.parent, "Warning", "No valid tiles generated. Check tile size and overlap settings.")
            return
        
        # 在画布上显示分块网格
        self.show_tile_grid()
        
        # 更新导航信息
        self.update_tile_navigation()
        
        self.parent.status_label.setText(f"✅ Generated {len(self.tiles)} tiles. Click 'Process All Tiles' to start segmentation.")

        # 启用FOV处理按钮
        if hasattr(self.parent, 'process_fov_multi_btn'):
            self.parent.process_fov_multi_btn.setEnabled(True)
    
    def calculate_adaptive_tiles(self, tile_width, tile_height, overlap_x, overlap_y):
        """计算更合理的自适应分块，确保边缘区块大小合适且分布均匀"""
        if self.original_image is None:
            return []
        
        img_width, img_height = self.original_image.size
        
        # 确保tile尺寸不超过图像尺寸
        tile_width = min(tile_width, img_width)
        tile_height = min(tile_height, img_height)
        
        # 计算在重叠情况下的有效步长
        stride_x = max(1, int(tile_width * (1 - overlap_x)))
        stride_y = max(1, int(tile_height * (1 - overlap_y)))
        
        # 计算需要的区块数量
        num_tiles_x = max(1, int(np.ceil((img_width - tile_width) / stride_x)) + 1)
        num_tiles_y = max(1, int(np.ceil((img_height - tile_height) / stride_y)) + 1)
        
        # 重新计算步长以确保均匀分布
        if num_tiles_x > 1:
            stride_x = (img_width - tile_width) / (num_tiles_x - 1)
        if num_tiles_y > 1:
            stride_y = (img_height - tile_height) / (num_tiles_y - 1)
        
        tiles = []
        tile_index = 0
        
        # 生成均匀分布的区块
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                # 计算区块起始位置
                x = int(j * stride_x)
                y = int(i * stride_y)
                
                # 确保区块不超出图像边界
                x_end = min(x + tile_width, img_width)
                y_end = min(y + tile_height, img_height)
                
                # 调整起始位置以确保最终区块大小正确
                if x_end == img_width and j == num_tiles_x - 1 and num_tiles_x > 1:
                    x = img_width - tile_width
                
                if y_end == img_height and i == num_tiles_y - 1 and num_tiles_y > 1:
                    y = img_height - tile_height
                
                x = max(0, x)
                y = max(0, y)
                x_end = min(img_width, x + tile_width)
                y_end = min(img_height, y + tile_height)
                
                # 检查区块是否有效
                actual_width = x_end - x
                actual_height = y_end - y
                
                # 只添加足够大的区块
                if actual_width >= tile_width * 0.5 and actual_height >= tile_height * 0.5:
                    try:
                        tile = self.original_image.crop((x, y, x_end, y_end))
                        tiles.append({
                            'index': tile_index,
                            'image': tile,
                            'position': (x, y, x_end, y_end),
                            'size': (actual_width, actual_height),
                            'grid_position': (j, i)  # 记录网格位置，便于调试
                        })
                        tile_index += 1
                    except Exception as e:
                        print(f"Error cropping tile at ({x}, {y}): {e}")
        
        # 打印调试信息
        print(f"Image size: {img_width}x{img_height}")
        print(f"Tile size: {tile_width}x{tile_height}")
        print(f"Grid: {num_tiles_x}x{num_tiles_y} tiles")
        print(f"Generated {len(tiles)} valid tiles")
        
        return tiles
    
    def show_tile_grid(self):
        """
        在画布上显示分块网格
        """
        if not self.tiles or self.original_array is None:
            return
        
        # 显示原始图像
        self.parent.canvas.ax.cla()
        self.parent.canvas.ax.imshow(self.original_array)
        self.parent.canvas.ax.axis('off')
        
        # 绘制分块网格
        import matplotlib.patches as patches
        
        for tile in self.tiles:
            x, y, x_end, y_end = tile['position']
            width = x_end - x
            height = y_end - y
            
            # 判断是否为当前选中的瓦片
            is_current = (tile['index'] == self.current_tile_index)
            
            # 样式区分：当前瓦片用橙色实线，其他用绿色虚线
            edge_color = '#f59e0b' if is_current else '#10b981' # Amber for current, Emerald for others
            line_width = 2.5 if is_current else 1.5
            alpha = 0.9 if is_current else 0.6
            line_style = '-' if is_current else '--'
            
            # 绘制矩形边框
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=line_width, edgecolor=edge_color, facecolor='none',
                linestyle=line_style, alpha=alpha
            )
            self.parent.canvas.ax.add_patch(rect)
            
            # 添加分块编号 (当前选中的编号更醒目)
            text_color = 'white'
            bg_color = '#f59e0b' if is_current else '#10b981'
            
            self.parent.canvas.ax.text(
                x + width/2, y + height/2, str(tile['index'] + 1),
                color=text_color, fontsize=10, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=bg_color, alpha=0.8)
            )
        
        # 添加统计信息
        stats_text = f"Tiles: {len(self.tiles)} | Grid: {self._get_grid_dimensions()}"
        self.parent.canvas.ax.text(
            0.02, 0.02, stats_text,
            transform=self.parent.canvas.ax.transAxes,
            color='white', fontsize=9, fontweight='bold',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
        )
        
        self.parent.canvas.draw_idle()
    
    def _get_grid_dimensions(self):
        """获取网格维度信息"""
        if not self.tiles:
            return "0x0"
        
        grid_positions = [tile.get('grid_position', (0, 0)) for tile in self.tiles]
        max_x = max(pos[0] for pos in grid_positions) + 1
        max_y = max(pos[1] for pos in grid_positions) + 1
        
        return f"{max_x}x{max_y}"
    
    def clear_tile_preview(self):
        """清除分块预览"""
        self.tiles = []
        self.current_tile_index = 0
        self.tile_results = {}
        self.tile_multi_results = {}
        self.is_processing = False
        
        # 重置导航按钮
        self.parent.prev_tile_btn.setEnabled(False)
        self.parent.next_tile_btn.setEnabled(False)
        self.parent.tile_nav_label.setText("Tile: 0/0")
        self.parent.merge_results_btn.setEnabled(False)
        if hasattr(self.parent, 'merge_fov_multi_btn'):
            self.parent.merge_fov_multi_btn.setEnabled(False)
        
        # 重新显示原始图像
        if self.original_array is not None:
            self.parent.current_image_array = self.original_array
            self.parent.current_image = self.original_image
            self.parent.update_display()
        
        self.parent.status_label.setText("🗑️ Tile preview cleared")
    
    def show_tile(self, tile_index):
        """
        显示特定分块及其分割结果（支持单提示词和多提示词结果）
        """
        if not self.tiles or tile_index < 0 or tile_index >= len(self.tiles):
            return
        
        tile = self.tiles[tile_index]
        self.current_tile_index = tile_index
        
        # 1. 清空画布并显示分块原图
        self.parent.canvas.ax.cla()
        tile_array = np.array(tile['image'])
        self.parent.canvas.ax.imshow(tile_array)
        self.parent.canvas.ax.axis('off')
        
        # 2. 尝试显示分割结果
        if tile_index in self.tile_multi_results:
            multi_result = self.tile_multi_results[tile_index]
            self.display_tile_multi_result(multi_result)
            
        # 检查单提示词结果
        elif tile_index in self.tile_results:
            result = self.tile_results[tile_index]
            self.display_tile_result(result)
        
        # 3. 添加分块信息文本 (保持原有逻辑)
        info_text = f"Tile {tile_index + 1}/{len(self.tiles)}\n"
        info_text += f"Position: ({tile['position'][0]}, {tile['position'][1]})\n"
        info_text += f"Size: {tile['size'][0]}×{tile['size'][1]} px"
        
        grid_pos = tile.get('grid_position', (0, 0))
        info_text += f"\nGrid: ({grid_pos[0]}, {grid_pos[1]})"
        
        self.parent.canvas.ax.text(
            0.02, 0.98, info_text,
            transform=self.parent.canvas.ax.transAxes,
            color='white', fontsize=9, fontweight='bold',
            ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.5)
        )
        
        self.parent.canvas.draw_idle()
        self.update_tile_navigation()
    
    def display_tile_result(self, result):
        """显示分块的分割结果"""
        if not result or 'masks' not in result:
            return
        
        tile_image = result.get('tile_image')
        if tile_image:
            tile_array = np.array(tile_image)
            img_h, img_w = tile_array.shape[:2]
            
            # 创建遮罩叠加层 - 使用uint8节省内存
            mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            
            for i, mask_np in enumerate(result['masks']):
                # 为每个实例分配颜色
                color_idx = i % len(self.parent.COLORS)
                color_rgb = self.parent.COLORS[color_idx]
                color_uint8 = np.array(color_rgb) * 255
                
                # 应用遮罩
                # 注意：masks现在是uint8格式，所以使用>127而不是0.5
                binary_mask = mask_np > 127
                mask_overlay[binary_mask, :3] = color_uint8
                mask_overlay[binary_mask, 3] = 102  # 0.4 * 255 = 102
                
                # 显示实例编号
                if binary_mask.any():
                    y_indices, x_indices = np.where(binary_mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        center_x = np.mean(x_indices)
                        center_y = np.mean(y_indices)
                        
                        annotation = self.parent.canvas.ax.text(
                            center_x, center_y, f"{i+1}",
                            color='white', fontsize=8, fontweight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color_rgb, alpha=0.8)
                        )
            
            # 添加遮罩层
            self.parent.canvas.ax.imshow(mask_overlay / 255.0)

    def display_tile_multi_result(self, result):
        """
        [新增] 在单张分块上显示多提示词分割结果
        """
        if not result or 'results' not in result:
            return

        tile_image = result.get('tile_image')
        # 如果result里没有存图，就用tiles列表里的
        if tile_image is None and self.current_tile_index < len(self.tiles):
            tile_image = self.tiles[self.current_tile_index]['image']

        if tile_image:
            tile_array = np.array(tile_image)
            img_h, img_w = tile_array.shape[:2]
            
            # 创建遮罩叠加层
            mask_overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            
            # 遍历所有提示词的结果 (stoma, pavement, etc.)
            results_dict = result.get('results', {})
            
            # 定义不同模式的颜色基准
            color_offsets = {
                'stoma': 0,
                'pavement-cell': 43,
                'area': 86
            }
            
            for prompt, prompt_res in results_dict.items():
                masks = prompt_res.get('masks', [])
                if not masks:
                    continue
                    
                # 确定该提示词的颜色起始点
                base_offset = 0
                for key, offset in color_offsets.items():
                    if key in prompt:
                        base_offset = offset
                        break
                
                for i, mask_np in enumerate(masks):
                    # 颜色循环
                    color_idx = (base_offset + i) % len(self.parent.COLORS)
                    color_rgb = self.parent.COLORS[color_idx]
                    color_uint8 = np.array(color_rgb) * 255
                    
                    # 绘制掩码
                    binary_mask = mask_np > 127
                    mask_overlay[binary_mask, :3] = color_uint8
                    mask_overlay[binary_mask, 3] = 100 # 透明度
            
            # 添加叠加层
            if np.any(mask_overlay):
                self.parent.canvas.ax.imshow(mask_overlay / 255.0)
    
    def update_tile_navigation(self):
        """
        更新分块导航控件状态
        """
        # 检查按钮是否存在
        if not hasattr(self.parent, 'prev_tile_btn') or not hasattr(self.parent, 'next_tile_btn'):
            return
            
        has_tiles = len(self.tiles) > 0
        
        # 更新按钮状态
        self.parent.prev_tile_btn.setEnabled(has_tiles and self.current_tile_index > 0)
        self.parent.next_tile_btn.setEnabled(has_tiles and self.current_tile_index < len(self.tiles) - 1)
        
        # 更新标签文本
        if hasattr(self.parent, 'tile_nav_label'):
            if has_tiles:
                self.parent.tile_nav_label.setText(f"Tile: {self.current_tile_index + 1}/{len(self.tiles)}")
            else:
                self.parent.tile_nav_label.setText("Tile: 0/0")
    
    def show_previous_tile(self):
        """显示上一个分块"""
        if self.current_tile_index > 0:
            self.show_tile(self.current_tile_index - 1)
    
    def show_next_tile(self):
        """显示下一个分块"""
        if self.current_tile_index < len(self.tiles) - 1:
            self.show_tile(self.current_tile_index + 1)

    def navigate_tile(self, step):
        """
        Args:
            step: 移动步长，-1 为上一张，1 为下一张
        """
        if not self.tiles:
            return
            
        new_index = self.current_tile_index + step
        if 0 <= new_index < len(self.tiles):
            self.current_tile_index = new_index
            self.update_tile_navigation()
            
            self.show_tile(self.current_tile_index)
    
    def setup_output_directory(self):
        """设置输出目录（只在处理时调用）"""
        if self.is_processing and not hasattr(self, 'project_dir'):
            # 使用项目根目录下的 fov_results 文件夹
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / "fov_results"
            self.output_dir.mkdir(exist_ok=True)
            
            # 创建项目特定的子目录
            if self.original_image:
                image_name = Path(self.parent.image_paths[0]).stem if hasattr(self.parent, 'image_paths') and self.parent.image_paths else "large_fov"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.project_name = f"{image_name}_tiles_{timestamp}"
                self.project_dir = self.output_dir / self.project_name
                self.project_dir.mkdir(exist_ok=True)
                
                # 保存配置信息
                self.save_config()
    
    def save_config(self):
        """保存配置信息"""
        config = {
            'image_name': Path(self.parent.image_paths[0]).name if hasattr(self.parent, 'image_paths') and self.parent.image_paths else "unknown",
            'image_size': self.original_image.size,
            'tile_width': self.parent.tile_width_spinbox.value(),
            'tile_height': self.parent.tile_height_spinbox.value(),
            'overlap_x': self.parent.overlap_x_spinbox.value(),
            'overlap_y': self.parent.overlap_y_spinbox.value(),
            'num_tiles': len(self.tiles),
            'grid_dimensions': self._get_grid_dimensions(),
            'created_time': datetime.now().isoformat(),
            'mask_storage_format': self.mask_storage_format,
            'multi_prompts': self.multi_prompts if hasattr(self, 'multi_prompts') else []
        }
        
        config_file = self.project_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _detect_cut_segments(self, binary_mask, threshold=5):
        """检测掩码在分块边缘的切割段"""
        cut = {'left': [], 'right': [], 'top': [], 'bottom': []}
        h, w = binary_mask.shape
        
        # Left edge
        if w > 0:
            touching = binary_mask[:, 0]
            if np.sum(touching) > threshold:
                labeled, num = ndi.label(touching)
                for l in range(1, num + 1):
                    ys = np.where(labeled == l)[0]
                    if len(ys) > threshold:
                        cut['left'].append((int(min(ys)), int(max(ys))))
        
        # Right edge
        if w > 0:
            touching = binary_mask[:, w - 1]
            if np.sum(touching) > threshold:
                labeled, num = ndi.label(touching)
                for l in range(1, num + 1):
                    ys = np.where(labeled == l)[0]
                    if len(ys) > threshold:
                        cut['right'].append((int(min(ys)), int(max(ys))))
        
        # Top edge
        if h > 0:
            touching = binary_mask[0, :]
            if np.sum(touching) > threshold:
                labeled, num = ndi.label(touching)
                for l in range(1, num + 1):
                    xs = np.where(labeled == l)[0]
                    if len(xs) > threshold:
                        cut['top'].append((int(min(xs)), int(max(xs))))
        
        # Bottom edge
        if h > 0:
            touching = binary_mask[h - 1, :]
            if np.sum(touching) > threshold:
                labeled, num = ndi.label(touching)
                for l in range(1, num + 1):
                    xs = np.where(labeled == l)[0]
                    if len(xs) > threshold:
                        cut['bottom'].append((int(min(xs)), int(max(xs))))
        
        return cut
    
    def save_tile_result(self, result):
        """保存分块结果 - 只保存合并所需的最小数据"""
        if not self.is_processing:
            return None
        
        tile_index = result['tile_index']
        tile_position = result['tile_position']
        masks = result.get('masks', [])
        boxes = result.get('boxes', [])
        scores = result.get('scores', [])
        
        if not masks:
            return None
        
        # 为每个mask准备数据
        masks_data = []
        bboxes = []
        centers = []
        areas = []
        cut_segments_list = []
        
        tile_x, tile_y, tile_x_end, tile_y_end = tile_position
        tile_h = tile_y_end - tile_y
        tile_w = tile_x_end - tile_x
        
        for i, mask_np in enumerate(masks):
            if mask_np is None or mask_np.size == 0:
                continue
                
            # 转换为二进制掩码
            binary_mask = mask_np > 127
            
            # 根据选择的格式存储掩码
            if self.mask_storage_format == 'sparse':
                # 稀疏坐标格式
                sparse_coords, shape = self.mask_to_sparse(binary_mask)
                mask_stored = sparse_coords
                mask_shape = shape
            elif self.mask_storage_format == 'rle':
                # RLE编码格式
                rle, shape = self.mask_to_rle(binary_mask)
                mask_stored = rle
                mask_shape = shape
            else:  # 'dense'
                # 密集格式
                mask_stored = binary_mask.astype(np.uint8)
                mask_shape = binary_mask.shape
            
            # 计算几何特征
            y_indices, x_indices = np.where(binary_mask)
            if len(y_indices) == 0:
                continue
                
            x_min_local = np.min(x_indices)
            x_max_local = np.max(x_indices)
            y_min_local = np.min(y_indices)
            y_max_local = np.max(y_indices)
            
            # 转换到全局坐标
            x_min_global = int(tile_x + x_min_local)
            x_max_global = int(tile_x + x_max_local)
            y_min_global = int(tile_y + y_min_local)
            y_max_global = int(tile_y + y_max_local)
            
            center_x = (x_min_global + x_max_global) / 2
            center_y = (y_min_global + y_max_global) / 2
            area = len(y_indices)
            
            # 检测切割段
            cut_segments_local = self._detect_cut_segments(binary_mask)
            cut_segments_global = {
                'left': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['left']],
                'right': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['right']],
                'top': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['top']],
                'bottom': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['bottom']]
            }
            
            # 存储数据
            masks_data.append({
                'data': mask_stored,
                'shape': mask_shape,
                'format': self.mask_storage_format
            })
            
            bboxes.append([x_min_global, y_min_global, x_max_global, y_max_global])
            centers.append([center_x, center_y])
            areas.append(area)
            cut_segments_list.append(cut_segments_global)
        
        if not masks_data:
            return None
        
        # 准备NPZ数据
        npz_data = {
            'tile_index': tile_index,
            'tile_position': np.array(tile_position),
            'bboxes': np.array(bboxes, dtype=np.float16),
            'centers': np.array(centers, dtype=np.float16),
            'areas': np.array(areas, dtype=np.float16),
            'scores': np.array(scores[:len(masks_data)], dtype=np.float16),
            'cut_segments': cut_segments_list,  # 直接保存为Python列表
            'mask_format': self.mask_storage_format
        }
        
        # 添加掩码数据
        for i, mask_info in enumerate(masks_data):
            if self.mask_storage_format == 'sparse':
                npz_data[f'mask_{i}_sparse'] = mask_info['data']
                npz_data[f'mask_{i}_shape'] = mask_info['shape']
            elif self.mask_storage_format == 'rle':
                npz_data[f'mask_{i}_rle'] = mask_info['data']
                npz_data[f'mask_{i}_shape'] = mask_info['shape']
            else:
                npz_data[f'mask_{i}'] = mask_info['data']
        
        # 保存NPZ文件
        npz_filename = f"tile_{tile_index + 1:03d}_data.npz"
        npz_path = self.project_dir / npz_filename
        np.savez_compressed(npz_path, **npz_data)
        
        # 保存简单的JSON元数据
        json_data = {
            'tile_index': tile_index,
            'tile_position': tile_position,
            'tile_size': [tile_w, tile_h],
            'num_instances': len(masks_data),
            'prompt': result.get('prompt', ''),
            'saved_file': npz_filename,
            'saved_time': datetime.now().isoformat()
        }
        
        json_filename = f"tile_{tile_index + 1:03d}_info.json"
        json_path = self.project_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return npz_path
    
    def save_tile_multi_result(self, result):
        """保存分块多提示分割结果"""
        if not self.is_processing:
            return None
        
        tile_index = result['tile_index']
        tile_position = result['tile_position']
        prompts = result.get('prompts', [])
        results_dict = result.get('results', {})
        
        if not results_dict:
            return None
        
        # 为每个提示保存数据
        multi_data = {}
        for prompt, prompt_result in results_dict.items():
            masks = prompt_result.get('masks', [])
            boxes = prompt_result.get('boxes', [])
            scores = prompt_result.get('scores', [])
            
            if not masks:
                continue
            
            # 为每个mask准备数据
            masks_data = []
            bboxes = []
            centers = []
            areas = []
            cut_segments_list = []
            
            tile_x, tile_y, tile_x_end, tile_y_end = tile_position
            
            for i, mask_np in enumerate(masks):
                if mask_np is None or mask_np.size == 0:
                    continue
                    
                # 转换为二进制掩码
                binary_mask = mask_np > 127
                
                # 根据选择的格式存储掩码
                if self.mask_storage_format == 'sparse':
                    # 稀疏坐标格式
                    sparse_coords, shape = self.mask_to_sparse(binary_mask)
                    mask_stored = sparse_coords
                    mask_shape = shape
                elif self.mask_storage_format == 'rle':
                    # RLE编码格式
                    rle, shape = self.mask_to_rle(binary_mask)
                    mask_stored = rle
                    mask_shape = shape
                else:  # 'dense'
                    # 密集格式
                    mask_stored = binary_mask.astype(np.uint8)
                    mask_shape = binary_mask.shape
                
                # 计算几何特征
                y_indices, x_indices = np.where(binary_mask)
                if len(y_indices) == 0:
                    continue
                    
                x_min_local = np.min(x_indices)
                x_max_local = np.max(x_indices)
                y_min_local = np.min(y_indices)
                y_max_local = np.max(y_indices)
                
                # 转换到全局坐标
                x_min_global = int(tile_x + x_min_local)
                x_max_global = int(tile_x + x_max_local)
                y_min_global = int(tile_y + y_min_local)
                y_max_global = int(tile_y + y_max_local)
                
                center_x = (x_min_global + x_max_global) / 2
                center_y = (y_min_global + y_max_global) / 2
                area = len(y_indices)
                
                # 检测切割段
                cut_segments_local = self._detect_cut_segments(binary_mask)
                cut_segments_global = {
                    'left': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['left']],
                    'right': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['right']],
                    'top': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['top']],
                    'bottom': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['bottom']]
                }
                
                # 存储数据
                masks_data.append({
                    'data': mask_stored,
                    'shape': mask_shape,
                    'format': self.mask_storage_format
                })
                
                bboxes.append([x_min_global, y_min_global, x_max_global, y_max_global])
                centers.append([center_x, center_y])
                areas.append(area)
                cut_segments_list.append(cut_segments_global)
            
            if masks_data:
                # 准备NPZ数据
                npz_data = {
                    'tile_index': tile_index,
                    'tile_position': np.array(tile_position),
                    'prompt': prompt,
                    'bboxes': np.array(bboxes, dtype=np.float16),
                    'centers': np.array(centers, dtype=np.float16),
                    'areas': np.array(areas, dtype=np.float16),
                    'scores': np.array(scores[:len(masks_data)], dtype=np.float16),
                    'cut_segments': cut_segments_list,
                    'mask_format': self.mask_storage_format
                }
                
                # 添加掩码数据
                for i, mask_info in enumerate(masks_data):
                    if self.mask_storage_format == 'sparse':
                        npz_data[f'mask_{i}_sparse'] = mask_info['data']
                        npz_data[f'mask_{i}_shape'] = mask_info['shape']
                    elif self.mask_storage_format == 'rle':
                        npz_data[f'mask_{i}_rle'] = mask_info['data']
                        npz_data[f'mask_{i}_shape'] = mask_info['shape']
                    else:
                        npz_data[f'mask_{i}'] = mask_info['data']
                
                multi_data[prompt] = npz_data
        
        if not multi_data:
            return None
        
        # 保存多提示NPZ文件
        npz_filename = f"tile_{tile_index + 1:03d}_multi_data.npz"
        npz_path = self.project_dir / npz_filename
        np.savez_compressed(npz_path, **multi_data)
        
        # 保存简单的JSON元数据
        json_data = {
            'tile_index': tile_index,
            'tile_position': tile_position,
            'prompts': prompts,
            'saved_file': npz_filename,
            'saved_time': datetime.now().isoformat()
        }
        
        json_filename = f"tile_{tile_index + 1:03d}_multi_info.json"
        json_path = self.project_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return npz_path
    
    def load_tile_for_merging(self, tile_index):
        """加载分块数据用于合并"""
        if not hasattr(self, 'project_dir'):
            return None
        
        npz_path = self.project_dir / f"tile_{tile_index + 1:03d}_data.npz"
        if not npz_path.exists():
            return None
        
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            
            tile_position = tuple(npz_data['tile_position'])
            bboxes = npz_data['bboxes'].astype(np.float16)
            centers = npz_data['centers'].astype(np.float16)
            areas = npz_data['areas'].astype(np.float16)
            scores = npz_data['scores'].astype(np.float16)
            
            # 修正：cut_segments是Python列表，不需要.item()
            cut_segments = npz_data['cut_segments']
            
            # 如果是numpy数组且dtype=object，直接转换为列表
            if isinstance(cut_segments, np.ndarray) and cut_segments.dtype == object:
                cut_segments = cut_segments.tolist()
            
            mask_format = str(npz_data['mask_format']) if 'mask_format' in npz_data else 'sparse'
            
            fragments = []
            tile_x, tile_y, tile_x_end, tile_y_end = tile_position
            
            for i in range(len(bboxes)):
                # 重建掩码
                if mask_format == 'sparse':
                    mask_key = f'mask_{i}_sparse'
                    shape_key = f'mask_{i}_shape'
                    if mask_key in npz_data and shape_key in npz_data:
                        sparse_coords = npz_data[mask_key]
                        shape = tuple(npz_data[shape_key])
                        mask = self.sparse_to_mask(sparse_coords, shape)
                    else:
                        continue
                elif mask_format == 'rle':
                    mask_key = f'mask_{i}_rle'
                    shape_key = f'mask_{i}_shape'
                    if mask_key in npz_data and shape_key in npz_data:
                        rle = npz_data[mask_key].tolist()
                        shape = tuple(npz_data[shape_key])
                        mask = self.rle_to_mask(rle, shape)
                    else:
                        continue
                else:
                    mask_key = f'mask_{i}'
                    if mask_key in npz_data:
                        mask = npz_data[mask_key].astype(bool)
                    else:
                        continue
                
                # 检查是否为边缘片段
                is_edge_fragment = self._is_edge_fragment(
                    bboxes[i][0], bboxes[i][2], bboxes[i][1], bboxes[i][3],
                    tile_x, tile_x_end, tile_y, tile_y_end
                )
                
                # 修正：直接使用cut_segments列表中的对应元素
                cut_segment = cut_segments[i] if i < len(cut_segments) else {}
                
                fragment = {
                    'id': tile_index * 1000 + i,  # 全局唯一ID
                    'mask': mask,
                    'bbox': bboxes[i].tolist(),
                    'center': centers[i].tolist(),
                    'area': float(areas[i]),
                    'score': float(scores[i]),
                    'cut_segments': cut_segment,  # 直接使用列表元素
                    'tile_index': tile_index,
                    'fragment_index': i,
                    'is_edge_fragment': is_edge_fragment,
                    'tile_position': tile_position  # 添加分块位置信息
                }
                
                fragments.append(fragment)
            
            return fragments
            
        except Exception as e:
            print(f"Error loading tile {tile_index}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_tile_multi_for_merging(self, tile_index):
            """加载分块多提示数据用于合并"""
            if not hasattr(self, 'project_dir'):
                return None
            
            npz_path = self.project_dir / f"tile_{tile_index + 1:03d}_multi_data.npz"
            if not npz_path.exists():
                return None
            
            try:
                npz_data = np.load(npz_path, allow_pickle=True)
                
                # 提取所有提示
                prompts = [key for key in npz_data.keys() if not key.startswith('__')]
                
                multi_fragments = {}
                
                for prompt in prompts:
                    if prompt in ['tile_index', 'tile_position']:
                        continue
                    
                    # [Fix Start] 修复 0维数组导致的 IndexError
                    raw_data = npz_data[prompt]
                    # 如果是 numpy 0-d array (scalar)，需要用 .item() 取出其中的对象(dict)
                    if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
                        prompt_data = raw_data.item()
                    else:
                        prompt_data = raw_data
                    
                    # 安全检查：确保提取出来的是字典
                    if not isinstance(prompt_data, dict):
                        continue
                    # [Fix End]
                    
                    tile_position = tuple(prompt_data['tile_position'])
                    bboxes = prompt_data['bboxes'].astype(np.float16) if 'bboxes' in prompt_data else []
                    centers = prompt_data['centers'].astype(np.float16) if 'centers' in prompt_data else []
                    areas = prompt_data['areas'].astype(np.float16) if 'areas' in prompt_data else []
                    scores = prompt_data['scores'].astype(np.float16) if 'scores' in prompt_data else []
                    
                    cut_segments = prompt_data['cut_segments'] if 'cut_segments' in prompt_data else []
                    if isinstance(cut_segments, np.ndarray) and cut_segments.dtype == object:
                        cut_segments = cut_segments.tolist()
                    
                    mask_format = str(prompt_data['mask_format']) if 'mask_format' in prompt_data else 'sparse'
                    
                    fragments = []
                    tile_x, tile_y, tile_x_end, tile_y_end = tile_position
                    
                    for i in range(len(bboxes)):
                        # 重建掩码
                        if mask_format == 'sparse':
                            mask_key = f'mask_{i}_sparse'
                            shape_key = f'mask_{i}_shape'
                            if mask_key in prompt_data and shape_key in prompt_data:
                                sparse_coords = prompt_data[mask_key]
                                shape = tuple(prompt_data[shape_key])
                                mask = self.sparse_to_mask(sparse_coords, shape)
                            else:
                                continue
                        elif mask_format == 'rle':
                            mask_key = f'mask_{i}_rle'
                            shape_key = f'mask_{i}_shape'
                            if mask_key in prompt_data and shape_key in prompt_data:
                                rle = prompt_data[mask_key].tolist()
                                shape = tuple(prompt_data[shape_key])
                                mask = self.rle_to_mask(rle, shape)
                            else:
                                continue
                        else:
                            mask_key = f'mask_{i}'
                            if mask_key in prompt_data:
                                mask = prompt_data[mask_key].astype(bool)
                            else:
                                continue
                        
                        # 检查是否为边缘片段
                        is_edge_fragment = self._is_edge_fragment(
                            bboxes[i][0], bboxes[i][2], bboxes[i][1], bboxes[i][3],
                            tile_x, tile_x_end, tile_y, tile_y_end
                        )
                        
                        cut_segment = cut_segments[i] if i < len(cut_segments) else {}
                        
                        fragment = {
                            'id': tile_index * 1000 + i,
                            'mask': mask,
                            'bbox': bboxes[i].tolist(),
                            'center': centers[i].tolist(),
                            'area': float(areas[i]),
                            'score': float(scores[i]),
                            'cut_segments': cut_segment,
                            'tile_index': tile_index,
                            'fragment_index': i,
                            'is_edge_fragment': is_edge_fragment,
                            'tile_position': tile_position,
                            'prompt': prompt
                        }
                        
                        fragments.append(fragment)
                    
                    if fragments:
                        multi_fragments[prompt] = fragments
                
                return multi_fragments
                
            except Exception as e:
                print(f"Error loading multi-prompt tile {tile_index}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
    def process_all_tiles(self):
        """处理所有分块"""
        if not self.tiles:
            QMessageBox.warning(self.parent, "Warning", "Please generate tiles first")
            return
        
        # 1. 获取提示词 (优先使用保存的FOV提示词)
        prompt = ""
        if self.current_fov_prompt:
            prompt = self.current_fov_prompt
        elif hasattr(self.parent, 'fov_text_input'):
            prompt = self.parent.fov_text_input.text().strip()
            
        if not prompt and hasattr(self.parent, 'text_input'):
            prompt = self.parent.text_input.text().strip()

        if not prompt:
            QMessageBox.warning(self.parent, "Warning", "Please enter a segmentation prompt (and click Save)")
            return
        
        # 2. 内存检查
        memory_usage = self._check_memory_usage()
        if memory_usage > 0.7:
            reply = QMessageBox.warning(
                self.parent, "High Memory Usage",
                f"Current memory usage is {memory_usage:.1%}. Processing {len(self.tiles)} tiles may cause memory issues.\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # 3. 确认对话框
        reply = QMessageBox.question(
            self.parent, "Process All Tiles",
            f"Process {len(self.tiles)} tiles with prompt: '{prompt}'?\nThis may take a while.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 4. 初始化状态
        self.is_processing = True
        
        # [新增] 确保开始前禁用合并按钮，防止误操作
        if hasattr(self.parent, 'merge_results_btn'):
            self.parent.merge_results_btn.setEnabled(False)
            
        self.setup_output_directory()
        
        self.progress_dialog = QProgressDialog(
            "Processing tiles...", "Cancel", 0, len(self.tiles), self.parent
        )
        self.progress_dialog.setWindowTitle("Tile Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        self.tile_results = {}
        
        # 5. 循环处理
        for i, tile in enumerate(self.tiles):
            self.progress_dialog.setValue(i)
            self.progress_dialog.setLabelText(f"Processing tile {i+1}/{len(self.tiles)}")
            QCoreApplication.processEvents()
            
            if self.progress_dialog.wasCanceled():
                self.is_processing = False
                break
            
            try:
                self.process_single_tile(i, tile, prompt)
                
                if i % 10 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing tile {i}: {e}")
        
        self.progress_dialog.close()
        
        # 6. 处理完成后的逻辑
        if self.tile_results:
            self.show_tile(0)
            self.parent.status_label.setText(f"✅ Processed {len(self.tile_results)} tiles. Results saved to {self.project_dir}")
            
            if hasattr(self.parent, 'merge_results_btn'):
                self.parent.merge_results_btn.setEnabled(True)
                self.parent.merge_results_btn.style().polish(self.parent.merge_results_btn)
                
        else:
            self.parent.status_label.setText("⚠️ No tiles were processed successfully")
    
    def process_single_tile(self, tile_index, tile, prompt):
        """处理单个分块（同步版本）"""
        try:
            # 设置当前分块图像到 processor
            tile_state = self.parent.sam3_manager.processor.set_image(tile['image'])
            
            # 使用文本提示进行分割
            self.parent.sam3_manager.processor.reset_all_prompts(tile_state)
            tile_state = self.parent.sam3_manager.processor.set_text_prompt(prompt, tile_state)
            
            # 提取结果
            masks = tile_state.get("masks", [])
            boxes = tile_state.get("boxes", [])
            scores = tile_state.get("scores", [])
            
            # 转换结果为numpy数组
            result_masks = []
            result_boxes = []
            result_scores = []
            
            for mask, box, score in zip(masks, boxes, scores):
                # 处理mask - 转换为uint8以节省内存
                mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                # 转换为uint8以节省内存
                mask_np = (mask_np * 255).astype(np.uint8)
                result_masks.append(mask_np)
                
                # 处理box
                if box is not None:
                    box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                    # 转换为绝对坐标
                    x_start, y_start, x_end, y_end = tile['position']
                    box_list[0] = box_list[0] * (x_end - x_start) + x_start
                    box_list[1] = box_list[1] * (y_end - y_start) + y_start
                    box_list[2] = box_list[2] * (x_end - x_start)
                    box_list[3] = box_list[3] * (y_end - y_start)
                    result_boxes.append(box_list)
                else:
                    result_boxes.append([])
                
                # 处理score
                score_value = score.item() if torch.is_tensor(score) else score
                result_scores.append(score_value)
            
            result = {
                'tile_index': tile_index,
                'tile_image': tile['image'],  # 仅用于显示
                'tile_position': tile['position'],
                'masks': result_masks,
                'boxes': result_boxes,
                'scores': result_scores,
                'prompt': prompt
            }
            
            # 保存结果到内存
            self.tile_results[tile_index] = result
            
            # 只在处理模式下保存文件
            if self.is_processing:
                self.save_tile_result(result)
            
            # 如果当前显示的就是这个分块，更新显示
            if self.current_tile_index == tile_index:
                self.display_tile_result(result)
                self.parent.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error processing tile {tile_index}: {e}")
    
    def process_all_tiles_multi_prompt(self):
        """处理所有分块的多提示分割"""
        if not self.tiles:
            QMessageBox.warning(self.parent, "Warning", "Please generate tiles first")
            return
        
        # 检查内存
        memory_usage = self._check_memory_usage()
        if memory_usage > 0.7:
            reply = QMessageBox.warning(
                self.parent, "High Memory Usage",
                f"Current memory usage is {memory_usage:.1%}. Processing {len(self.tiles)} tiles may cause memory issues.\nContinue anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # 确认多提示分割
        reply = QMessageBox.question(
            self.parent, "Process All Tiles with Multi-Prompt",
            f"Process {len(self.tiles)} tiles with {len(self.multi_prompts)} prompts?\n"
            f"Prompts: {', '.join(self.multi_prompts)}\n"
            f"This may take a while.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 标记为处理中
        self.is_processing = True
        
        # 设置输出目录
        self.setup_output_directory()
        
        # 创建进度对话框
        total_steps = len(self.tiles) * len(self.multi_prompts)
        self.progress_dialog = QProgressDialog(
            "Processing tiles with multi-prompt...", "Cancel", 0, total_steps, self.parent
        )
        self.progress_dialog.setWindowTitle("Tile Multi-Prompt Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # 重置结果
        self.tile_multi_results = {}
        
        # 处理所有分块
        current_step = 0
        for i, tile in enumerate(self.tiles):
            if self.progress_dialog.wasCanceled():
                self.is_processing = False
                break
            
            try:
                # 处理单个分块的多提示分割
                self.process_single_tile_multi_prompt(i, tile, current_step)
                current_step += len(self.multi_prompts)
                
                # 定期清理内存
                if i % 5 == 0:
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing tile {i} with multi-prompt: {e}")
        
        self.progress_dialog.close()
        
        # 显示第一个分块的结果
        if self.tile_multi_results:
            self.show_tile(0)
            self.parent.status_label.setText(f"✅ Processed {len(self.tile_multi_results)} tiles with multi-prompt. Results saved to {self.project_dir}")
        else:
            self.parent.status_label.setText("⚠️ No tiles were processed successfully with multi-prompt")

        if hasattr(self.parent, 'merge_fov_multi_btn'):
            self.parent.merge_fov_multi_btn.setEnabled(True)
    
    def process_single_tile_multi_prompt(self, tile_index, tile, start_step):
        """处理单个分块的多提示分割（同步版本）"""
        try:
            results = {}
            
            for i, prompt in enumerate(self.multi_prompts):
                # 更新进度
                step = start_step + i
                self.progress_dialog.setValue(step)
                self.progress_dialog.setLabelText(f"Processing tile {tile_index+1}/{len(self.tiles)}\nPrompt: {prompt}")
                QCoreApplication.processEvents()
                
                if self.progress_dialog.wasCanceled():
                    return
                
                try:
                    # 设置当前分块图像到 processor
                    tile_state = self.parent.sam3_manager.processor.set_image(tile['image'])
                    
                    # 使用文本提示进行分割
                    self.parent.sam3_manager.processor.reset_all_prompts(tile_state)
                    tile_state = self.parent.sam3_manager.processor.set_text_prompt(prompt, tile_state)
                    
                    # 提取结果
                    masks = tile_state.get("masks", [])
                    boxes = tile_state.get("boxes", [])
                    scores = tile_state.get("scores", [])
                    
                    # 转换结果为numpy数组
                    result_masks = []
                    result_boxes = []
                    result_scores = []
                    
                    for mask, box, score in zip(masks, boxes, scores):
                        # 处理mask - 转换为uint8以节省内存
                        mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                        mask_np = (mask_np * 255).astype(np.uint8)
                        result_masks.append(mask_np)
                        
                        # 处理box
                        if box is not None:
                            box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                            # 转换为绝对坐标
                            x_start, y_start, x_end, y_end = tile['position']
                            box_list[0] = box_list[0] * (x_end - x_start) + x_start
                            box_list[1] = box_list[1] * (y_end - y_start) + y_start
                            box_list[2] = box_list[2] * (x_end - x_start)
                            box_list[3] = box_list[3] * (y_end - y_start)
                            result_boxes.append(box_list)
                        else:
                            result_boxes.append([])
                        
                        # 处理score
                        score_value = score.item() if torch.is_tensor(score) else score
                        result_scores.append(score_value)
                    
                    # 存储该提示的结果
                    results[prompt] = {
                        'masks': result_masks,
                        'boxes': result_boxes,
                        'scores': result_scores
                    }
                    
                except Exception as e:
                    print(f"Error processing prompt '{prompt}' for tile {tile_index}: {e}")
                    # 继续处理其他提示
                    results[prompt] = {
                        'masks': [],
                        'boxes': [],
                        'scores': []
                    }
            
            # 保存完整结果
            result = {
                'tile_index': tile_index,
                'tile_image': tile['image'],
                'tile_position': tile['position'],
                'results': results,
                'prompts': self.multi_prompts
            }
            
            # 保存结果到内存
            self.tile_multi_results[tile_index] = result
            
            # 只在处理模式下保存文件
            if self.is_processing:
                self.save_tile_multi_result(result)
            
        except Exception as e:
            print(f"Error processing tile {tile_index} with multi-prompt: {e}")
    
    def merge_tile_results_with_edge_healing(self):
        """合并所有分块结果，特别处理边缘切割的细胞（性能优化完整版）"""
        # 1. 基础检查
        if not self.tile_results and not hasattr(self, 'project_dir'):
            QMessageBox.warning(self.parent, "Warning", "No tile results to merge")
            return
        
        if self.original_array is None:
            QMessageBox.warning(self.parent, "Warning", "Original image not found")
            return
        
        # 获取原始图像尺寸
        img_h, img_w = self.original_array.shape[:2]
        
        # 2. 内存模式检查
        memory_usage = self._check_memory_usage()
        use_safe_mode = False
        if memory_usage > 0.8:
            reply = QMessageBox.warning(
                self.parent, "High Memory Usage",
                f"Current memory usage is {memory_usage:.1%}. Merging may cause memory issues.\n"
                f"Would you like to use memory-safe mode? (slower but more stable)",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            use_safe_mode = reply == QMessageBox.Yes
        else:
            # 对于大图像 (>1600万像素)，强制使用安全模式以防万一
            if img_h * img_w > 4000 * 4000: 
                use_safe_mode = True
        
        # 3. 初始化进度条
        progress = QProgressDialog("Merging with edge healing...", "Cancel", 0, 100, self.parent)
        progress.setWindowTitle("Edge-Aware Merging")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0) # 立即显示
        progress.show()
        QCoreApplication.processEvents()
        
        try:
            # === 第一步：收集所有掩码片段 ===
            progress.setValue(10)
            progress.setLabelText("Collecting mask fragments...")
            QCoreApplication.processEvents()
            
            all_fragments = []
            fragment_id = 0
            
            # 优先从文件加载（如果已保存到磁盘）
            if hasattr(self, 'project_dir'):
                tile_files = list(self.project_dir.glob("tile_*_data.npz"))
                
                for tile_file in tile_files:
                    if progress.wasCanceled(): return

                    match = re.search(r'tile_(\d+)_data\.npz', tile_file.name)
                    if match:
                        tile_index = int(match.group(1)) - 1
                        fragments = self.load_tile_for_merging(tile_index)
                        if fragments:
                            all_fragments.extend(fragments)
            
            # 如果磁盘没数据，尝试从内存加载
            if not all_fragments and self.tile_results:
                for tile_index in self.tile_results.keys():
                    if progress.wasCanceled(): return
                    
                    result = self.tile_results[tile_index]
                    tile_position = result['tile_position']
                    masks = result.get('masks', [])
                    scores = result.get('scores', [])
                    
                    tile_x, tile_y, tile_x_end, tile_y_end = tile_position
                    
                    for i, mask_np in enumerate(masks):
                        if mask_np is None or mask_np.size == 0: continue
                        
                        # 确保转换为 bool
                        binary_mask = mask_np > 127
                        
                        # 计算几何特征 (仅用于判断边缘，不用做最终输出)
                        if not np.any(binary_mask): continue
                        
                        # 检测切割段
                        cut_segments_local = self._detect_cut_segments(binary_mask)
                        cut_segments_global = {
                            'left': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['left']],
                            'right': [(int(tile_y + y1), int(tile_y + y2)) for y1, y2 in cut_segments_local['right']],
                            'top': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['top']],
                            'bottom': [(int(tile_x + x1), int(tile_x + x2)) for x1, x2 in cut_segments_local['bottom']]
                        }
                        
                        # 计算边界框以判断是否接触边缘 (优化：使用简单的行列判断)
                        rows = np.any(binary_mask, axis=1)
                        cols = np.any(binary_mask, axis=0)
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        
                        x_min_global = int(tile_x + x_min)
                        x_max_global = int(tile_x + x_max)
                        y_min_global = int(tile_y + y_min)
                        y_max_global = int(tile_y + y_max)
                        
                        is_edge_fragment = self._is_edge_fragment(
                            x_min_global, x_max_global, y_min_global, y_max_global,
                            tile_x, tile_x_end, tile_y, tile_y_end
                        )
                        
                        center_x = (x_min_global + x_max_global) / 2
                        center_y = (y_min_global + y_max_global) / 2
                        area = np.sum(binary_mask)
                        
                        fragment = {
                            'id': fragment_id,
                            'mask': binary_mask,
                            'bbox': (x_min_global, y_min_global, x_max_global, y_max_global),
                            'center': (center_x, center_y),
                            'area': area,
                            'score': scores[i] if i < len(scores) else 0.5,
                            'cut_segments': cut_segments_global,
                            'tile_index': tile_index,
                            'fragment_index': i,
                            'is_edge_fragment': is_edge_fragment,
                            'tile_position': tile_position
                        }
                        
                        all_fragments.append(fragment)
                        fragment_id += 1
            
            if not all_fragments:
                progress.close()
                QMessageBox.warning(self.parent, "Warning", "No valid fragments found")
                return

            # === 第二步：统一合并细胞 ===
            progress.setValue(30)
            progress.setLabelText(f"Merging {len(all_fragments)} fragments...")
            QCoreApplication.processEvents()
            
            if use_safe_mode:
                unified_cells = self._merge_cells_efficiently_safe(all_fragments, img_h, img_w)
            else:
                unified_cells = self._merge_cells_efficiently(all_fragments, img_h, img_w)
            
            # === 第三步：应用分水岭 (如果不是安全模式) ===
            progress.setValue(60)
            progress.setLabelText("Refining cells...")
            QCoreApplication.processEvents()
            
            if use_safe_mode:
                final_cells = unified_cells
            else:
                final_cells = self._apply_watershed_to_merged_cells(unified_cells, img_h, img_w)
            
            # 计算统计信息 (打印到控制台)
            self.calculate_merge_statistics(all_fragments, final_cells)
            
            # === 第四步：准备最终输出 (关键优化点：极速处理) ===
            progress.setValue(80)
            progress.setLabelText("Optimizing output data...")
            QCoreApplication.processEvents()
            
            merged_masks = []
            merged_boxes = []
            merged_scores = []
            
            total_cells = len(final_cells)
            
            for idx, cell in enumerate(final_cells):
                # 每处理 50 个更新一次进度，避免界面完全冻结
                if idx % 50 == 0:
                    if progress.wasCanceled(): return
                    # 进度 80-95%
                    current_progress = 80 + int(15 * idx / total_cells)
                    progress.setValue(current_progress)
                    QCoreApplication.processEvents()
                
                mask = cell['mask']
                
                # 1. 如果是稀疏表示，转换为密集掩码
                if cell.get('sparse', False):
                    sparse_array = mask
                    # 使用 image_shape 如果有，否则使用 img_h, img_w
                    shape = cell.get('image_shape', (img_h, img_w))
                    mask = np.zeros(shape, dtype=np.bool_)
                    if len(sparse_array) > 0:
                        mask[sparse_array[:, 0], sparse_array[:, 1]] = True
                
                # 2. 确保是 bool 类型 (使用 copy=False 提速)
                if mask.dtype != np.bool_:
                    mask = mask.astype(np.bool_, copy=False)
                
                merged_masks.append(mask)
                merged_scores.append(cell.get('score', 0.5))
                
                # 3. 极速计算边界框 (使用 OpenCV 替代 np.where)
                if mask.any():
                    # cv2.boundingRect 需要 uint8，转换开销很小
                    mask_uint8 = mask.astype(np.uint8)
                    x, y, w, h = cv2.boundingRect(mask_uint8)
                    # 转换为 [cx, cy, w, h] 格式
                    center_x = x + w / 2
                    center_y = y + h / 2
                    merged_boxes.append([center_x, center_y, w, h])
                else:
                    merged_boxes.append([0, 0, 0, 0])
            
            # === 第五步：更新界面数据 (一次性赋值，无 GC 干扰) ===
            progress.setLabelText("Updating interface...")
            QCoreApplication.processEvents()
            
            self.parent.instance_masks = merged_masks
            self.parent.instance_boxes = merged_boxes
            self.parent.instance_confidences = merged_scores
            
            # === 第六步：更新显示与保存 ===
            self.parent.current_image = self.original_image
            self.parent.current_image_array = self.original_array
            
            # 显示结果
            self.show_unified_merged_results_memory_safe(final_cells, use_safe_mode)
            
            # 保存到磁盘
            self.save_unified_healing_results(final_cells)
            
            # 计算表型 (可选，因为这步比较耗时，放在最后)
            if merged_masks:
                progress.setLabelText("Calculating phenotypes...")
                QCoreApplication.processEvents()
                self.calculate_merged_phenotype(merged_masks, merged_boxes, merged_scores)
            
            progress.setValue(100)
            progress.close()
            
            # === 完成反馈 ===
            edge_fragments = sum(1 for f in all_fragments if f.get('is_edge_fragment', False))
            merged_from_edges = sum(1 for cell in final_cells if cell.get('merged_from_edge_fragments', False))
            mode_text = " (Safe Mode)" if use_safe_mode else ""
            
            self.parent.status_label.setText(
                f"✅ Unified merge{mode_text}: {len(final_cells)} cells"
            )
            
            QMessageBox.information(
                self.parent,
                "Merge Results",
                f"Merge Statistics:\n"
                f"• Fragments: {len(all_fragments)}\n"
                f"• Unified Cells: {len(final_cells)}\n"
                f"• Healed Edges: {merged_from_edges}/{edge_fragments}\n"
                f"• Mode: {'Safe' if use_safe_mode else 'Standard'}"
            )
            
            # 只有在全部结束后才手动 GC 一次
            gc.collect()
            
        except MemoryError:
            progress.close()
            gc.collect()
            QMessageBox.critical(self.parent, "Memory Error", 
                               "Out of memory during merge. Try reducing tile size or using safe mode.")
        except Exception as e:
            progress.close()
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Merge failed: {str(e)}")
            
    def load_merged_multi_results_from_disk(self):
        """从磁盘加载合并结果到内存"""
        if not hasattr(self, 'project_dir') or not self.project_dir:
            return False
            
        save_path = self.project_dir / "fov_merged_results.npz"
        if not save_path.exists():
            return False
            
        try:
            print(f"Loading merged results from {save_path}...")
            loaded = np.load(save_path, allow_pickle=True)
            
            if 'data' in loaded:
                raw_data = loaded['data']
                # 处理 numpy 0-d array 封装
                if isinstance(raw_data, np.ndarray) and raw_data.ndim == 0:
                    self.merged_multi_results = raw_data.item()
                else:
                    self.merged_multi_results = raw_data
                print("Successfully restored merged results from disk.")
                return True
            return False
        except Exception as e:
            print(f"Failed to load merged results: {e}")
            return False
        
    def display_merged_multi_results(self, prompt):
        """
        显示指定提示的合并结果（修复版：恢复多彩实例显示，强制刷新视图）
        """
        # 1. 自动加载机制：检查内存，若无则从磁盘加载
        if not self.merged_multi_results or prompt not in self.merged_multi_results:
            print(f"Cache miss for '{prompt}', trying to load from disk...")
            if hasattr(self, 'load_merged_multi_results_from_disk'):
                success = self.load_merged_multi_results_from_disk()
                if not success:
                    print(f"Failed to load data for {prompt}")
                    if hasattr(self.parent, 'status_label'):
                        self.parent.status_label.setText(f"⚠️ No data found for {prompt}. Please Merge first.")
                    return

        # 2. 再次检查数据
        if prompt not in self.merged_multi_results:
            return
        
        # 3. 获取数据
        result = self.merged_multi_results[prompt]
        masks = result['masks']
        
        # 同步数据到主窗口（用于导出等功能，但不用于本次渲染）
        self.parent.instance_masks = masks
        self.parent.instance_boxes = result['boxes']
        self.parent.instance_confidences = result['scores']
        self.parent.current_prompt = prompt
        
        # 4. 定义图层显示名称
        display_names = {
            'stoma': 'Stomata (气孔)',
            'pavement-cell': 'Pavement Cells (表皮细胞)',
            'area': 'Apertures (气孔开口)'
        }
        name = display_names.get(prompt, prompt)

        # 5. 调用专用渲染函数 (恢复多彩显示)
        self._render_fov_layer_colorful(masks, name)
        
        # 更新状态栏
        self.parent.status_label.setText(f"👁️ Displaying {len(masks)} {name} with individual colors")

    def _render_fov_layer(self, masks, color, layer_name):
        """
        [新增] 专门用于渲染大视场单层结果的函数
        Args:
            masks: 掩码列表
            color: (R, G, B) 元组
            layer_name: 图层名称用于显示状态
        """
        if self.original_array is None:
            return

        print(f"Rendering {len(masks)} objects for {layer_name}...")
        
        # 1. 清理画布并显示底图
        self.parent.canvas.ax.cla()
        self.parent.canvas.ax.imshow(self.original_array)
        self.parent.canvas.ax.axis('off')
        
        if not masks:
            self.parent.canvas.draw_idle()
            return

        try:
            img_h, img_w = self.original_array.shape[:2]
            
            # 2. 创建叠加层 (使用 uint8 RGBA 以节省内存)
            # 默认使用降采样以保证流畅度，特别是对于大图
            scale = 1.0
            if img_h * img_w > 10000000: # 如果大于1000万像素，缩放显示
                scale = 0.5
            
            overlay_h = int(img_h * scale)
            overlay_w = int(img_w * scale)
            
            # 初始化全透明叠加层
            overlay = np.zeros((overlay_h, overlay_w, 4), dtype=np.uint8)
            
            # 准备颜色
            r, g, b = color
            color_array = np.array([r, g, b, 100], dtype=np.uint8) # Alpha = 100 (约40%透明度)
            
            # 3. 将所有掩码绘制到叠加层上
            # 批量处理以提高性能
            combined_mask = np.zeros((overlay_h, overlay_w), dtype=bool)
            
            for mask in masks:
                if mask is None: continue
                
                # 处理不同格式的掩码
                if scale != 1.0:
                    # 如果需要缩放
                    if mask.shape != (overlay_h, overlay_w):
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8), 
                            (overlay_w, overlay_h), 
                            interpolation=cv2.INTER_NEAREST
                        ) > 0
                        combined_mask = np.logical_or(combined_mask, mask_resized)
                    else:
                        combined_mask = np.logical_or(combined_mask, mask)
                else:
                    # 无缩放
                    combined_mask = np.logical_or(combined_mask, mask)
            
            # 4. 应用颜色
            # 仅在有掩码的地方应用颜色
            overlay[combined_mask] = color_array
            
            # 5. 显示叠加层
            self.parent.canvas.ax.imshow(overlay, extent=[0, img_w, img_h, 0])
            
            # 6. 添加状态文本
            self.parent.canvas.ax.text(
                10, 10, 
                f"Layer: {layer_name}\nCount: {len(masks)}",
                color='white', fontsize=12, fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
            )
            
            # 7. 强制刷新画布
            self.parent.canvas.draw_idle()
            
            # 更新底部状态栏
            self.parent.status_label.setText(f"✅ Displayed {len(masks)} {layer_name}")
            
        except Exception as e:
            print(f"Error rendering FOV layer: {e}")
            import traceback
            traceback.print_exc()
            self.parent.status_label.setText(f"❌ Render Error: {str(e)}")
            
    def _render_fov_layer_colorful(self, masks, layer_name):
        """
        渲染大视场图层：为不同实例赋予不同颜色 (恢复原始功能)
        """
        if self.original_array is None:
            return

        print(f"Rendering {len(masks)} colorful objects for {layer_name}...")
        
        # 1. 重置画布并显示底图
        self.parent.canvas.ax.cla()
        self.parent.canvas.ax.imshow(self.original_array)
        self.parent.canvas.ax.axis('off')
        
        if not masks:
            self.parent.canvas.draw_idle()
            return

        try:
            img_h, img_w = self.original_array.shape[:2]
            
            # 2. 内存优化：针对超大图像进行降采样显示的判断
            scale = 1.0
            # 如果图像超过 1000 万像素，使用 0.5 倍缩放的遮罩层以提升性能
            if img_h * img_w > 10000000:
                scale = 0.5
            
            overlay_h = int(img_h * scale)
            overlay_w = int(img_w * scale)
            
            # 3. 创建全透明叠加层 (RGBA)
            overlay = np.zeros((overlay_h, overlay_w, 4), dtype=np.uint8)
            
            # 检查是否有预定义的颜色列表
            colors = getattr(self.parent, 'COLORS', None)
            
            # [Fix] 安全检查 colors 是否有效
            has_colors = False
            if colors is not None:
                if isinstance(colors, np.ndarray):
                    has_colors = colors.size > 0
                else:
                    has_colors = len(colors) > 0

            if not has_colors:
                # 如果没有，生成一组默认颜色
                import colorsys
                colors = [tuple(int(c*255) for c in colorsys.hls_to_rgb(h, 0.5, 1.0)) 
                         for h in np.linspace(0, 1, 20)]

            # 4. 批量绘制每个实例
            for i, mask in enumerate(masks):
                if mask is None: continue
                
                # 获取当前实例的颜色
                color_idx = i % len(colors)
                color_rgb = colors[color_idx] # 假设是 (R, G, B) 0-255 或 0-1
                
                # 确保颜色是 0-255 格式
                if max(color_rgb) <= 1.0:
                    color_rgb = [int(c * 255) for c in color_rgb]
                
                # 处理掩码缩放
                if scale != 1.0:
                    if mask.shape != (overlay_h, overlay_w):
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8), 
                            (overlay_w, overlay_h), 
                            interpolation=cv2.INTER_NEAREST
                        ) > 0
                    else:
                        mask_resized = mask
                else:
                    mask_resized = mask

                overlay[mask_resized, 0] = color_rgb[0]
                overlay[mask_resized, 1] = color_rgb[1]
                overlay[mask_resized, 2] = color_rgb[2]
                # 赋值 Alpha (透明度)，例如 100 (约40%)
                overlay[mask_resized, 3] = 100
            
            # 5. 一次性显示叠加层
            self.parent.canvas.ax.imshow(overlay, extent=[0, img_w, img_h, 0])
            
            # 6. 添加图层信息文本
            self.parent.canvas.ax.text(
                0.02, 0.98, 
                f"Layer: {layer_name}\nCount: {len(masks)}",
                transform=self.parent.canvas.ax.transAxes,
                color='white', fontsize=10, fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.6)
            )
            
            # 7. 强制刷新
            self.parent.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error rendering colorful layer: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_merged_phenotype(self, masks, boxes, scores):
        """计算合并后的表型"""
        try:
            # 使用表型计算器计算基础属性
            phenotype_df = self.phenotype_calculator.calculate_basic_properties(masks, boxes, scores)
            
            if phenotype_df is not None and not phenotype_df.empty:
                # 更新界面显示
                total_count = len(phenotype_df)
                avg_area = phenotype_df['area_um2'].mean()
                
                info_text = f"【Merged FOV Phenotype Analysis】\n"
                info_text += "=" * 60 + "\n"
                info_text += f"📁 Class: {self.parent.current_prompt if self.parent.current_prompt else 'Unspecified'}\n"
                info_text += f"🔢 Count: {total_count}\n"
                info_text += f"📏 Average Area: {avg_area:.2f} μm²\n"
                info_text += "=" * 60 + "\n\n"
                
                # 详细列表 (前 20 个实例)
                info_text += "📊 Instance Details (Top 20):\n"
                info_text += "-" * 90 + "\n"
                info_text += f"{'ID':<4} {'Area(μm²)':<12} {'Major(μm)':<10} {'Minor(μm)':<10} {'Circ.':<8} {'Solid.':<8}\n"
                info_text += "-" * 90 + "\n"

                for idx, row in phenotype_df.head(20).iterrows():
                    info_text += (
                        f"{int(row['instance_id']):<4} "
                        f"{row['area_um2']:<12.2f} "
                        f"{row['major_axis_um']:<10.2f} "
                        f"{row['minor_axis_um']:<10.2f} "
                        f"{row['circularity']:<8.2f} "
                        f"{row['solidity']:<8.2f}\n"
                    )
                
                self.parent.phenotype_display.setText(info_text)
                
                # 保存数据用于导出
                self.current_phenotype_df = phenotype_df
                
                return True
            else:
                self.parent.phenotype_display.setText("No valid phenotype data after merging")
                return False
                
        except Exception as e:
            print(f"Error calculating merged phenotype: {e}")
            import traceback
            traceback.print_exc()
            self.parent.phenotype_display.setText(f"Error calculating phenotype: {str(e)}")
            return False
    
    def calculate_comprehensive_phenotype_multi(self):
        """计算多提示的综合表型"""
        try:
            stoma_masks = self.merged_multi_results.get("stoma", {}).get("masks", [])
            pavement_masks = self.merged_multi_results.get("pavement-cell", {}).get("masks", [])
            aperture_masks = self.merged_multi_results.get("area", {}).get("masks", [])
            
            stoma_boxes = self.merged_multi_results.get("stoma", {}).get("boxes", [])
            pavement_boxes = self.merged_multi_results.get("pavement-cell", {}).get("boxes", [])
            aperture_boxes = self.merged_multi_results.get("area", {}).get("boxes", [])
            
            stoma_confidences = self.merged_multi_results.get("stoma", {}).get("scores", [])
            pavement_confidences = self.merged_multi_results.get("pavement-cell", {}).get("scores", [])
            aperture_confidences = self.merged_multi_results.get("area", {}).get("scores", [])
            
            stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_masks, stoma_boxes, stoma_confidences)
            pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_masks, pavement_boxes, pavement_confidences)
            aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_masks, aperture_boxes, aperture_confidences) if aperture_masks else None
            
            if self.original_array is not None:
                height, width = self.original_array.shape[:2]
                image_area_px = height * width
                image_area_um2 = image_area_px * (1.0 / self.parent.scale_factor) ** 2
            else:
                image_area_um2 = 0
            
            image_name = Path(self.parent.image_paths[0]).name if hasattr(self.parent, 'image_paths') and self.parent.image_paths else "large_fov"
            image_path = self.parent.image_paths[0] if hasattr(self.parent, 'image_paths') and self.parent.image_paths else "N/A"
            
            self.comprehensive_report = self.phenotype_calculator.generate_comprehensive_report(
                stoma_df, pavement_df, aperture_df, image_area_um2, image_name, image_path
            )
            self.update_comprehensive_phenotype_display()
            
        except Exception as e:
            print(f"Error calculating comprehensive phenotype: {e}")
            import traceback
            traceback.print_exc()
    
    def update_comprehensive_phenotype_display(self):
        """更新综合表型显示"""
        if not self.comprehensive_report:
            self.parent.phenotype_display.setText("No comprehensive phenotype data available")
            return
        
        report = self.comprehensive_report
        info_text = "【COMPREHENSIVE FOV PHENOTYPE ANALYSIS】\n"
        info_text += "=" * 60 + "\n\n"
        
        image_info = report['image_info']
        info_text += f"📁 Image: {image_info['image_name']}\n"
        info_text += f"📏 Scale: {image_info['scale_factor_px_per_um']:.2f} px/μm\n"
        if image_info['image_area_um2'] > 0:
            info_text += f"📐 Image Area: {image_info['image_area_um2']:.0f} μm² ({image_info['image_area_um2']/1e6:.3f} mm²)\n"
        
        counts = report['basic_counts']
        info_text += f"\n🔢 Instance Counts:\n"
        info_text += f"  • Stomata: {counts['stomatal_count']}\n"
        info_text += f"  • Pavement Cells: {counts['pavement_cell_count']}\n"
        if 'aperture_count' in counts:
            info_text += f"  • Stomatal Apertures: {counts['aperture_count']}\n"
        
        indices = report['composite_indices']
        if 'stomatal_density_per_mm2' in indices:
            info_text += f"\n📊 Stomatal Density: {indices['stomatal_density_per_mm2']:.2f} /mm²\n"
        if 'stomatal_index_percent' in indices:
            info_text += f"📈 Stomatal Index: {indices['stomatal_index_percent']:.2f}%\n"
        
        if 'stomatal_area_mean_um2' in indices:
            info_text += f"\n🌱 Stomatal Size:\n"
            info_text += f"  • Mean Area: {indices['stomatal_area_mean_um2']:.2f} μm²\n"
            info_text += f"  • Mean Length: {indices['stomatal_length_mean_um']:.2f} μm\n"
            info_text += f"  • Mean Width: {indices['stomatal_width_mean_um']:.2f} μm\n"
            info_text += f"  • Mean Aspect Ratio: {indices['stomatal_aspect_ratio_mean']:.2f}\n"
        
        if 'aperture_area_mean_um2' in indices:
            info_text += f"\n🕳️ Stomatal Aperture:\n"
            info_text += f"  • Mean Aperture Area: {indices['aperture_area_mean_um2']:.2f} μm²\n"
            info_text += f"  • Aperture Ratio: {indices.get('aperture_ratio_mean', 0):.3f}\n"
        
        info_text += "\n💡 Tip: Use buttons to view different segmentation results\n"
        info_text += "💡 Tip: Export data for detailed analysis"
        
        self.parent.phenotype_display.setText(info_text)
    
    def export_comprehensive_data_multi(self):
        """导出多提示综合数据"""
        if not self.comprehensive_report:
            QMessageBox.warning(self.parent, "Warning", "No comprehensive data to export")
            return
        
        default_name = "fov_comprehensive_phenotype_data.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Export FOV Comprehensive Data", 
            default_name, 
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                stoma_masks = self.merged_multi_results.get("stoma", {}).get("masks", [])
                pavement_masks = self.merged_multi_results.get("pavement-cell", {}).get("masks", [])
                aperture_masks = self.merged_multi_results.get("area", {}).get("masks", [])
                
                stoma_boxes = self.merged_multi_results.get("stoma", {}).get("boxes", [])
                pavement_boxes = self.merged_multi_results.get("pavement-cell", {}).get("boxes", [])
                aperture_boxes = self.merged_multi_results.get("area", {}).get("boxes", [])
                
                stoma_confidences = self.merged_multi_results.get("stoma", {}).get("scores", [])
                pavement_confidences = self.merged_multi_results.get("pavement-cell", {}).get("scores", [])
                aperture_confidences = self.merged_multi_results.get("area", {}).get("scores", [])
                
                stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_masks, stoma_boxes, stoma_confidences)
                pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_masks, pavement_boxes, pavement_confidences)
                aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_masks, aperture_boxes, aperture_confidences) if aperture_masks else None
                
                export_path = self.phenotype_calculator.export_to_csv(
                    stoma_df, pavement_df, aperture_df, self.comprehensive_report, file_path
                )
                
                self.parent.status_label.setText(f"✅ FOV comprehensive data exported to {os.path.basename(export_path)}")
                QMessageBox.information(
                    self.parent, "Success", 
                    f"FOV comprehensive phenotype data exported successfully!\nSaved to: {export_path}"
                )
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to export FOV comprehensive data:\n{str(e)}")
    
    def set_multi_prompts(self, prompts):
        """设置多提示列表"""
        if prompts and isinstance(prompts, list):
            self.multi_prompts = prompts
            print(f"Multi-prompt set to: {prompts}")
        else:
            print("Warning: Invalid prompts provided, using default")
            self.multi_prompts = ["stoma", "pavement-cell", "area"]
    
    # ==================== 辅助方法 ====================
    
    def _is_edge_fragment(self, x_min, x_max, y_min, y_max, tile_x, tile_x_end, tile_y, tile_y_end):
        """判断片段是否在分块边缘"""
        edge_threshold = 10
        near_left_edge = abs(x_min - tile_x) < edge_threshold
        near_right_edge = abs(x_max - tile_x_end) < edge_threshold
        near_top_edge = abs(y_min - tile_y) < edge_threshold
        near_bottom_edge = abs(y_max - tile_y_end) < edge_threshold
        return near_left_edge or near_right_edge or near_top_edge or near_bottom_edge
    
    def _calculate_cut_alignment_score(self, frag1, frag2):
        """计算两个片段切割边缘的直线重合度"""
        tile_pos1 = self.tiles[frag1['tile_index']]['position']
        tile_pos2 = self.tiles[frag2['tile_index']]['position']
        
        if frag1['tile_index'] == frag2['tile_index']:
            return 0.0
        
        max_overlap_ratio = 0.0
        
        # 检查水平相邻（tile1 在 tile2 左侧）
        if tile_pos1[2] >= tile_pos2[0] and tile_pos1[0] < tile_pos2[0] and abs(tile_pos1[2] - tile_pos2[0]) > 0:
            segments1 = frag1['cut_segments']['right']
            segments2 = frag2['cut_segments']['left']
            for s1 in segments1:
                for s2 in segments2:
                    overlap_start = max(s1[0], s2[0])
                    overlap_end = min(s1[1], s2[1])
                    overlap_len = max(0, overlap_end - overlap_start)
                    min_len = min(s1[1] - s1[0], s2[1] - s2[0])
                    if min_len > 0:
                        ratio = overlap_len / min_len
                        max_overlap_ratio = max(max_overlap_ratio, ratio)
        
        # 检查水平相邻（tile1 在 tile2 右侧）
        elif tile_pos2[2] >= tile_pos1[0] and tile_pos2[0] < tile_pos1[0] and abs(tile_pos2[2] - tile_pos1[0]) > 0:
            segments1 = frag1['cut_segments']['left']
            segments2 = frag2['cut_segments']['right']
            for s1 in segments1:
                for s2 in segments2:
                    overlap_start = max(s1[0], s2[0])
                    overlap_end = min(s1[1], s2[1])
                    overlap_len = max(0, overlap_end - overlap_start)
                    min_len = min(s1[1] - s1[0], s2[1] - s2[0])
                    if min_len > 0:
                        ratio = overlap_len / min_len
                        max_overlap_ratio = max(max_overlap_ratio, ratio)
        
        # 检查垂直相邻（tile1 在 tile2 上方）
        if tile_pos1[3] >= tile_pos2[1] and tile_pos1[1] < tile_pos2[1] and abs(tile_pos1[3] - tile_pos2[1]) > 0:
            segments1 = frag1['cut_segments']['bottom']
            segments2 = frag2['cut_segments']['top']
            for s1 in segments1:
                for s2 in segments2:
                    overlap_start = max(s1[0], s2[0])
                    overlap_end = min(s1[1], s2[1])
                    overlap_len = max(0, overlap_end - overlap_start)
                    min_len = min(s1[1] - s1[0], s2[1] - s2[0])
                    if min_len > 0:
                        ratio = overlap_len / min_len
                        max_overlap_ratio = max(max_overlap_ratio, ratio)
        
        # 检查垂直相邻（tile1 在 tile2 下方）
        elif tile_pos2[3] >= tile_pos1[1] and tile_pos2[1] < tile_pos1[1] and abs(tile_pos2[3] - tile_pos1[1]) > 0:
            segments1 = frag1['cut_segments']['top']
            segments2 = frag2['cut_segments']['bottom']
            for s1 in segments1:
                for s2 in segments2:
                    overlap_start = max(s1[0], s2[0])
                    overlap_end = min(s1[1], s2[1])
                    overlap_len = max(0, overlap_end - overlap_start)
                    min_len = min(s1[1] - s1[0], s2[1] - s2[0])
                    if min_len > 0:
                        ratio = overlap_len / min_len
                        max_overlap_ratio = max(max_overlap_ratio, ratio)
        
        return max_overlap_ratio
    
    def _adjust_merge_thresholds_based_on_image_size(self, img_w, img_h):
        """根据图像尺寸调整合并阈值"""
        image_area = img_w * img_h
        
        if image_area > 10000000:  # 超大图像 (>10M像素)
            self.merge_iou_threshold = 0.25  # 提高IOU阈值
            self.merge_distance_threshold_ratio = 0.05  # 降低距离阈值比例
            self.merge_total_score_threshold = 0.20  # 提高总分数阈值
            print(f"超大图像模式: 面积={image_area/1000000:.1f}M像素, IOU阈值={self.merge_iou_threshold}, 距离阈值比例={self.merge_distance_threshold_ratio}")
        elif image_area > 4000000:  # 大图像 (>4M像素)
            self.merge_iou_threshold = 0.20
            self.merge_distance_threshold_ratio = 0.08
            self.merge_total_score_threshold = 0.15
            print(f"大图像模式: 面积={image_area/1000000:.1f}M像素, IOU阈值={self.merge_iou_threshold}, 距离阈值比例={self.merge_distance_threshold_ratio}")
        else:  # 小图像
            self.merge_iou_threshold = 0.15
            self.merge_distance_threshold_ratio = 0.10
            self.merge_total_score_threshold = 0.10
            print(f"小图像模式: 面积={image_area/1000000:.1f}M像素, IOU阈值={self.merge_iou_threshold}, 距离阈值比例={self.merge_distance_threshold_ratio}")
    
    def _estimate_typical_cell_size(self, fragments):
        """估计典型细胞尺寸"""
        if not fragments:
            return 50.0  # 默认值
        
        areas = [f['area'] for f in fragments if 'area' in f]
        if not areas:
            return 50.0
        
        # 计算细胞直径（假设细胞近似圆形）
        diameters = [2 * np.sqrt(area / np.pi) for area in areas]
        
        # 使用中位数作为典型尺寸
        typical_diameter = np.median(diameters)
        
        # 限制在合理范围内
        return max(20.0, min(200.0, typical_diameter))
    
    def _merge_cells_efficiently(self, fragments, img_h, img_w):
        """统一合并细胞碎片，为融合后的细胞创建统一标识（优化阈值）"""
        try:
            if not fragments:
                return []
            
            print(f"开始合并 {len(fragments)} 个片段，图像尺寸: {img_w}x{img_h}")
            
            # 重置统一ID管理
            self.unified_cell_ids = {}
            self.next_unified_id = 0
            
            # 根据图像尺寸调整合并阈值
            self._adjust_merge_thresholds_based_on_image_size(img_w, img_h)
            
            # 创建邻接图
            fragment_graph = nx.Graph()
            for i, fragment in enumerate(fragments):
                fragment_graph.add_node(i, fragment=fragment)
            
            # 基于空间邻近性连接片段
            centers = np.array([f['center'] for f in fragments])
            
            # 计算距离矩阵
            distances = cdist(centers, centers)
            
            # 设置合理的距离阈值（基于细胞典型尺寸）
            typical_cell_size = self._estimate_typical_cell_size(fragments)
            distance_threshold = typical_cell_size * 3.0  # 3倍细胞尺寸
            
            print(f"距离阈值: {distance_threshold:.1f}像素 (典型细胞尺寸: {typical_cell_size:.1f})")
            
            potential_edges = 0
            added_edges = 0
            
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    if distances[i, j] < distance_threshold:
                        potential_edges += 1
                        
                        # 检查是否应该合并
                        frag_i = fragments[i]
                        frag_j = fragments[j]
                        
                        # 计算切割对齐分数
                        cut_alignment = self._calculate_cut_alignment_score(frag_i, frag_j)
                        
                        # 对于大图像，提高切割对齐阈值
                        cut_alignment_threshold = 0.3 if img_w * img_h > 4000000 else 0.2
                        
                        if cut_alignment > cut_alignment_threshold:
                            # 计算其他分数
                            contour_score = self._calculate_contour_match_score(frag_i, frag_j)
                            overlap_score = self._calculate_overlap_score(frag_i, frag_j)
                            
                            # 综合评分，强调切割对齐和轮廓匹配
                            total_score = cut_alignment * 0.4 + contour_score * 0.4 + overlap_score * 0.2
                            
                            # 动态调整总分数阈值
                            total_score_threshold = max(0.15, self.merge_total_score_threshold)
                            
                            if total_score > total_score_threshold:
                                fragment_graph.add_edge(i, j, 
                                                       distance=distances[i, j],
                                                       score=total_score)
                                added_edges += 1
            
            print(f"潜在边: {potential_edges}, 实际添加边: {added_edges}")
            
            # 找到连通分量
            merged_cells = []
            
            components = list(nx.connected_components(fragment_graph))
            print(f"找到 {len(components)} 个连通分量")
            
            for comp_id, comp_indices in enumerate(components):
                if len(comp_indices) == 1:
                    # 单个片段，直接作为完整细胞
                    frag = fragments[list(comp_indices)[0]]
                    unified_cell = self._create_unified_cell_from_fragment(frag, comp_id)
                    if unified_cell is not None:
                        merged_cells.append(unified_cell)
                        
                        # 记录统一ID映射
                        for idx in comp_indices:
                            self.unified_cell_ids[idx] = comp_id
                else:
                    # 多个片段，需要合并
                    frag_list = [fragments[i] for i in comp_indices]
                    unified_cell = self._merge_fragments_into_unified_cell(frag_list, comp_id, 
                                                                          img_h, img_w)
                    if unified_cell is not None:
                        merged_cells.append(unified_cell)
                        
                        # 记录统一ID映射
                        for idx in comp_indices:
                            self.unified_cell_ids[idx] = comp_id
            
            print(f"合并 {len(fragments)} 个片段 -> {len(merged_cells)} 个统一细胞")
            return merged_cells
            
        except Exception as e:
            print(f"合并细胞时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_unified_cell_from_fragment(self, fragment, cell_id):
        """从单个片段创建统一细胞（确保掩码在原始图像中的正确位置）"""
        # 获取原始图像尺寸
        if self.original_array is not None:
            img_h, img_w = self.original_array.shape[:2]
        else:
            # 如果没有原始图像，使用默认尺寸
            return None
        
        mask = fragment['mask']
        
        # 检查掩码数据类型，如果不是bool则转换
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        
        # 获取片段在原始图像中的位置信息
        tile_x, tile_y, tile_x_end, tile_y_end = 0, 0, img_w, img_h
        
        if 'tile_position' in fragment:
            # 如果片段有分块位置信息
            tile_x, tile_y, tile_x_end, tile_y_end = fragment['tile_position']
        elif 'bbox' in fragment:
            # 如果有边界框信息
            bbox = fragment['bbox']
            tile_x, tile_y, tile_x_end, tile_y_end = bbox[0], bbox[1], bbox[2], bbox[3]
        elif 'tile_index' in fragment:
            # 如果有分块索引，尝试从分块信息获取位置
            tile_index = fragment['tile_index']
            if tile_index < len(self.tiles):
                tile_info = self.tiles[tile_index]
                tile_x, tile_y, tile_x_end, tile_y_end = tile_info['position']
        
        # 计算片段在原始图像中的区域
        frag_h, frag_w = mask.shape
        
        # 确保掩码是原始图像尺寸
        if mask.shape != (img_h, img_w):
            # 创建一个与原始图像相同尺寸的掩码
            mask_original_size = np.zeros((img_h, img_w), dtype=bool)
            
            # 计算片段在原始图像中的区域
            x_start = max(0, int(tile_x))
            y_start = max(0, int(tile_y))
            
            # 确保结束坐标不超过图像边界
            x_end = min(img_w, x_start + frag_w)
            y_end = min(img_h, y_start + frag_h)
            
            # 调整掩码尺寸以匹配实际可用区域
            actual_w = x_end - x_start
            actual_h = y_end - y_start
            
            if actual_w > 0 and actual_h > 0:
                # 调整掩码到实际可用区域
                if frag_w != actual_w or frag_h != actual_h:
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), 
                        (actual_w, actual_h),
                        interpolation=cv2.INTER_NEAREST
                    ) > 0
                else:
                    mask_resized = mask.astype(bool)
                
                # 将调整后的掩码放到正确位置
                mask_original_size[y_start:y_end, x_start:x_end] = mask_resized
                mask = mask_original_size
            else:
                # 如果区域无效，返回None
                return None
        
        return {
            'id': cell_id,
            'unified_id': cell_id,  # 统一ID
            'mask': mask,
            'contour': fragment.get('contour', []),
            'center': fragment['center'],
            'area': fragment['area'],
            'score': fragment.get('score', 0.5),
            'merged_from_edge_fragments': fragment.get('is_edge_fragment', False),
            'num_fragments': 1,
            'fragment_ids': [fragment['id']],
            'is_unified': True
        }
    
    def _merge_fragments_into_unified_cell(self, fragments, cell_id, img_h, img_w):
        """将多个片段合并为一个统一细胞（确保掩码在原始图像中的正确位置）"""
        if not fragments:
            return None
        
        # 创建合并的掩码 - 使用uint8类型以节省内存
        merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        for fragment in fragments:
            mask = fragment['mask']
            
            # 获取片段在原始图像中的位置信息
            if 'tile_position' in fragment:
                # 如果片段有分块位置信息，使用它来定位
                tile_x, tile_y, tile_x_end, tile_y_end = fragment['tile_position']
            elif 'bbox' in fragment:
                # 如果有边界框信息，使用边界框
                bbox = fragment['bbox']
                tile_x, tile_y, tile_x_end, tile_y_end = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                # 默认假设片段已经处于正确位置
                tile_x, tile_y = 0, 0
                tile_x_end, tile_y_end = mask.shape[1], mask.shape[0]
            
            # 计算片段在原始图像中的区域
            frag_h, frag_w = mask.shape
            
            # 检查掩码数据类型
            if mask.dtype != np.uint8:
                if mask.dtype == np.bool_:
                    mask_uint8 = mask.astype(np.uint8) * 255
                else:
                    mask_uint8 = mask.astype(np.uint8)
            else:
                mask_uint8 = mask
            
            # 将分块内的掩码映射到原始图像中
            if mask.shape != (img_h, img_w):
                # 创建一个与原始图像相同尺寸的掩码
                mask_original_size = np.zeros((img_h, img_w), dtype=np.uint8)
                
                # 计算片段在原始图像中的区域
                x_start = max(0, int(tile_x))
                y_start = max(0, int(tile_y))
                
                # 确保结束坐标不超过图像边界
                x_end = min(img_w, x_start + frag_w)
                y_end = min(img_h, y_start + frag_h)
                
                # 调整掩码尺寸以匹配实际可用区域
                actual_w = x_end - x_start
                actual_h = y_end - y_start
                
                if actual_w > 0 and actual_h > 0:
                    # 调整掩码到实际可用区域
                    if frag_w != actual_w or frag_h != actual_h:
                        mask_resized = cv2.resize(
                            mask_uint8, 
                            (actual_w, actual_h),
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        mask_resized = mask_uint8
                    
                    # 将调整后的掩码放到正确位置
                    mask_original_size[y_start:y_end, x_start:x_end] = mask_resized
                    mask_final = mask_original_size
                else:
                    # 如果区域无效，跳过这个片段
                    continue
            else:
                # 掩码已经是原始图像尺寸
                mask_final = mask_uint8
            
            # 使用最大值合并掩码
            merged_mask = np.maximum(merged_mask, mask_final)
        
        # 检查合并后的掩码是否有内容
        if np.sum(merged_mask) == 0:
            return None
        
        # 应用形态学操作填充间隙
        kernel = np.ones((5, 5), np.uint8)
        merged_mask_filled = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)
        merged_mask_filled = cv2.morphologyEx(merged_mask_filled, cv2.MORPH_OPEN, kernel)
        # 转换回bool
        merged_mask_filled_bool = merged_mask_filled > 127
        
        # 再次检查是否有内容
        if not np.any(merged_mask_filled_bool):
            return None
        
        # 提取轮廓
        merged_contour = self._extract_contour(merged_mask_filled_bool)
        
        # 计算中心点
        y_indices, x_indices = np.where(merged_mask_filled_bool)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return None
        
        center_x = np.mean(x_indices)
        center_y = np.mean(y_indices)
        area = np.sum(merged_mask_filled_bool)
        
        # 检查是否为边缘片段合并
        is_from_edges = any(f.get('is_edge_fragment', False) for f in fragments)
        
        # 计算平均分数
        avg_score = np.mean([f.get('score', 0.5) for f in fragments])
        
        return {
            'id': cell_id,
            'unified_id': cell_id,  # 统一ID
            'mask': merged_mask_filled_bool,
            'contour': merged_contour,
            'center': (center_x, center_y),
            'area': area,
            'score': avg_score,
            'merged_from_edge_fragments': is_from_edges,
            'num_fragments': len(fragments),
            'fragment_ids': [f['id'] for f in fragments],
            'is_unified': True
        }
    
    def _extract_contour(self, mask):
        """提取掩码的轮廓"""
        if not np.any(mask):
            return []
        
        # 使用形态学操作平滑边界
        kernel = np.ones((3, 3), np.uint8)
        # 转换为uint8进行操作
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_smoothed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 取最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 简化轮廓（减少点数）
        epsilon = 0.01 * cv2.arcLength(main_contour, True)
        simplified = cv2.approxPolyDP(main_contour, epsilon, True)
        
        return simplified.squeeze().tolist() if len(simplified) > 0 else []
    
    def _calculate_contour_match_score(self, frag1, frag2):
        """计算两个轮廓的匹配度"""
        contour1 = frag1.get('contour')
        contour2 = frag2.get('contour')
        
        if not contour1 or not contour2:
            return 0.0
        
        # 将轮廓转换为多边形
        poly1 = Polygon(contour1)
        poly2 = Polygon(contour2)
        
        if poly1.is_empty or poly2.is_empty:
            return 0.0
        
        # 计算凸包
        hull1 = poly1.convex_hull
        hull2 = poly2.convex_hull
        
        # 计算联合凸包
        union_hull = hull1.union(hull2).convex_hull
        
        # 计算原始多边形的总面积
        total_area = poly1.area + poly2.area
        
        # 计算联合凸包面积
        union_area = union_hull.area
        
        # 如果联合凸包比原始总面积大不了多少，说明它们形状匹配
        if union_area == 0:
            return 0.0
        
        ratio = total_area / union_area
        return min(1.0, ratio)
    
    def _calculate_overlap_score(self, frag1, frag2):
        """计算两个掩码的重叠度（考虑膨胀后的重叠）"""
        # 对掩码进行轻微膨胀以捕获邻近区域
        kernel = np.ones((5, 5), np.uint8)
        
        # 转换为uint8进行操作
        mask1 = frag1['mask']
        mask2 = frag2['mask']
        
        # 确保掩码是uint8类型
        if mask1.dtype == bool:
            mask1_uint8 = mask1.astype(np.uint8) * 255
        else:
            mask1_uint8 = mask1.astype(np.uint8)
            
        if mask2.dtype == bool:
            mask2_uint8 = mask2.astype(np.uint8) * 255
        else:
            mask2_uint8 = mask2.astype(np.uint8)
        
        mask1_dilated = cv2.dilate(mask1_uint8, kernel)
        mask2_dilated = cv2.dilate(mask2_uint8, kernel)
        
        # 转换回bool
        mask1_dilated_bool = mask1_dilated > 127
        mask2_dilated_bool = mask2_dilated > 127
        
        # 计算重叠
        overlap = np.logical_and(mask1_dilated_bool, mask2_dilated_bool)
        overlap_area = np.sum(overlap)
        
        # 计算重叠相对于较小细胞的比例
        min_area = min(frag1['area'], frag2['area'])
        
        if min_area == 0:
            return 0.0
        
        return overlap_area / min_area
    
    def _apply_watershed_to_merged_cells(self, cells, img_h, img_w):
        """对合并后的细胞应用分水岭算法，分离可能粘连的细胞"""
        if len(cells) <= 1:
            return cells
        
        final_cells = []
        
        for cell in cells:
            mask = cell['mask']
            
            # 检查是否为稀疏表示
            if cell.get('sparse', False):
                # 稀疏表示，需要转换为密集掩码进行分水岭
                sparse_array = mask
                
                # 只转换部分点以节省内存
                if len(sparse_array) > 500000:  # 大于50万个点
                    print(f"细胞 {cell['id']} 有太多稀疏点 ({len(sparse_array):,})，跳过分水岭")
                    final_cells.append(cell)
                    continue
                
                # 转换为密集掩码
                dense_mask = np.zeros((img_h, img_w), dtype=np.bool_)
                for y, x in sparse_array:
                    dense_mask[y, x] = True
                mask = dense_mask
            
            # 标记连通区域
            labeled_mask, num_labels = label(mask)
            
            if num_labels == 1:
                # 单个区域，直接添加
                final_cells.append(cell)
            else:
                # 多个区域，分别处理
                for label_id in range(1, num_labels + 1):
                    component_mask = (labeled_mask == label_id)
                    
                    # 检查区域大小
                    area = np.sum(component_mask)
                    if area < 10:  # 忽略太小的区域
                        continue
                    
                    # 提取轮廓
                    component_contour = self._extract_contour(component_mask)
                    
                    # 计算中心点
                    y_indices, x_indices = np.where(component_mask)
                    center_x = np.mean(x_indices) if len(x_indices) > 0 else 0
                    center_y = np.mean(y_indices) if len(y_indices) > 0 else 0
                    
                    # 继承统一ID
                    unified_id = cell.get('unified_id', cell['id'])
                    
                    new_cell = {
                        'id': len(final_cells),
                        'unified_id': unified_id,  # 保持相同的统一ID
                        'mask': component_mask,
                        'contour': component_contour,
                        'center': (center_x, center_y),
                        'area': area,
                        'score': cell['score'],
                        'merged_from_edge_fragments': cell.get('merged_from_edge_fragments', False),
                        'num_fragments': cell.get('num_fragments', 1),
                        'is_unified': True,
                        'sparse': False  # 标记为非稀疏
                    }
                    
                    final_cells.append(new_cell)
        
        return final_cells
    
    def calculate_merge_statistics(self, fragments, unified_cells):
        """计算合并统计信息"""
        print("\n=== 合并统计 ===")
        print(f"原始片段数量: {len(fragments)}")
        print(f"合并后细胞数量: {len(unified_cells)}")
        
        # 统计片段合并情况
        cells_by_fragment_count = {}
        for cell in unified_cells:
            num_frags = cell.get('num_fragments', 1)
            cells_by_fragment_count[num_frags] = cells_by_fragment_count.get(num_frags, 0) + 1
        
        print("\n按包含片段数量的细胞分布:")
        for num_frags in sorted(cells_by_fragment_count.keys()):
            count = cells_by_fragment_count[num_frags]
            print(f"  {num_frags}个片段: {count}个细胞 ({count/len(unified_cells)*100:.1f}%)")
        
        # 统计边缘修复情况
        edge_fragments = sum(1 for f in fragments if f.get('is_edge_fragment', False))
        edge_cells = sum(1 for c in unified_cells if c.get('merged_from_edge_fragments', False))
        
        print(f"\n边缘片段: {edge_fragments} ({edge_fragments/len(fragments)*100:.1f}%)")
        print(f"边缘修复细胞: {edge_cells} ({edge_cells/len(unified_cells)*100:.1f}%)")
        
        # 统计细胞尺寸
        if unified_cells:
            areas = [c['area'] for c in unified_cells if 'area' in c]
            if areas:
                avg_area = np.mean(areas)
                median_area = np.median(areas)
                min_area = np.min(areas)
                max_area = np.max(areas)
                
                print(f"\n细胞尺寸统计:")
                print(f"  平均面积: {avg_area:.1f}像素²")
                print(f"  中位数面积: {median_area:.1f}像素²")
                print(f"  最小面积: {min_area:.1f}像素²")
                print(f"  最大面积: {max_area:.1f}像素²")
                print(f"  平均直径: {2 * np.sqrt(avg_area / np.pi):.1f}像素")
    
    def show_unified_merged_results(self, cells):
        """显示统一合并结果，融合细胞使用统一颜色"""
        try:
            if self.original_array is None or not cells:
                return
            
            img_h, img_w = self.original_array.shape[:2]
            
            # 显示原始图像
            self.parent.canvas.ax.cla()
            self.parent.canvas.ax.imshow(self.original_array)
            
            # 使用内存高效的叠加层创建方法
            overlay = self._create_memory_efficient_overlay(img_h, img_w, cells)
            
            # 添加叠加层
            if isinstance(overlay, tuple):
                # 降采样版本
                overlay_small, scale = overlay
                self.parent.canvas.ax.imshow(overlay_small / 255.0, 
                                           extent=[0, img_w, img_h, 0])
            else:
                # 完整版本
                self.parent.canvas.ax.imshow(overlay / 255.0)
            
            # 添加标签（仅在细胞数量不多时）
            if len(cells) <= 200:  # 限制标签数量
                # 按统一ID分组细胞
                unified_groups = {}
                for cell in cells:
                    unified_id = cell.get('unified_id', cell['id'])
                    if unified_id not in unified_groups:
                        unified_groups[unified_id] = []
                    unified_groups[unified_id].append(cell)
                
                for unified_id, cell_group in unified_groups.items():
                    color = self.get_unified_color(unified_id)
                    
                    for cell in cell_group:
                        mask = cell['mask']
                        if mask.any():
                            y_indices, x_indices = np.where(mask)
                            center_x = np.mean(x_indices)
                            center_y = np.mean(y_indices)
                            
                            # 构建标签文本
                            label_text = f"U{unified_id}"
                            if len(cell_group) > 1:
                                label_text += f"-{cell_group.index(cell)+1}"
                            
                            self.parent.canvas.ax.text(
                                center_x, center_y, label_text,
                                color='white', fontsize=8, fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8)
                            )
            
            # 添加统计信息
            num_unified_cells = len(set(cell.get('unified_id', cell['id']) for cell in cells))
            num_total_fragments = len(cells)
            num_edge_healed = sum(1 for cell in cells if cell.get('merged_from_edge_fragments', False))
            
            stats_text = (f"Unified Cells: {num_unified_cells}\n"
                         f"Total Segments: {num_total_fragments}\n"
                         f"Edge Healed: {num_edge_healed}")
            
            self.parent.canvas.ax.text(
                0.02, 0.98, stats_text,
                transform=self.parent.canvas.ax.transAxes,
                fontsize=10, color='white', fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
            )
            
            self.parent.canvas.ax.axis('off')
            self.parent.canvas.draw_idle()
            
        except MemoryError as e:
            print(f"Memory error in unified results display: {e}")
            # 尝试降采样显示
            self.show_unified_merged_results_memory_safe(cells, True)
        except Exception as e:
            print(f"Error showing unified results: {e}")
            import traceback
            traceback.print_exc()
    
    def show_unified_merged_results_memory_safe(self, cells, use_safe_mode=False):
        """内存安全版本的统一合并结果显示"""
        try:
            if self.original_array is None or not cells:
                return
            
            img_h, img_w = self.original_array.shape[:2]
            
            # 显示原始图像
            self.parent.canvas.ax.cla()
            self.parent.canvas.ax.imshow(self.original_array)
            
            # 创建降采样的叠加层以节省内存
            scale = 0.25 if use_safe_mode else 0.5
            small_h = int(img_h * scale)
            small_w = int(img_w * scale)
            
            overlay_small = np.zeros((small_h, small_w, 4), dtype=np.uint8)
            
            # 按统一ID分组细胞
            unified_groups = {}
            for cell in cells:
                unified_id = cell.get('unified_id', cell['id'])
                if unified_id not in unified_groups:
                    unified_groups[unified_id] = []
                unified_groups[unified_id].append(cell)
            
            # 分批处理
            group_ids = list(unified_groups.keys())
            for batch_start in range(0, len(group_ids), 20):  # 每批20个组
                batch_ids = group_ids[batch_start:batch_start + 20]
                
                for unified_id in batch_ids:
                    cell_group = unified_groups[unified_id]
                    color = self.get_unified_color(unified_id)
                    color_rgb = np.array(color) * 255
                    
                    for cell in cell_group:
                        mask = cell['mask']
                        if mask.any():
                            # 降采样掩码
                            mask_small = cv2.resize(mask.astype(np.uint8), 
                                                   (small_w, small_h),
                                                   interpolation=cv2.INTER_NEAREST) > 0
                            
                            # 应用颜色
                            overlay_small[mask_small, :3] = color_rgb.astype(np.uint8)
                            overlay_small[mask_small, 3] = 102
            # 添加降采样叠加层
            self.parent.canvas.ax.imshow(overlay_small / 255.0, 
                                       extent=[0, img_w, img_h, 0])
            
            # 添加统计信息
            num_unified_cells = len(unified_groups)
            num_total_fragments = len(cells)
            num_edge_healed = sum(1 for cell in cells if cell.get('merged_from_edge_fragments', False))
            
            stats_text = (f"Unified Cells: {num_unified_cells}\n"
                         f"Total Segments: {num_total_fragments}\n"
                         f"Edge Healed: {num_edge_healed}\n"
                         f"Display: {scale*100:.0f}% scale (Memory-Safe)")
            
            self.parent.canvas.ax.text(
                0.02, 0.98, stats_text,
                transform=self.parent.canvas.ax.transAxes,
                fontsize=10, color='white', fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7)
            )
            
            self.parent.canvas.ax.axis('off')
            self.parent.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error showing memory-safe results: {e}")
            import traceback
            traceback.print_exc()
    
    def save_unified_healing_results(self, cells):
        """保存统一边缘修复结果"""
        if not self.is_processing:
            return
        
        # 按统一ID分组
        unified_groups = {}
        for cell in cells:
            unified_id = cell.get('unified_id', cell['id'])
            if unified_id not in unified_groups:
                unified_groups[unified_id] = []
            unified_groups[unified_id].append(cell)
        
        # 准备NPZ数据
        npz_data = {
            'num_unified_cells': len(unified_groups),
            'num_total_segments': len(cells),
            'num_edge_healed': sum(1 for cell in cells if cell.get('merged_from_edge_fragments', False)),
            'image_size': self.original_array.shape[:2] if self.original_array is not None else [0, 0],
            'unified_cell_ids': np.array(list(unified_groups.keys()), dtype=np.int32)
        }
        
        # 为每个统一细胞保存数据
        for i, (unified_id, cell_group) in enumerate(unified_groups.items()):
            # 合并所有片段的掩码
            combined_mask = None
            for cell in cell_group:
                if combined_mask is None:
                    combined_mask = cell['mask'].copy()
                else:
                    combined_mask = np.logical_or(combined_mask, cell['mask'])
            
            if combined_mask is not None:
                # 转换为稀疏格式保存
                sparse_coords, shape = self.mask_to_sparse(combined_mask)
                npz_data[f'cell_{unified_id}_sparse'] = sparse_coords
                npz_data[f'cell_{unified_id}_shape'] = shape
            
            # 保存细胞信息
            npz_data[f'cell_{unified_id}_info'] = np.array([{
                'unified_id': unified_id,
                'num_segments': len(cell_group),
                'center_x': np.mean([cell['center'][0] for cell in cell_group]),
                'center_y': np.mean([cell['center'][1] for cell in cell_group]),
                'total_area': sum(cell['area'] for cell in cell_group),
                'avg_score': np.mean([cell.get('score', 0.5) for cell in cell_group]),
                'is_edge_healed': any(cell.get('merged_from_edge_fragments', False) for cell in cell_group)
            }], dtype=object)
        
        # 保存到文件
        if hasattr(self, 'project_dir'):
            result_file = self.project_dir / "unified_results.npz"
            np.savez_compressed(result_file, **npz_data)
            
            # 保存颜色映射
            color_map = {}
            for unified_id in unified_groups.keys():
                color = self.get_unified_color(unified_id)
                color_map[unified_id] = [float(c) for c in color]
            
            color_file = self.project_dir / "unified_color_map.json"
            with open(color_file, 'w') as f:
                json.dump(color_map, f, indent=2)
    
    def visualize_edge_fragments(self):
        """可视化边缘片段（调试用）"""
        if not self.tile_results:
            QMessageBox.warning(self.parent, "Warning", "No tile results to visualize")
            return
        
        # 创建可视化图像
        if self.original_array is None:
            return
        
        # 降采样以节省内存
        img_h, img_w = self.original_array.shape[:2]
        scale = 0.5
        small_h = int(img_h * scale)
        small_w = int(img_w * scale)
        
        vis_image_small = cv2.resize(self.original_array, (small_w, small_h))
        
        # 绘制分块边界
        for tile in self.tiles:
            x, y, x_end, y_end = tile['position']
            x_s, y_s = int(x * scale), int(y * scale)
            x_end_s, y_end_s = int(x_end * scale), int(y_end * scale)
            cv2.rectangle(vis_image_small, (x_s, y_s), (x_end_s, y_end_s), (255, 0, 0), 2)
        
        # 显示图像
        self.parent.canvas.ax.cla()
        self.parent.canvas.ax.imshow(vis_image_small)
        self.parent.canvas.ax.axis('off')
        self.parent.canvas.draw_idle()
        
        self.parent.status_label.setText("👁️ Tile boundaries visualized in blue (50% scale for memory)")
    
    # 安全模式相关方法
    def _merge_cells_efficiently_safe(self, fragments, img_h, img_w):
        """安全模式下的细胞合并（使用稀疏表示，优化阈值）"""
        try:
            if not fragments:
                return []
            
            print(f"安全模式: 开始合并 {len(fragments)} 个片段，图像尺寸: {img_w}x{img_h}")
            
            # 检查是否需要启用最小内存模式
            memory_usage = self._check_memory_usage()
            if memory_usage > 0.7 or img_h * img_w > 20000000:  # 大于2000万像素
                self.dtype_optimization['min_memory_mode'] = True
                print(f"启用最小内存模式，图像尺寸: {img_h}x{img_w} = {img_h*img_w:,}像素")
            
            # 根据图像尺寸调整合并阈值
            self._adjust_merge_thresholds_based_on_image_size(img_w, img_h)
            
            # 重置统一ID管理
            self.unified_cell_ids = {}
            self.next_unified_id = 0
            
            # 创建邻接图（基于边界框和中心点）
            fragment_graph = nx.Graph()
            for i, fragment in enumerate(fragments):
                fragment_graph.add_node(i, fragment=fragment)
            
            # 基于边界框邻近性连接片段
            bboxes = np.array([f['bbox'] for f in fragments])
            centers = np.array([f['center'] for f in fragments])
            
            # 计算中心点距离矩阵（分块计算以节省内存）
            added_edges = 0
            batch_size = 100  # 每批处理100个片段
            num_fragments = len(fragments)
            
            for i in range(0, num_fragments, batch_size):
                batch_end = min(i + batch_size, num_fragments)
                print(f"处理片段 {i+1} 到 {batch_end}...")
                
                # 计算当前批次与其他所有片段的距离
                batch_centers = centers[i:batch_end]
                
                # 分块计算距离矩阵
                for j in range(0, num_fragments, batch_size):
                    other_end = min(j + batch_size, num_fragments)
                    other_centers = centers[j:other_end]
                    
                    # 计算距离矩阵（小批量）
                    batch_distances = cdist(batch_centers, other_centers)
                    
                    # 设置合理的距离阈值
                    typical_cell_size = self._estimate_typical_cell_size(fragments)
                    distance_threshold = typical_cell_size * 2.5  # 2.5倍细胞尺寸
                    
                    print(f"安全模式距离阈值: {distance_threshold:.1f}像素")
                    
                    # 在当前批次内查找连接
                    for local_i in range(batch_end - i):
                        global_i = i + local_i
                        for local_j in range(other_end - j):
                            global_j = j + local_j
                            
                            if global_i >= global_j:  # 避免重复计算
                                continue
                                
                            if batch_distances[local_i, local_j] < distance_threshold:
                                frag_i = fragments[global_i]
                                frag_j = fragments[global_j]
                                
                                # 检查边界框重叠
                                bbox_i = frag_i['bbox']
                                bbox_j = frag_j['bbox']
                                
                                # 计算IOU（使用整数计算避免浮点）
                                ix1 = max(bbox_i[0], bbox_j[0])
                                iy1 = max(bbox_i[1], bbox_j[1])
                                ix2 = min(bbox_i[2], bbox_j[2])
                                iy2 = min(bbox_i[3], bbox_j[3])
                                
                                if ix1 < ix2 and iy1 < iy2:
                                    # 有重叠
                                    intersection = (ix2 - ix1) * (iy2 - iy1)
                                    area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
                                    area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                                    union = area_i + area_j - intersection
                                    
                                    if union > 0:
                                        iou = intersection / union
                                        
                                        # 计算切割对齐分数
                                        cut_alignment = self._calculate_cut_alignment_score(frag_i, frag_j)
                                        
                                        # 对于大图像，提高阈值
                                        if img_w * img_w > 4000000:
                                            iou_threshold = 0.25
                                            cut_alignment_threshold = 0.4
                                        else:
                                            iou_threshold = 0.20
                                            cut_alignment_threshold = 0.3
                                        
                                        if iou > iou_threshold or cut_alignment > cut_alignment_threshold:
                                            # 综合评分 - 调整权重
                                            total_score = iou * 0.4 + cut_alignment * 0.6
                                            
                                            total_score_threshold = max(0.20, self.merge_total_score_threshold)
                                            
                                            if total_score > total_score_threshold:
                                                fragment_graph.add_edge(global_i, global_j, 
                                                                       distance=batch_distances[local_i, local_j],
                                                                       score=total_score,
                                                                       iou=iou)
                                                added_edges += 1
                
                # 清理内存
                del batch_distances
                gc.collect()
            
            print(f"安全模式: 实际添加边: {added_edges}")
            
            # 找到连通分量
            merged_cells = []
            
            # 找到连通分量
            components = list(nx.connected_components(fragment_graph))
            print(f"安全模式: 找到 {len(components)} 个连通分量")
            
            # 分块处理组件以节省内存
            cell_batch_size = 50
            for batch_start in range(0, len(components), cell_batch_size):
                batch_end = min(batch_start + cell_batch_size, len(components))
                print(f"处理组件 {batch_start+1} 到 {batch_end}...")
                
                for comp_id in range(batch_start, batch_end):
                    comp_indices = components[comp_id]
                    frag_list = [fragments[i] for i in comp_indices]
                    
                    if len(frag_list) == 1:
                        # 单个片段
                        frag = frag_list[0]
                        unified_cell = self._create_unified_cell_safe(frag, comp_id, img_h, img_w)
                    else:
                        # 多个片段
                        unified_cell = self._merge_fragments_safe(frag_list, comp_id, img_h, img_w)
                    
                    if unified_cell is not None:
                        merged_cells.append(unified_cell)
                        
                        # 记录统一ID映射
                        for idx in comp_indices:
                            self.unified_cell_ids[idx] = comp_id
                
                # 清理内存
                gc.collect()
            
            print(f"安全模式: 合并 {len(fragments)} 个片段 -> {len(merged_cells)} 个统一细胞")
            return merged_cells
            
        except Exception as e:
            print(f"安全模式合并细胞时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_unified_cell_safe(self, fragment, cell_id, img_h, img_w):
        """安全模式下从单个片段创建细胞（使用uint8/bool节省内存）"""
        # 获取原始图像尺寸
        if self.original_array is None:
            return None
        
        mask = fragment['mask']
        
        # 检查掩码数据类型，如果不是bool则转换
        if mask.dtype != np.bool_:
            # 对于大掩码，使用uint8然后转换为bool
            if mask.size > 1000000:  # 大于100万像素
                mask_uint8 = mask.astype(np.uint8)
                mask_bool = mask_uint8 > 0
                del mask_uint8
            else:
                mask_bool = mask.astype(bool)
        else:
            mask_bool = mask
        
        # 获取片段在原始图像中的位置信息
        tile_x, tile_y, tile_x_end, tile_y_end = 0, 0, img_w, img_h
        
        if 'tile_position' in fragment:
            tile_x, tile_y, tile_x_end, tile_y_end = fragment['tile_position']
        elif 'bbox' in fragment:
            bbox = fragment['bbox']
            tile_x, tile_y, tile_x_end, tile_y_end = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # 计算片段在原始图像中的区域
        frag_h, frag_w = mask_bool.shape
        
        # 确保掩码是原始图像尺寸
        if mask_bool.shape != (img_h, img_w):
            # 创建一个与原始图像相同尺寸的掩码（使用bool类型）
            mask_original_size = np.zeros((img_h, img_w), dtype=np.bool_)
            
            # 计算片段在原始图像中的区域
            x_start = max(0, int(tile_x))
            y_start = max(0, int(tile_y))
            
            # 确保结束坐标不超过图像边界
            x_end = min(img_w, x_start + frag_w)
            y_end = min(img_h, y_start + frag_h)
            
            # 调整掩码尺寸以匹配实际可用区域
            actual_w = x_end - x_start
            actual_h = y_end - y_start
            
            if actual_w > 0 and actual_h > 0:
                # 调整掩码到实际可用区域（使用最近邻插值）
                if frag_w != actual_w or frag_h != actual_h:
                    # 使用cv2.resize但要注意内存使用
                    mask_uint8 = mask_bool.astype(np.uint8) * 255
                    mask_resized_uint8 = cv2.resize(
                        mask_uint8, 
                        (actual_w, actual_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    mask_resized = mask_resized_uint8 > 127
                    del mask_uint8, mask_resized_uint8
                else:
                    mask_resized = mask_bool
                
                # 将调整后的掩码放到正确位置
                mask_original_size[y_start:y_end, x_start:x_end] = mask_resized
                mask_final = mask_original_size
            else:
                return None
        else:
            mask_final = mask_bool
        
        # 清理中间变量
        if 'mask_bool' in locals() and mask_bool is not mask_final:
            del mask_bool
        
        return {
            'id': cell_id,
            'unified_id': cell_id,
            'mask': mask_final,
            'center': fragment['center'],
            'area': fragment['area'],
            'score': fragment.get('score', 0.5),
            'merged_from_edge_fragments': fragment.get('is_edge_fragment', False),
            'num_fragments': 1,
            'is_unified': True
        }
    
    def _merge_fragments_safe(self, fragments, cell_id, img_h, img_w):
        """安全模式下合并多个片段（内存优化）"""
        if not fragments:
            return None
        
        # 创建合并的掩码 - 根据内存模式选择数据类型
        if self.dtype_optimization['min_memory_mode']:
            # 最小内存模式：使用稀疏方式存储
            merged_mask_sparse = set()
            
            for fragment in fragments:
                mask = fragment['mask']
                
                # 获取片段位置
                if 'tile_position' in fragment:
                    tile_x, tile_y, tile_x_end, tile_y_end = fragment['tile_position']
                else:
                    tile_x, tile_y = 0, 0
                    tile_x_end, tile_y_end = mask.shape[1], mask.shape[0]
                
                # 将掩码转换为坐标集合（稀疏表示）
                if mask.any():
                    y_indices, x_indices = np.where(mask)
                    
                    # 转换到全局坐标
                    global_y = y_indices + int(tile_y)
                    global_x = x_indices + int(tile_x)
                    
                    # 添加到稀疏集合
                    for y, x in zip(global_y, global_x):
                        if 0 <= y < img_h and 0 <= x < img_w:
                            merged_mask_sparse.add((y, x))
            
            # 从稀疏表示创建密集掩码（只在需要时）
            if len(merged_mask_sparse) == 0:
                return None
            
            # 如果稀疏点太多，分批处理
            if len(merged_mask_sparse) > 1000000:  # 大于100万个点
                print(f"警告: 稀疏点太多 ({len(merged_mask_sparse):,})，使用分块处理")
                return self._create_cell_from_sparse_large(merged_mask_sparse, fragments, cell_id, img_h, img_w)
            
            # 创建密集掩码
            merged_mask = np.zeros((img_h, img_w), dtype=np.bool_)
            for y, x in merged_mask_sparse:
                merged_mask[y, x] = True
            
        else:
            # 普通安全模式：使用uint8合并
            merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            
            for fragment in fragments:
                mask = fragment['mask']
                
                # 获取片段位置
                if 'tile_position' in fragment:
                    tile_x, tile_y, tile_x_end, tile_y_end = fragment['tile_position']
                else:
                    tile_x, tile_y = 0, 0
                    tile_x_end, tile_y_end = mask.shape[1], mask.shape[0]
                
                # 转换为uint8
                if mask.dtype == np.bool_:
                    mask_uint8 = mask.astype(np.uint8) * 255
                else:
                    mask_uint8 = mask.astype(np.uint8)
                
                # 计算在原始图像中的位置
                frag_h, frag_w = mask_uint8.shape
                x_start = max(0, int(tile_x))
                y_start = max(0, int(tile_y))
                x_end = min(img_w, x_start + frag_w)
                y_end = min(img_h, y_start + frag_h)
                
                actual_w = x_end - x_start
                actual_h = y_end - y_start
                
                if actual_w > 0 and actual_h > 0:
                    if frag_w != actual_w or frag_h != actual_h:
                        mask_resized = cv2.resize(mask_uint8, (actual_w, actual_h), 
                                                 interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = mask_uint8
                    
                    # 使用最大值合并
                    merged_mask[y_start:y_end, x_start:x_end] = np.maximum(
                        merged_mask[y_start:y_end, x_start:x_end], mask_resized)
                
                # 及时清理
                del mask_uint8
            
            # 转换为bool
            merged_mask = merged_mask > 127
        
        # 检查合并后的掩码是否有内容
        if isinstance(merged_mask, np.ndarray) and np.sum(merged_mask) == 0:
            return None
        
        # 计算细胞属性
        if isinstance(merged_mask, np.ndarray):
            y_indices, x_indices = np.where(merged_mask)
            center_x = np.mean(x_indices) if len(x_indices) > 0 else 0
            center_y = np.mean(y_indices) if len(y_indices) > 0 else 0
            area = len(x_indices)
        else:
            # 对于稀疏表示，需要单独计算
            return self._create_cell_from_sparse_large(merged_mask_sparse, fragments, cell_id, img_h, img_w)
        
        # 检查是否为边缘片段合并
        is_from_edges = any(f.get('is_edge_fragment', False) for f in fragments)
        
        # 计算平均分数
        avg_score = np.mean([f.get('score', 0.5) for f in fragments])
        
        return {
            'id': cell_id,
            'unified_id': cell_id,
            'mask': merged_mask,
            'center': (center_x, center_y),
            'area': area,
            'score': avg_score,
            'merged_from_edge_fragments': is_from_edges,
            'num_fragments': len(fragments),
            'is_unified': True
        }
    
    def _create_cell_from_sparse_large(self, sparse_set, fragments, cell_id, img_h, img_w):
        """从大型稀疏集合创建细胞（进一步优化内存）"""
        if not sparse_set:
            return None
        
        # 将稀疏集合转换为numpy数组并分批处理
        sparse_array = np.array(list(sparse_set))
        
        # 分批计算中心点
        batch_size = 100000  # 每批10万个点
        centers_x = []
        centers_y = []
        
        for i in range(0, len(sparse_array), batch_size):
            batch = sparse_array[i:i+batch_size]
            if len(batch) > 0:
                centers_x.append(np.mean(batch[:, 1]))
                centers_y.append(np.mean(batch[:, 0]))
        
        center_x = np.mean(centers_x) if centers_x else 0
        center_y = np.mean(centers_y) if centers_y else 0
        area = len(sparse_array)
        
        # 检查是否为边缘片段合并
        is_from_edges = any(f.get('is_edge_fragment', False) for f in fragments)
        avg_score = np.mean([f.get('score', 0.5) for f in fragments])
        
        # 创建一个特殊的细胞对象，包含稀疏表示
        return {
            'id': cell_id,
            'unified_id': cell_id,
            'mask': sparse_array,  # 存储稀疏数组而不是密集掩码
            'sparse': True,  # 标记为稀疏表示
            'center': (center_x, center_y),
            'area': area,
            'score': avg_score,
            'merged_from_edge_fragments': is_from_edges,
            'num_fragments': len(fragments),
            'is_unified': True,
            'image_shape': (img_h, img_w)  # 保存图像形状用于后续转换
        }