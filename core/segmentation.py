# file: segmentation.py
import os
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
from PyQt5.QtCore import QCoreApplication, Qt
from ui.widegets import ImageProcessor, BatchProcessor, ImageNavigationProcessor, MultiPromptProcessor, BatchMultiPromptProcessor, BatchComprehensiveExportProcessor
import torch
import pandas as pd
# 假设 phenotype_calculator.py 位于 core 文件夹下，如果是在同一目录，请改为 from phenotype_calculator import PhenotypeCalculator
from core.phenotype_calculator import PhenotypeCalculator

class SegmentationManager:
    def __init__(self, parent):
        self.parent = parent
        self.phenotype_calculator = PhenotypeCalculator(parent.scale_factor)
        
        # 存储当前的详细表型数据 (Pandas DataFrame)
        self.current_phenotype_df = None
        
        # 存储多提示分割结果
        self.stomata_results = None  # 气孔结果
        self.pavement_cell_results = None  # 表皮细胞结果
        self.aperture_results = None  # 气孔开口结果
        self.comprehensive_report = None  # 综合报告
        
        # 批量处理相关
        self.batch_multi_results = None  # 批量多提示结果
        self.batch_comprehensive_reports = []  # 批量综合报告

         # FOV相关状态
        self.fov_mode = False  # 新增：FOV模式标志
        self.fov_manager = None  # 新增：FOV管理器
        
        # 新增：FOV多提示结果
        self.fov_stomata_results = None
        self.fov_pavement_results = None
        self.fov_aperture_results = None
        self.fov_comprehensive_report = None

    # ==================== 单图上传方法 ====================
    def on_upload_image(self):
        # 检查是否在FOV处理模式下
        if hasattr(self.parent, 'fov_manager') and self.parent.fov_manager.tiles:
            reply = QMessageBox.question(
                self.parent, "Upload New Image",
                "You are currently in tile processing mode. Uploading a new image will clear all tile data.\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            
        # 重置FOV相关状态
        self.fov_mode = False
        self.fov_manager = None
        self.fov_stomata_results = None
        self.fov_pavement_results = None
        self.fov_aperture_results = None
        self.fov_comprehensive_report = None

        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent, "Select Image", 
            "", "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All Files (*.*)"
        )
        if file_path:
            try:
                self.parent.set_loading(True, "Processing image...")
                image = Image.open(file_path).convert("RGB")
                self.parent.current_image = image
                self.parent.current_image_array = np.array(image)
                self.parent.state = self.parent.sam3_manager.processor.set_image(image)
                
                # 重置所有中间状态
                self.reset_segmentation_results()
                
                # 重置多提示结果
                self.reset_multi_prompt_results()
                
                # 重置批量模式相关状态
                self.parent.batch_mode = False
                self.parent.image_paths = [file_path]
                self.parent.current_image_index = 0
                
                # 禁用跳转控件（单图模式）
                if hasattr(self.parent, 'jump_spinbox'):
                    self.parent.jump_spinbox.setEnabled(False)
                    self.parent.jump_spinbox.setMaximum(1)
                    self.parent.jump_spinbox.setValue(1)
                
                if hasattr(self.parent, 'jump_btn'):
                    self.parent.jump_btn.setEnabled(False)
                
                # 禁用批量分割按钮（单图模式）
                if hasattr(self.parent, 'batch_segment_btn'):
                    self.parent.batch_segment_btn.setEnabled(False)
                
                # 禁用批量多提示按钮
                if hasattr(self.parent, 'batch_multi_prompt_btn'):
                    self.parent.batch_multi_prompt_btn.setEnabled(False)
                
                # 更新导航信息
                if hasattr(self.parent, 'update_navigation_info'):
                    self.parent.update_navigation_info()
                
                self.parent.set_loading(False)
                self.parent.status_label.setText(f"✅ Image loaded: {image.size[0]}×{image.size[1]} pixels")
                self.parent.current_xlim = None
                self.parent.current_ylim = None
                self.parent.resize_figure()
                self.parent.update_display()
                
            except Exception as e:
                self.parent.set_loading(False)
                QMessageBox.critical(self.parent, "Error", f"Failed to load image:\n{str(e)}")

    # ==================== 上传文件夹方法 ====================
    def on_upload_folder(self):
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        
        folder_path = QFileDialog.getExistingDirectory(
            self.parent, "Select Image Folder", ""
        )
        
        if folder_path:
            try:
                # 获取所有支持的图片文件
                image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif']
                self.parent.image_paths = []
                
                for file in os.listdir(folder_path):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        self.parent.image_paths.append(os.path.join(folder_path, file))
                
                if not self.parent.image_paths:
                    QMessageBox.warning(self.parent, "Warning", "No image files found in the selected folder")
                    return
                
                # 按文件名排序
                self.parent.image_paths.sort()
                
                self.parent.image_folder = folder_path
                self.parent.current_image_index = 0
                self.parent.batch_mode = True
                self.parent.batch_results = []

                # 启用导航控件
                if hasattr(self.parent, 'jump_spinbox'):
                    self.parent.jump_spinbox.setEnabled(True)
                    self.parent.jump_spinbox.setMaximum(len(self.parent.image_paths))
                    self.parent.jump_spinbox.setValue(1)
                
                if hasattr(self.parent, 'jump_btn'):
                    self.parent.jump_btn.setEnabled(True)
                
                # 启用批量分割按钮
                if hasattr(self.parent, 'batch_segment_btn'):
                    self.parent.batch_segment_btn.setEnabled(True)
                
                # 启用批量多提示分割按钮
                if hasattr(self.parent, 'batch_multi_prompt_btn'):
                    self.parent.batch_multi_prompt_btn.setEnabled(True)
                
                # 更新导航信息
                if hasattr(self.parent, 'update_navigation_info'):
                    self.parent.update_navigation_info()
                
                # 加载第一张图片
                self.load_image_by_index(0)
                
                self.parent.status_label.setText(f"✅ Folder loaded: {len(self.parent.image_paths)} images")
                
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to load folder:\n{str(e)}")

    # ==================== 按索引加载图片 ====================
    def load_image_by_index(self, index):
        if index < 0 or index >= len(self.parent.image_paths):
            return
        
        try:
            self.parent.set_loading(True, f"Loading image {index+1}/{len(self.parent.image_paths)}...")
            
            # 更新跳转控件的值
            if hasattr(self.parent, 'jump_spinbox'):
                self.parent.jump_spinbox.setValue(index + 1)
            
            # 使用线程加载图片
            self.navigation_processor = ImageNavigationProcessor(
                self.parent.sam3_manager.processor,
                self.parent.image_paths[index],
                self.parent.scale_factor
            )
            self.navigation_processor.finished.connect(self.on_navigation_finished)
            self.navigation_processor.error.connect(self.on_navigation_error)
            self.navigation_processor.start()
            
        except Exception as e:
            self.parent.set_loading(False)
            QMessageBox.critical(self.parent, "Error", f"Failed to load image:\n{str(e)}")
    
    def on_navigation_finished(self, result):
        try:
            image = result['image']
            image_array = result['image_array']
            state = result['state']
            
            self.parent.current_image = image
            self.parent.current_image_array = image_array
            self.parent.state = state
            
            # 重置分割结果
            self.reset_segmentation_results()
            
            # 检查是否有之前的分割结果
            self.load_batch_result_for_image(result['image_path'])
            
            # 更新当前图片索引
            image_path = result['image_path']
            if self.parent.batch_mode and image_path in self.parent.image_paths:
                self.parent.current_image_index = self.parent.image_paths.index(image_path)
            
            # 更新导航标签和按钮
            if hasattr(self.parent, 'image_nav_label'):
                self.parent.image_nav_label.setText(f"Image: {self.parent.current_image_index+1}/{len(self.parent.image_paths)}")
            if hasattr(self.parent, 'prev_image_btn'):
                self.parent.prev_image_btn.setEnabled(self.parent.current_image_index > 0)
            if hasattr(self.parent, 'next_image_btn'):
                self.parent.next_image_btn.setEnabled(self.parent.current_image_index < len(self.parent.image_paths) - 1)
            
            # 检查是否有之前的多提示分割结果
            if self.batch_multi_results and self.parent.current_image_index < len(self.batch_multi_results):
                self.load_multi_prompt_results_for_image(image_path)
            
            self.parent.set_loading(False)
            self.parent.status_label.setText(f"✅ Image {self.parent.current_image_index+1}/{len(self.parent.image_paths)} loaded: {image.size[0]}×{image.size[1]} pixels")
            
            # 调整画布大小并更新显示
            self.parent.resize_figure()
            self.parent.update_display()
            
        except Exception as e:
            self.parent.set_loading(False)
            import traceback
            error_details = traceback.format_exc()
            print(f"Navigation error details:\n{error_details}")
            QMessageBox.critical(self.parent, "Error", f"Failed to process loaded image:\n{str(e)}\n\nDetails:\n{error_details}")
    
    def on_navigation_error(self, error_msg):
        self.parent.set_loading(False)
        QMessageBox.critical(self.parent, "Error", f"Failed to load image:\n{error_msg}")

    # ==================== 加载批量分割结果 ====================
    def load_batch_result_for_image(self, image_path):
        """加载特定图像的批量分割结果"""
        for result in self.parent.batch_results:
            if result['image_path'] == image_path:
                # 恢复分割结果到 parent
                self.parent.instance_masks = result.get('masks', [])
                self.parent.instance_boxes = result.get('boxes', [])
                self.parent.instance_confidences = result.get('confidences', [])
                # 旧的列表也恢复，以防 calc 失败
                self.parent.instance_areas_um2 = result.get('instance_areas_um2', [])
                self.parent.instance_centers = result.get('instance_centers', [])
                self.parent.current_prompt = result.get('prompt', "")
                
                # 重新计算表型数据以生成 DataFrame
                if self.parent.instance_masks:
                    self.on_calc_phenotype()
                
                break
        else:
            # 如果没有找到结果，重置数据
            self.parent.instance_major_axes = []
            self.parent.instance_minor_axes = []
            self.current_phenotype_df = None
            self.parent.phenotype_display.setText("No segmentation results yet.")

    # ==================== 加载多提示分割结果 ====================
    def load_multi_prompt_results_for_image(self, image_path):
        """加载特定图像的多提示分割结果"""
        if not self.batch_multi_results:
            return
        
        for idx, results in enumerate(self.batch_multi_results):
            if idx < len(self.parent.image_paths) and self.parent.image_paths[idx] == image_path:
                if results:
                    self.stomata_results = results.get("stoma", {})
                    self.pavement_cell_results = results.get("pavement-cell", {})
                    self.aperture_results = results.get("area", {})
                    
                    # === 启用标准分析Tab的按钮 ===
                    if hasattr(self.parent, 'show_stomata_btn'):
                        self.parent.show_stomata_btn.setEnabled(True)
                    if hasattr(self.parent, 'show_pavement_btn'):
                        self.parent.show_pavement_btn.setEnabled(True)
                    if hasattr(self.parent, 'show_aperture_btn'):
                        self.parent.show_aperture_btn.setEnabled(True)

                    # === 启用批量处理Tab的按钮 ===
                    if hasattr(self.parent, 'batch_show_stomata_btn'):
                        self.parent.batch_show_stomata_btn.setEnabled(True)
                    if hasattr(self.parent, 'batch_show_pavement_btn'):
                        self.parent.batch_show_pavement_btn.setEnabled(True)
                    if hasattr(self.parent, 'batch_show_aperture_btn'):
                        self.parent.batch_show_aperture_btn.setEnabled(True)
                    
                    # 计算综合表型
                    self.calculate_comprehensive_phenotype()
                    
                    # 默认显示气孔结果
                    self.display_stomata_results()
                break

    # ==================== 上一张图片 ====================
    def on_prev_image(self):
        if self.parent.batch_mode and self.parent.current_image_index > 0:
            # 保存当前图像的分割结果
            self.save_current_to_batch_results()
            self.load_image_by_index(self.parent.current_image_index - 1)

    # ==================== 下一张图片 ====================
    def on_next_image(self):
        if self.parent.batch_mode and self.parent.current_image_index < len(self.parent.image_paths) - 1:
            # 保存当前图像的分割结果
            self.save_current_to_batch_results()
            self.load_image_by_index(self.parent.current_image_index + 1)

    # ==================== 批量分割所有图片 ====================
    def on_batch_segment(self):
        if not self.parent.batch_mode or not self.parent.image_paths:
            QMessageBox.warning(self.parent, "Warning", "Please upload a folder first")
            return
        
        # 优先读取 Batch Tab 下的 batch_text_input
        prompt = ""
        if hasattr(self.parent, 'batch_text_input'):
            prompt = self.parent.batch_text_input.text().strip()
        
        # 如果批量输入框为空，尝试读取标准输入框
        if not prompt:
            prompt = self.parent.text_input.text().strip()

        if not prompt:
            QMessageBox.warning(self.parent, "Warning", "Please enter a prompt for batch segmentation in the input box.")
            return
        
        reply = QMessageBox.question(
            self.parent, "Batch Segmentation",
            f"Segment all {len(self.parent.image_paths)} images with prompt: '{prompt}'?\nThis may take a while.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 创建进度对话框
        self.batch_progress = QProgressDialog("Batch processing images...", "Cancel", 0, len(self.parent.image_paths), self.parent)
        self.batch_progress.setWindowTitle("Batch Segmentation")
        self.batch_progress.setWindowModality(Qt.WindowModal)
        self.batch_progress.setCancelButton(None)  # 暂时禁用取消按钮
        self.batch_progress.show()
        
        self.parent.batch_results = []
        self.parent.current_prompt = prompt
        
        # 创建批量处理器
        self.batch_processor = BatchProcessor(
            self.parent.sam3_manager.processor,
            self.parent.image_paths,
            prompt,
            self.parent.scale_factor
        )
        # 询问是否保存可视化图像
        save_vis = False
        vis_output_dir = ""
        
        msg_box = QMessageBox(self.parent)
        msg_box.setWindowTitle("Batch Visualization")
        msg_box.setText("Do you want to save visualization images for each processed file?")
        msg_box.setInformativeText("This will save images with masks, labels, and scientific overlays matching current settings.")
        msg_box.addButton("Yes (Save Images)", QMessageBox.YesRole)
        msg_box.addButton("No (Data Only)", QMessageBox.NoRole)
        
        if msg_box.exec_() == 0: # Yes
            save_vis = True
            vis_output_dir = QFileDialog.getExistingDirectory(
                self.parent, "Select Directory to Save Visualization Images"
            )
            if not vis_output_dir:
                return # 用户取消了选择文件夹
        
        # 收集当前界面的可视化配置
        vis_settings = {
            'show_boxes': self.parent.show_boxes_checkbox.isChecked() if hasattr(self.parent, 'show_boxes_checkbox') else False,
            'label_mode': self.parent.show_labels_combo.currentIndex() if hasattr(self.parent, 'show_labels_combo') else 2,
            'font_size': self.parent.label_font_size,
            'explain_mode': self.parent.explain_metric_combo.currentText() if hasattr(self.parent, 'explain_metric_combo') else "None",
            'confidence_threshold': self.parent.confidence_slider.value() / 100.0,
            'scale_factor': self.parent.scale_factor
        }

        # 创建进度对话框
        self.batch_progress = QProgressDialog("Batch processing images...", "Cancel", 0, len(self.parent.image_paths), self.parent)
        # ... [保留进度条设置] ...
        
        self.parent.batch_results = []
        self.parent.current_prompt = prompt
        
        # [修改] 创建批量处理器，传入可视化参数
        self.batch_processor = BatchProcessor(
            self.parent.sam3_manager.processor,
            self.parent.image_paths,
            prompt,
            self.parent.scale_factor,
            save_vis=save_vis,          # 新增参数
            vis_output_dir=vis_output_dir, # 新增参数
            vis_settings=vis_settings   # 新增参数
        )

        # 连接信号
        self.batch_processor.progress.connect(self.on_batch_progress)
        self.batch_processor.finished.connect(self.on_batch_finished)
        self.batch_processor.error.connect(self.on_batch_error)
        
        # 开始批量处理
        self.batch_processor.start()
    
    def on_batch_progress(self, index, filename):
        # 更新进度
        if self.batch_progress:
            self.batch_progress.setValue(index)
            self.batch_progress.setLabelText(f"Processing {filename} ({index+1}/{len(self.parent.image_paths)})")
            QCoreApplication.processEvents()
    
    def on_batch_finished(self, results):
        # 关闭进度对话框
        if self.batch_progress:
            self.batch_progress.close()
        
        self.parent.batch_results = results
        
        # 更新当前显示的图片的分割结果
        if self.parent.batch_mode and self.parent.current_image_index < len(self.parent.image_paths):
            current_image_path = self.parent.image_paths[self.parent.current_image_index]
            self.load_batch_result_for_image(current_image_path)
        
        # 更新导航信息
        if hasattr(self.parent, 'update_navigation_info'):
            self.parent.update_navigation_info()
        
        # 更新显示
        self.parent.update_display()
        
        # 统计成功处理的图片数量
        success_count = sum(1 for r in results if r.get('instance_count', 0) > 0)
        error_count = sum(1 for r in results if r.get('error'))
        
        self.parent.status_label.setText(f"✅ Batch segmentation complete: {success_count} images processed")
        
        if error_count > 0:
            QMessageBox.warning(self.parent, "Batch Complete", 
                f"Batch segmentation completed with {error_count} errors.\nSuccessfully processed {success_count} images.")
        else:
            QMessageBox.information(self.parent, "Success", 
                f"Batch segmentation completed!\nSuccessfully processed {len(results)} images.")
    
    def on_batch_error(self, error_msg):
        # 关闭进度对话框
        if self.batch_progress:
            self.batch_progress.close()
        
        QMessageBox.critical(self.parent, "Error", f"Batch processing failed:\n{error_msg}")

    # ==================== 文本提示分割方法 ====================
    def on_text_prompt(self):
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        if self.parent.state is None:
            QMessageBox.warning(self.parent, "Warning", "Please upload an image first")
            return
        prompt = self.parent.text_input.text().strip()
        if not prompt:
            QMessageBox.warning(self.parent, "Warning", "Please enter a prompt")
            return
        self.parent.current_prompt = prompt
        self.parent.set_loading(True, f'Segmenting: "{prompt}"...')
        try:
            self.parent.sam3_manager.processor.reset_all_prompts(self.parent.state)
            self.parent.state = self.parent.sam3_manager.processor.set_text_prompt(prompt, self.parent.state)
            
            # 保存中间状态：从state中提取masks, boxes, scores
            if "masks" in self.parent.state:
                masks = self.parent.state.get("masks", [])
                boxes = self.parent.state.get("boxes", [])
                scores = self.parent.state.get("scores", [])
                
                # 清空之前的中间状态
                self.parent.instance_masks = []
                self.parent.instance_boxes = []
                self.parent.instance_confidences = []
                
                # 保存到中间状态变量中
                for mask, box, score in zip(masks, boxes, scores):
                    # 处理mask
                    mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                    self.parent.instance_masks.append(mask_np)
                    
                    # 处理box
                    if box is not None:
                        box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                        self.parent.instance_boxes.append(box_list)
                    else:
                        self.parent.instance_boxes.append([])
                    
                    # 处理score
                    score_value = score.item() if torch.is_tensor(score) else score
                    self.parent.instance_confidences.append(score_value)
            
            # 重置轴线数据
            self.parent.instance_major_axes = []
            self.parent.instance_minor_axes = []
            self.parent.clear_axis_lines()
            
            self.parent.update_display()
            self.parent.status_label.setText(f"✅ Text segmentation complete")
            self.save_current_to_batch_results()
            self.on_calc_phenotype()
        except Exception as e:
            self.parent.set_loading(False)
            QMessageBox.critical(self.parent, "Error", f"Segmentation failed: {str(e)}")
    
    def save_current_to_batch_results(self):
        """将当前图片的分割结果保存到批量结果中"""
        if not self.parent.batch_mode or self.parent.current_image_index >= len(self.parent.image_paths):
            return
        
        current_path = self.parent.image_paths[self.parent.current_image_index]
        
        # 查找是否已存在该图片的结果
        existing_index = -1
        for i, result in enumerate(self.parent.batch_results):
            if result['image_path'] == current_path:
                existing_index = i
                break
        
        # 准备结果数据
        result_data = {
            'image_path': current_path,
            'image_name': os.path.basename(current_path),
            'prompt': self.parent.current_prompt,
            'masks': self.parent.instance_masks.copy(),
            'boxes': self.parent.instance_boxes.copy(),
            'confidences': self.parent.instance_confidences.copy(),
            'instance_count': len(self.parent.instance_masks),
            # 这些列表也保存，以便兼容性，但实际加载时会重算
            'major_axes': [], 
            'minor_axes': [],
            'instance_areas_um2': self.parent.instance_areas_um2.copy() if hasattr(self.parent, 'instance_areas_um2') else [],
            'instance_centers': self.parent.instance_centers.copy() if hasattr(self.parent, 'instance_centers') else []
        }
        
        if existing_index >= 0:
            # 更新现有结果
            self.parent.batch_results[existing_index] = result_data
        else:
            # 添加新结果
            self.parent.batch_results.append(result_data)

    # ==================== 几何框分割方法 ====================
    def box_segment(self):
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        if self.parent.state is None:
            QMessageBox.warning(self.parent, "Warning", "Please upload an image first")
            return
            
        # 1. 检查是否有用户绘制的框数据
        if not hasattr(self.parent, 'user_roi_data') or not self.parent.user_roi_data:
            QMessageBox.warning(self.parent, "Warning", "Please draw a box first")
            return
            
        self.parent.set_loading(True, f"Segmenting with {len(self.parent.user_roi_data)} boxes...")
        
        # 2. 准备所有框的数据
        # 我们需要遍历所有绘制的框，而不是只取最后一个
        img_w = self.parent.state["original_width"]
        img_h = self.parent.state["original_height"]
        
        all_prompts = []
        
        for roi in self.parent.user_roi_data:
            box_coords = roi['box'] # [x0, y0, x1, y1]
            mode = roi['mode']      # 'positive' or 'negative'
            
            x0, y0, x1, y1 = box_coords
            width = x1 - x0
            height = y1 - y0
            
            # 归一化中心点和宽高 [cx, cy, w, h]
            center_x = (x0 + width/2) / img_w
            center_y = (y0 + height/2) / img_h
            norm_width = width / img_w
            norm_height = height / img_h
            
            box_norm = [center_x, center_y, norm_width, norm_height]
            
            # 关键：根据模式设置标签
            # True = 正向提示 (Include), False = 负向提示 (Exclude)
            label = (mode == "positive")
            
            all_prompts.append((box_norm, label))
            
        # 3. 发送给处理线程
        # 注意：我们将整个 all_prompts 列表传给 segment_with_box
        self.parent.processing_thread = ImageProcessor(
            self.parent.sam3_manager.processor,
            self.segment_with_box,
            all_prompts, self.parent.state
        )
        self.parent.processing_thread.finished.connect(self.on_box_segmentation_finished)
        self.parent.processing_thread.error.connect(self.parent.on_segmentation_error)
        self.parent.processing_thread.start()

    def segment_with_box(self, all_prompts, state):
        """
        处理多个几何框提示。
        all_prompts: list of tuples [(box_norm, label), ...]
        """
        try:
            # 1. 先重置之前的提示，确保从干净的状态开始
            self.parent.sam3_manager.processor.reset_all_prompts(state)
            
            # 2. 依次注入所有框
            # SAM3 的 processor.add_geometric_prompt 会将新框追加到当前状态中
            # 并根据所有累积的框重新计算 mask
            current_state = state
            
            for box, label in all_prompts:
                # add_geometric_prompt 返回更新后的 state
                current_state = self.parent.sam3_manager.processor.add_geometric_prompt(
                    box=box, 
                    label=label, 
                    state=current_state
                )
                
            return {
                'state': current_state, 
                'prompt': f"multi-box ({len(all_prompts)} prompts)"
            }
        except Exception as e:
            raise Exception(f"Box segmentation failed: {str(e)}")

    def on_box_segmentation_finished(self, result):
        self.parent.set_loading(False)
        self.parent.state = result['state']
        prompt = result['prompt']
        
        # 保存中间状态
        if "masks" in self.parent.state:
            masks = self.parent.state.get("masks", [])
            boxes = self.parent.state.get("boxes", [])
            scores = self.parent.state.get("scores", [])
            
            # 清空之前的中间状态
            self.parent.instance_masks = []
            self.parent.instance_boxes = []
            self.parent.instance_confidences = []
            
            # 保存到中间状态变量中
            for mask, box, score in zip(masks, boxes, scores):
                # 处理mask
                mask_np = mask[0].cpu().numpy() if torch.is_tensor(mask[0]) else mask[0]
                self.parent.instance_masks.append(mask_np)
                
                # 处理box
                if box is not None:
                    box_list = box.cpu().tolist() if torch.is_tensor(box) else box
                    self.parent.instance_boxes.append(box_list)
                else:
                    self.parent.instance_boxes.append([])
                
                # 处理score
                score_value = score.item() if torch.is_tensor(score) else score
                self.parent.instance_confidences.append(score_value)
        
        # 重置轴线数据
        self.parent.instance_major_axes = []
        self.parent.instance_minor_axes = []
        self.parent.clear_axis_lines()
        
        self.parent.update_display()  # 确保分割后立即更新画布
        self.parent.status_label.setText(f"✅ {prompt} segmentation complete")
        self.on_calc_phenotype()

    # ==================== 表型计算方法 (完全重构 & 可视化增强) ====================
    def on_calc_phenotype(self):
        """
        [Scientific] 计算表型并生成可视化几何数据
        核心改进：将 PCA 统计数据反算为图像坐标系上的几何图形，实现所见即所得。
        """
        if len(self.parent.instance_masks) == 0:
            self.parent.phenotype_display.setText("No valid masks")
            return
        
        # 1. 调用计算器 (获取数据)
        try:
            self.current_phenotype_df = self.phenotype_calculator.calculate_basic_properties(
                self.parent.instance_masks,
                self.parent.instance_boxes,
                self.parent.instance_confidences
            )
        except Exception as e:
            print(f"Error in phenotype calculation: {e}")
            import traceback
            traceback.print_exc()
            self.parent.phenotype_display.setText(f"Calculation Error: {str(e)}")
            return

        if self.current_phenotype_df is None or self.current_phenotype_df.empty:
            self.parent.phenotype_display.setText("Calculation yielded no results")
            return

        # 2. 回填基础数据 (兼容性)
        self.parent.instance_areas_um2 = self.current_phenotype_df['area_um2'].tolist()
        self.parent.instance_centers = list(zip(
            self.current_phenotype_df['center_x'], 
            self.current_phenotype_df['center_y']
        ))
        self.parent.instance_indices = list(range(len(self.current_phenotype_df)))
        
        # 3. [可视化核心] 计算 PCA 主轴/次轴的绘图坐标
        major_axes_coords = []
        minor_axes_coords = []
        
        scale = self.parent.scale_factor # px/um
        
        for _, row in self.current_phenotype_df.iterrows():
            cx, cy = row['center_x'], row['center_y']
            
            # 将微米转回像素用于绘图
            major_len_px = row['major_axis_um'] * scale
            minor_len_px = row['minor_axis_um'] * scale
            angle_deg = row['pca_angle'] # PCA 计算出的物理角度
            
            # 将角度转换为弧度
            theta = np.radians(angle_deg)
            
            # 计算主轴端点 (dx, dy)
            dx_major = (major_len_px / 2) * np.cos(theta)
            dy_major = (major_len_px / 2) * np.sin(theta)
            
            # 计算次轴端点 (垂直于主轴, theta + 90度)
            dx_minor = (minor_len_px / 2) * np.cos(theta + np.pi/2)
            dy_minor = (minor_len_px / 2) * np.sin(theta + np.pi/2)
            
            # 存储坐标 [(x1, y1), (x2, y2)]
            major_axes_coords.append([(cx - dx_major, cy - dy_major), (cx + dx_major, cy + dy_major)])
            minor_axes_coords.append([(cx - dx_minor, cy - dy_minor), (cx + dx_minor, cy + dy_minor)])
            
        self.parent.instance_major_axes = major_axes_coords
        self.parent.instance_minor_axes = minor_axes_coords
        
        # 4. 更新界面显示
        self.update_phenotype_display()
        
        # 5. 触发画布重绘 (显示轴线)
        # 如果主窗口有相应的控制方法，调用它
        if hasattr(self.parent, 'update_display'):
            self.parent.update_display()

    def update_phenotype_display(self):
        """更新表型显示文本 (基于 DataFrame 的丰富数据)"""
        if self.current_phenotype_df is None or self.current_phenotype_df.empty:
            self.parent.phenotype_display.setText("No data available")
            return
        
        df = self.current_phenotype_df
        
        # 统计数据
        total_count = len(df)
        avg_area = df['area_um2'].mean()
        std_area = df['area_um2'].std()
        
        # 高级形状因子统计
        avg_circularity = df['circularity'].mean()
        avg_solidity = df['solidity'].mean()
        avg_aspect_ratio = df['aspect_ratio'].mean()
        
        # 构建显示文本
        info_text = "【Scientific Phenotype Analysis】\n"
        info_text += "=" * 60 + "\n"
        info_text += f"📁 Class: {self.parent.current_prompt if self.parent.current_prompt else 'Unspecified'}\n"
        info_text += f"🔢 Count: {total_count} | Scale: {self.parent.scale_factor:.2f} px/μm\n"
        
        if self.parent.batch_mode:
            info_text += f"📂 Image: {self.parent.current_image_index+1}/{len(self.parent.image_paths)}\n"
        
        info_text += "=" * 60 + "\n\n"
        
        # 详细列表 (前 20 个实例)
        info_text += "📊 Instance Details (Top 20):\n"
        info_text += "-" * 90 + "\n"
        info_text += f"{'ID':<4} {'Area(μm²)':<12} {'Major(μm)':<10} {'Minor(μm)':<10} {'Circ.':<8} {'Solid.':<8} {'Asp.R':<8}\n"
        info_text += "-" * 90 + "\n"

        for idx, row in df.head(20).iterrows():
            info_text += (
                f"{int(row['instance_id']):<4} "
                f"{row['area_um2']:<12.2f} "
                f"{row['major_axis_um']:<10.2f} "
                f"{row['minor_axis_um']:<10.2f} "
                f"{row['circularity']:<8.2f} "
                f"{row['solidity']:<8.2f} "
                f"{row['aspect_ratio']:<8.2f}\n"
            )
            
        if total_count > 20:
            info_text += f"... and {total_count - 20} more instances.\n"

        info_text += "-" * 90 + "\n\n"
        
        # 汇总统计
        info_text += "📈 Population Statistics:\n"
        info_text += f"  • Area (μm²):     Mean={avg_area:.2f} ± {std_area:.2f}\n"
        info_text += f"  • Shape Factors:  Circularity={avg_circularity:.2f} (1.0=Circle)\n"
        info_text += f"                    Solidity={avg_solidity:.2f}, Aspect Ratio={avg_aspect_ratio:.2f}\n"
        
        info_text += "\n💡 Note: Axis lengths are calculated using PCA (Principal Component Analysis)."
        if self.parent.batch_mode:
            info_text += "\n💡 Tip: Use Prev/Next buttons to navigate between images"

        self.parent.phenotype_display.setText(info_text)
        self.parent.status_label.setText(f"✅ Phenotype calculated (Count: {total_count})")

    # ==================== 导出数据方法 (使用 Excel) ====================
    def on_export_data(self):
        """导出当前分割结果 (使用 PhenotypeCalculator 的 Excel 导出)"""
        if self.current_phenotype_df is None and (not self.parent.batch_mode or not self.parent.batch_results):
            QMessageBox.warning(self.parent, "Warning", "No data to export. Please calculate phenotype first.")
            return
        
        default_name = "phenotype_analysis.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Export Data", 
            default_name, 
            "Excel Files (*.xlsx);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # 批量导出逻辑
                if self.parent.batch_mode and self.parent.batch_results:
                    from ui.widegets import BatchExportProcessor
                    self.export_processor = BatchExportProcessor(
                        self.parent.batch_results,
                        self.parent.scale_factor,
                        file_path
                    )
                    self.export_progress = QProgressDialog("Exporting batch data...", "Cancel", 0, len(self.parent.batch_results), self.parent)
                    self.export_progress.setWindowTitle("Export Data")
                    self.export_progress.setWindowModality(Qt.WindowModal)
                    self.export_progress.show()
                    
                    self.export_processor.progress.connect(self.on_export_progress)
                    self.export_processor.finished.connect(self.on_export_finished)
                    self.export_processor.error.connect(self.on_export_error)
                    self.export_processor.start()
                
                # 单图导出逻辑 (完全替换)
                elif self.current_phenotype_df is not None:
                    img_name = os.path.basename(self.parent.image_paths[self.parent.current_image_index]) if self.parent.image_paths else "image"
                    report = {
                        "Image": img_name,
                        "Prompt": self.parent.current_prompt,
                        "Count": len(self.current_phenotype_df),
                        "Scale": self.parent.scale_factor
                    }
                    
                    saved_path = self.phenotype_calculator.export_to_csv(
                        stoma_df=self.current_phenotype_df, 
                        pavement_cell_df=pd.DataFrame(),
                        report=report,
                        output_path=file_path
                    )
                    
                    if saved_path:
                        self.parent.status_label.setText(f"✅ Data exported to {os.path.basename(saved_path)}")
                        QMessageBox.information(self.parent, "Success", f"Data exported successfully to:\n{saved_path}")
                
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to export data:\n{str(e)}")
    
    def on_export_progress(self, index, filename):
        if self.export_progress:
            self.export_progress.setValue(index)
            self.export_progress.setLabelText(f"Exporting {filename} ({index+1}/{len(self.parent.batch_results)})")
            QCoreApplication.processEvents()
    
    def on_export_finished(self, export_path):
        if self.export_progress:
            self.export_progress.close()
        
        self.parent.status_label.setText(f"✅ Batch data exported to {os.path.basename(export_path)}")
        QMessageBox.information(self.parent, "Success", 
            f"Batch data exported successfully!\nTotal images: {len(self.parent.batch_results)}\nSaved to: {export_path}")
    
    def on_export_error(self, error_msg):
        if self.export_progress:
            self.export_progress.close()
        
        QMessageBox.critical(self.parent, "Error", f"Failed to export batch data:\n{error_msg}")

    # ==================== 可视化导出方法 ====================
    def export_visualization_image(self):
        """
        [Updated] 导出科研可视化图像 - 调用主窗口的一致性保存方法
        """
        # 检查是否有数据
        has_results = (len(self.parent.instance_masks) > 0 or 
                       len(self.stomata_masks) > 0 or 
                       len(self.pavement_masks) > 0)
                       
        if self.parent.current_image_array is None or not has_results:
            QMessageBox.warning(self.parent, "Warning", "Please perform segmentation first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Export Visualization", 
            "phenotype_visualization.png", 
            "PNG Images (*.png);;JPEG Images (*.jpg)"
        )
        
        if not file_path:
            return

        try:
            self.parent.set_loading(True, "Exporting High-Res Visualization...")
            # 直接调用我们在 main_window 中新增的方法，确保一致性
            self.parent.save_visualization_to_file(file_path)
            self.parent.set_loading(False)
            
            QMessageBox.information(self.parent, "Success", f"Visualization saved to:\n{file_path}")
            
        except Exception as e:
            self.parent.set_loading(False)
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Error", f"Failed to export visualization:\n{str(e)}")

    # ==================== 置信度变化方法 ====================
    def on_confidence_change(self, value):
        confidence = value / 100.0
        self.parent.confidence_label.setText(f"📊 Confidence Threshold: {confidence:.2f}")
        # 当置信度阈值改变时，直接更新显示，因为中间状态已经保存
        self.parent.update_display()

    # ==================== 清理提示方法 ====================
    def on_clear_prompts(self):
        """
        [修正版] 清除所有提示和分割结果，保留图像和模型状态。
        如果状态不慎丢失，会尝试自动从当前图像恢复，防止报 'Please upload image first' 错误。
        """
        try:
            # 1. 清除用户绘制的矩形框 (包括数据和视觉对象)
            if hasattr(self.parent, 'user_drawn_patches'):
                for patch in self.parent.user_drawn_patches:
                    try:
                        patch.remove()
                    except Exception:
                        pass
                self.parent.user_drawn_patches = []

            if hasattr(self.parent, 'user_roi_data'):
                self.parent.user_roi_data = []

            # (兼容旧代码)
            if hasattr(self.parent.canvas, 'drawn_boxes'):
                self.parent.canvas.drawn_boxes = []

            # 2. 移除当前正在绘制但未释放的矩形
            if self.parent.canvas.current_rect is not None:
                try:
                    self.parent.canvas.current_rect.remove()
                except Exception:
                    pass
                finally:
                    self.parent.canvas.current_rect = None

            # 3. 重置 SAM3 processor 的提示状态
            if self.parent.sam3_manager.processor is not None:
                # 尝试重置提示
                if self.parent.state is not None:
                    try:
                        self.parent.sam3_manager.processor.reset_all_prompts(self.parent.state)
                        # 显式清空 state 中的推理结果，防止模型缓存干扰
                        if isinstance(self.parent.state, dict):
                            if 'masks' in self.parent.state: self.parent.state['masks'] = []
                            if 'boxes' in self.parent.state: self.parent.state['boxes'] = []
                            if 'scores' in self.parent.state: self.parent.state['scores'] = []
                    except Exception as e:
                        print(f"Error resetting prompts in processor: {e}")

                # [关键修正] 状态完整性检查与自动恢复
                # 如果 state 为空，但当前有图像，说明状态丢失，必须重新生成
                if self.parent.state is None and self.parent.current_image is not None:
                    print("⚠️ State was None but image exists. Restoring state automatically...")
                    try:
                        # 临时显示加载状态，因为 set_image 可能需要几秒钟
                        self.parent.set_loading(True, "Restoring image state...")
                        self.parent.state = self.parent.sam3_manager.processor.set_image(self.parent.current_image)
                        self.parent.set_loading(False)
                    except Exception as e:
                        self.parent.set_loading(False)
                        print(f"Error restoring image state: {e}")

            # 4. 清空中间分割结果变量
            self.reset_segmentation_results()

            # 5. 安全清除轴线显示
            try:
                self.parent.clear_axis_lines()
            except Exception:
                pass

            # 6. 重置表型显示面板
            if hasattr(self.parent, 'phenotype_display'):
                self.parent.phenotype_display.clear()
                self.parent.phenotype_display.setPlaceholderText("After segmentation, click Calculate to view phenotype info...")

            # 7. 更新画布显示 (这会重绘原图，从而清除所有覆盖的mask和box，但保留底图)
            if self.parent.current_image_array is not None:
                self.parent.current_xlim = None
                self.parent.current_ylim = None
                self.parent.update_display()
                
            self.parent.status_label.setText("🗑️ Prompts cleared. Image ready.")

            # 8. 恢复控件可用性
            if hasattr(self.parent, 'segment_btn'):
                self.parent.segment_btn.setEnabled(True)
            if hasattr(self.parent, 'box_segment_btn'):
                self.parent.box_segment_btn.setEnabled(True)
            if hasattr(self.parent, 'text_input'):
                self.parent.text_input.setEnabled(True)

        except Exception as e:
            print(f"Error in on_clear_prompts: {e}")
            import traceback
            traceback.print_exc()
            self.parent.status_label.setText("⚠️ Error clearing prompts.")

    def on_clear_all(self):
        """清除所有内容，包括图像、分割结果、FOV数据等"""
        reply = QMessageBox.question(
            self.parent, "Clear All",
            "Are you sure you want to clear everything (image, segmentation results, FOV data, etc.)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
        
        try:
            # 1. 重置SAM3处理器状态
            if self.parent.sam3_manager.processor is not None and self.parent.state is not None:
                try:
                    if hasattr(self.parent.sam3_manager.processor, 'reset_state'):
                        self.parent.sam3_manager.processor.reset_state(self.parent.state)
                except:
                    pass
            
            # 2. 清除图像数据
            self.parent.current_image = None
            self.parent.current_image_array = None
            self.parent.state = None
            
            # 3. 清除画布
            self.parent.canvas.ax.cla()
            self.parent.canvas.draw_idle()
            
            # 4. 重置分割结果
            self.reset_segmentation_results()
            self.reset_multi_prompt_results()
            
            # 5. 清除批量模式数据
            self.parent.batch_mode = False
            self.parent.image_folder = None
            self.parent.image_paths = []
            self.parent.current_image_index = 0
            self.parent.batch_results = []
            self.parent.folder_segmentation_active = False
            self.batch_multi_results = None
            self.batch_comprehensive_reports = []
            
            # 6. 清除FOV相关数据
            self.fov_mode = False
            self.fov_manager = None
            self.fov_stomata_results = None
            self.fov_pavement_results = None
            self.fov_aperture_results = None
            self.fov_comprehensive_report = None
            
            # 如果存在FOV管理器，也清除其数据
            if hasattr(self.parent, 'fov_manager'):
                self.parent.fov_manager.clear_tile_preview()
                # 重置FOV管理器状态
                self.parent.fov_manager.tiles = []
                self.parent.fov_manager.current_tile_index = 0
                self.parent.fov_manager.tile_results = {}
                self.parent.fov_manager.tile_multi_results = {}
                self.parent.fov_manager.is_processing = False
                self.parent.fov_manager.merged_multi_results = {}
                self.parent.fov_manager.comprehensive_report = None
            
            # 7. 清除输入和提示
            self.parent.text_input.clear()
            self.parent.current_prompt = ""
            
            # 8. 清除表型显示
            self.parent.phenotype_display.clear()
            self.parent.phenotype_display.setPlaceholderText("After segmentation, click Calculate to view phenotype info...")
            
            # 9. 清除画布上的绘制元素
            self.parent.canvas.drawn_boxes = []
            if self.parent.canvas.current_rect is not None:
                try:
                    self.parent.canvas.current_rect.remove()
                except Exception:
                    pass
                finally:
                    self.parent.canvas.current_rect = None
            
            self.parent.canvas.instance_annotations = []
            for patch in self.parent.canvas.instance_boxes_patches:
                try:
                    patch.remove()
                except:
                    pass
            self.parent.canvas.instance_boxes_patches = []
            
            # 10. 禁用导航控件
            if hasattr(self.parent, 'prev_image_btn'): 
                self.parent.prev_image_btn.setEnabled(False)
            if hasattr(self.parent, 'next_image_btn'): 
                self.parent.next_image_btn.setEnabled(False)
            
            if hasattr(self.parent, 'jump_spinbox'):
                self.parent.jump_spinbox.setEnabled(False)
                self.parent.jump_spinbox.setMaximum(1)
                self.parent.jump_spinbox.setValue(1)
            
            if hasattr(self.parent, 'jump_btn'): 
                self.parent.jump_btn.setEnabled(False)
            
            if hasattr(self.parent, 'image_nav_label'): 
                self.parent.image_nav_label.setText("📸 Image: 0/0")
            
            # 11. 禁用FOV导航按钮
            if hasattr(self.parent, 'prev_tile_btn'): 
                self.parent.prev_tile_btn.setEnabled(False)
            if hasattr(self.parent, 'next_tile_btn'): 
                self.parent.next_tile_btn.setEnabled(False)
            if hasattr(self.parent, 'tile_nav_label'): 
                self.parent.tile_nav_label.setText("Tile: 0/0")
            if hasattr(self.parent, 'merge_results_btn'): 
                self.parent.merge_results_btn.setEnabled(False)
            if hasattr(self.parent, 'merge_multi_results_btn'): 
                self.parent.merge_multi_results_btn.setEnabled(False)
            
            # 12. 禁用批量处理按钮
            if hasattr(self.parent, 'batch_segment_btn'): 
                self.parent.batch_segment_btn.setEnabled(False)
            if hasattr(self.parent, 'batch_multi_prompt_btn'): 
                self.parent.batch_multi_prompt_btn.setEnabled(False)
            
            # 13. 禁用单图多提示结果按钮
            if hasattr(self.parent, 'show_stomata_btn'): 
                self.parent.show_stomata_btn.setEnabled(False)
            if hasattr(self.parent, 'show_pavement_btn'): 
                self.parent.show_pavement_btn.setEnabled(False)
            if hasattr(self.parent, 'show_aperture_btn'): 
                self.parent.show_aperture_btn.setEnabled(False)
            if hasattr(self.parent, 'export_comprehensive_btn'): 
                self.parent.export_comprehensive_btn.setEnabled(False)
            if hasattr(self.parent, 'export_batch_comprehensive_btn'): 
                self.parent.export_batch_comprehensive_btn.setEnabled(False)
            
            # 14. 禁用FOV多提示处理按钮
            if hasattr(self.parent, 'process_fov_multi_btn'): 
                self.parent.process_fov_multi_btn.setEnabled(False)
            if hasattr(self.parent, 'merge_fov_multi_btn'): 
                self.parent.merge_fov_multi_btn.setEnabled(False)
            if hasattr(self.parent, 'show_fov_stomata_btn'): 
                self.parent.show_fov_stomata_btn.setEnabled(False)
            if hasattr(self.parent, 'show_fov_pavement_btn'): 
                self.parent.show_fov_pavement_btn.setEnabled(False)
            if hasattr(self.parent, 'show_fov_aperture_btn'): 
                self.parent.show_fov_aperture_btn.setEnabled(False)
            if hasattr(self.parent, 'export_fov_comprehensive_btn'): 
                self.parent.export_fov_comprehensive_btn.setEnabled(False)
            
            # 15. 启用上传按钮
            if hasattr(self.parent, 'upload_btn'): 
                self.parent.upload_btn.setEnabled(True)
            
            # 16. 更新导航信息
            if hasattr(self.parent, 'update_navigation_info'): 
                self.parent.update_navigation_info()
            
            # 17. 重置滑块和复选框
            self.parent.size_slider.setValue(960)
            self.parent.confidence_slider.setValue(50)
            if hasattr(self.parent, 'show_major_axis_checkbox'): 
                self.parent.show_major_axis_checkbox.setChecked(False)
            if hasattr(self.parent, 'show_minor_axis_checkbox'): 
                self.parent.show_minor_axis_checkbox.setChecked(False)
            
            # 18. 重置画布绘制模式
            if hasattr(self.parent.canvas, 'box_drawing_mode'): 
                self.parent.canvas.box_drawing_mode = False
            
            # 19. 更新状态和显示
            self.parent.status_label.setText("🗑️ All cleared - ready for new image")
            self.parent.canvas.draw_idle()
            self.parent.update_display()
            
        except Exception as e:
            print(f"Error in on_clear_all: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保上传按钮可用
            if hasattr(self.parent, 'upload_btn'): 
                self.parent.upload_btn.setEnabled(True)
            
            self.parent.status_label.setText("⚠️ Clear completed with some warnings")

    # ==================== 跳转到指定图像 ====================
    def on_jump_to_image(self, index):
        """跳转到指定序号的图像（1-based 索引）"""
        if not self.parent.batch_mode or not self.parent.image_paths:
            return
        
        target_index = index - 1
        if 0 <= target_index < len(self.parent.image_paths):
            if self.parent.current_image_index < len(self.parent.image_paths):
                self.save_current_to_batch_results()
            self.load_image_by_index(target_index)
        else:
            valid_index = max(1, min(index, len(self.parent.image_paths)))
            self.parent.jump_spinbox.setValue(valid_index)
            QMessageBox.warning(self.parent, "Invalid Index", 
                            f"Please enter a number between 1 and {len(self.parent.image_paths)}")

    # ==================== 多提示分割方法 ====================
    def on_multi_prompt_segmentation(self):
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        if self.parent.state is None:
            QMessageBox.warning(self.parent, "Warning", "Please upload an image first")
            return
        
        self.multi_prompt_progress = QProgressDialog("Multi-prompt segmentation in progress...", "Cancel", 0, 3, self.parent)
        self.multi_prompt_progress.setWindowTitle("Multi-Prompt Segmentation")
        self.multi_prompt_progress.setWindowModality(Qt.WindowModal)
        self.multi_prompt_progress.show()
        
        prompts = ["stoma", "pavement-cell", "area"]
        self.multi_prompt_processor = MultiPromptProcessor(
            self.parent.sam3_manager.processor,
            self.parent.state,
            prompts,
            self.parent.scale_factor
        )
        
        self.multi_prompt_processor.progress.connect(self.on_multi_prompt_progress)
        self.multi_prompt_processor.finished.connect(self.on_multi_prompt_finished)
        self.multi_prompt_processor.error.connect(self.on_multi_prompt_error)
        self.multi_prompt_processor.start()
    
    def on_multi_prompt_progress(self, index, prompt, description):
        if self.multi_prompt_progress:
            self.multi_prompt_progress.setValue(index)
            prompt_descriptions = {
                "stoma": "Stomata",
                "pavement-cell": "Pavement Cells",
                "area": "Stomatal Apertures"
            }
            description = prompt_descriptions.get(prompt, prompt)
            self.multi_prompt_progress.setLabelText(f"Segmenting {description}... ({index+1}/3)")
            QCoreApplication.processEvents()
    
    def on_multi_prompt_finished(self, results):
        if self.multi_prompt_progress:
            self.multi_prompt_progress.close()
        
        self.stomata_results = results.get("stoma", {})
        self.pavement_cell_results = results.get("pavement-cell", {})
        self.aperture_results = results.get("area", {})
        
        self.calculate_comprehensive_phenotype()
        
        if hasattr(self.parent, 'show_stomata_btn'): self.parent.show_stomata_btn.setEnabled(True)
        if hasattr(self.parent, 'show_pavement_btn'): self.parent.show_pavement_btn.setEnabled(True)
        if hasattr(self.parent, 'show_aperture_btn'): self.parent.show_aperture_btn.setEnabled(True)
        if hasattr(self.parent, 'export_comprehensive_btn'): self.parent.export_comprehensive_btn.setEnabled(True)
        
        self.display_stomata_results()
        self.parent.status_label.setText("✅ Multi-prompt segmentation complete. All phenotypes calculated.")
        self.show_result_selection_dialog()
    
    def on_multi_prompt_error(self, error_msg):
        if self.multi_prompt_progress:
            self.multi_prompt_progress.close()
        QMessageBox.critical(self.parent, "Error", f"Multi-prompt segmentation failed:\n{error_msg}")
    
    def calculate_comprehensive_phenotype(self):
        """
        计算综合表型，并保存各模式的 DataFrame 供后续切换显示使用
        """
        try:
            # 获取原始数据
            stoma_masks = self.stomata_results.get("masks", [])
            pavement_masks = self.pavement_cell_results.get("masks", [])
            aperture_masks = self.aperture_results.get("masks", [])
            
            stoma_boxes = self.stomata_results.get("boxes", [])
            pavement_boxes = self.pavement_cell_results.get("boxes", [])
            aperture_boxes = self.aperture_results.get("boxes", [])
            
            stoma_confidences = self.stomata_results.get("confidences", [])
            pavement_confidences = self.pavement_cell_results.get("confidences", [])
            aperture_confidences = self.aperture_results.get("confidences", [])
            
            # 1. 计算并【保存】各模式的 DataFrame 到实例变量
            self.stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_masks, stoma_boxes, stoma_confidences)
            self.pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_masks, pavement_boxes, pavement_confidences)
            self.aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_masks, aperture_boxes, aperture_confidences) if aperture_masks else None
            
            # 更新父窗口的 phenotype_data (供 main_window 读取)
            self.parent.phenotype_data['stomata'] = self.stoma_df.to_dict('records') if not self.stoma_df.empty else []
            self.parent.phenotype_data['pavement'] = self.pavement_df.to_dict('records') if not self.pavement_df.empty else []
            self.parent.phenotype_data['aperture'] = self.aperture_df.to_dict('records') if (self.aperture_df is not None and not self.aperture_df.empty) else []

            # 图像面积计算
            if self.parent.current_image_array is not None:
                height, width = self.parent.current_image_array.shape[:2]
                image_area_px = height * width
                image_area_um2 = image_area_px * (1.0 / self.parent.scale_factor) ** 2
            else:
                image_area_um2 = 0
            
            image_name = os.path.basename(self.parent.image_paths[self.parent.current_image_index]) \
                if self.parent.batch_mode and self.parent.current_image_index < len(self.parent.image_paths) else "single_image"
            
            image_path = self.parent.image_paths[self.parent.current_image_index] \
                if self.parent.batch_mode and self.parent.current_image_index < len(self.parent.image_paths) else "N/A"
            
            # 生成报告
            self.comprehensive_report = self.phenotype_calculator.generate_comprehensive_report(
                self.stoma_df, self.pavement_df, self.aperture_df, image_area_um2, image_name, image_path
            )
            self.update_comprehensive_phenotype_display()
            
        except Exception as e:
            print(f"Error calculating comprehensive phenotype: {e}")
            import traceback
            traceback.print_exc()

    def _update_state_from_df(self, df):
        """
        [辅助函数] 根据选定的 DataFrame 更新当前的表型状态
        这确保了 Metric Overlay 和 轴线绘制 (Ellipses) 能正确显示
        """
        if df is None or df.empty:
            self.current_phenotype_df = None
            self.parent.instance_major_axes = []
            self.parent.instance_minor_axes = []
            return

        # 1. 更新当前活跃的 DataFrame
        self.current_phenotype_df = df
        
        # 2. 更新父窗口的基础数据
        self.parent.instance_areas_um2 = df['area_um2'].tolist()
        self.parent.instance_centers = list(zip(df['center_x'], df['center_y']))
        self.parent.instance_indices = list(range(len(df)))

        # 3. 重新生成轴线绘图坐标 (用于在画布上画椭圆/线)
        # 这段逻辑从 on_calc_phenotype 复用
        major_axes_coords = []
        minor_axes_coords = []
        scale = self.parent.scale_factor

        for _, row in df.iterrows():
            cx, cy = row['center_x'], row['center_y']
            
            # 将微米转回像素用于绘图
            major_len_px = row['major_axis_um'] * scale
            minor_len_px = row['minor_axis_um'] * scale
            angle_deg = row['pca_angle']
            theta = np.radians(angle_deg)
            
            # 计算主轴端点
            dx_major = (major_len_px / 2) * np.cos(theta)
            dy_major = (major_len_px / 2) * np.sin(theta)
            major_axes_coords.append([(cx - dx_major, cy - dy_major), (cx + dx_major, cy + dy_major)])
            
            # 计算次轴端点
            dx_minor = (minor_len_px / 2) * np.cos(theta + np.pi/2)
            dy_minor = (minor_len_px / 2) * np.sin(theta + np.pi/2)
            minor_axes_coords.append([(cx - dx_minor, cy - dy_minor), (cx + dx_minor, cy + dy_minor)])

        self.parent.instance_major_axes = major_axes_coords
        self.parent.instance_minor_axes = minor_axes_coords
    
    def update_comprehensive_phenotype_display(self):
        if not self.comprehensive_report:
            self.parent.phenotype_display.setText("No comprehensive phenotype data available")
            return
        
        report = self.comprehensive_report
        info_text = "【COMPREHENSIVE PHENOTYPE ANALYSIS】\n"
        info_text += "=" * 60 + "\n\n"
        
        # --- 1. 图像基本信息 ---
        image_info = report['image_info']
        info_text += f"📁 Image: {image_info['image_name']}\n"
        info_text += f"📏 Scale: {image_info['scale_factor_px_per_um']:.2f} px/μm\n"
        if image_info['image_area_um2'] > 0:
            info_text += f"📐 Image Area: {image_info['image_area_um2']:.0f} μm² ({image_info['image_area_um2']/1e6:.3f} mm²)\n"
        
        # --- 2. 数量统计 ---
        counts = report['basic_counts']
        info_text += f"\n🔢 Instance Counts:\n"
        info_text += f"  • Stomata: {counts['stomatal_count']}\n"
        info_text += f"  • Pavement Cells: {counts['pavement_cell_count']}\n"
        if 'aperture_count' in counts:
            info_text += f"  • Stomatal Apertures: {counts['aperture_count']}\n"
        
        # --- 3. 密度与指数 ---
        indices = report['composite_indices']
        if 'stomatal_density_per_mm2' in indices:
            info_text += f"\n📊 Stomatal Density: {indices['stomatal_density_per_mm2']:.2f} /mm²\n"
        if 'stomatal_index_percent' in indices:
            info_text += f"📈 Stomatal Index: {indices['stomatal_index_percent']:.2f}%\n"
        
        # --- 4. 基础尺寸指标 ---
        if 'stomatal_area_mean_um2' in indices:
            info_text += f"\n🌱 Stomatal Size (Basic):\n"
            info_text += f"  • Mean Area: {indices['stomatal_area_mean_um2']:.2f} μm²\n"
            
            # 使用安全检查避免 KeyError
            if 'stomatal_length_mean_um' in indices:
                info_text += f"  • Mean Length: {indices['stomatal_length_mean_um']:.2f} μm\n"
            
            if 'stomatal_width_mean_um' in indices:
                info_text += f"  • Mean Width: {indices['stomatal_width_mean_um']:.2f} μm\n"
            
            # [修复] 增加对 stomatal_aspect_ratio_mean 的存在性检查
            if 'stomatal_aspect_ratio_mean' in indices:
                info_text += f"  • Mean Aspect Ratio: {indices['stomatal_aspect_ratio_mean']:.2f}\n"
        
        # --- 5. 高级形态学指标 ---
        advanced_keys = ['stomatal_solidity_mean', 'stomatal_convexity_mean', 'stomatal_lobeyness_mean', 'stomatal_undulation_index_mean']
        has_advanced = any(k in indices for k in advanced_keys) or ('stomatal_circularity_mean' in indices)
        
        if has_advanced:
            info_text += f"\n🧩 Advanced Morphology:\n"
            if 'stomatal_circularity_mean' in indices:
                info_text += f"  • Circularity: {indices['stomatal_circularity_mean']:.3f} (1.0=Circle)\n"
            if 'stomatal_solidity_mean' in indices:
                info_text += f"  • Solidity: {indices['stomatal_solidity_mean']:.3f}\n"
            if 'stomatal_convexity_mean' in indices:
                info_text += f"  • Convexity: {indices['stomatal_convexity_mean']:.3f}\n"
            if 'stomatal_undulation_index_mean' in indices:
                info_text += f"  • Undulation Index: {indices['stomatal_undulation_index_mean']:.3f}\n"
            if 'stomatal_lobeyness_mean' in indices:
                info_text += f"  • Lobeyness: {indices['stomatal_lobeyness_mean']:.3f}\n"
            if 'stomatal_ect_complexity_mean' in indices:
                 info_text += f"  • ECT Complexity: {indices['stomatal_ect_complexity_mean']:.3f}\n"

        # --- 6. 生理功能潜力 ---
        if 'theoretical_gsmax_mol_m2_s' in indices:
            info_text += f"\n💧 Physiological Potential:\n"
            val = indices['theoretical_gsmax_mol_m2_s']
            info_text += f"  • Theoretical g_smax: {val:.4e} mol m⁻² s⁻¹\n"
        
        # --- 7. 气孔开度 ---
        if 'aperture_area_mean_um2' in indices:
            info_text += f"\n🕳️ Stomatal Aperture:\n"
            info_text += f"  • Mean Aperture Area: {indices['aperture_area_mean_um2']:.2f} μm²\n"
            info_text += f"  • Aperture Ratio: {indices.get('aperture_ratio_mean', 0):.3f}\n"
        
        # --- 8. 分布均匀性 ---
        if 'stomatal_distribution_uniformity' in indices:
            info_text += f"\n🎯 Distribution Uniformity: {indices['stomatal_distribution_uniformity']:.3f}\n"
        
        info_text += "\n💡 Tip: Use buttons below to view different segmentation results\n"
        info_text += "💡 Tip: Click 'Export Comprehensive Data' to save all results"
        
        self.parent.phenotype_display.setText(info_text)
    
    def display_stomata_results(self):
        if self.stomata_results:
            # 1. 更新掩码和框
            self.parent.instance_masks = self.stomata_results.get("masks", [])
            self.parent.instance_boxes = self.stomata_results.get("boxes", [])
            self.parent.instance_confidences = self.stomata_results.get("confidences", [])
            
            if hasattr(self, 'stoma_df'):
                self._update_state_from_df(self.stoma_df)
            
            self.parent.current_prompt = "stoma"
            
            # 3. 更新侧边栏文本显示
            self.update_phenotype_display()
            
            # 4. 刷新画布
            self.parent.update_display()
    
    def display_pavement_cell_results(self):
        if self.pavement_cell_results:
            self.parent.instance_masks = self.pavement_cell_results.get("masks", [])
            self.parent.instance_boxes = self.pavement_cell_results.get("boxes", [])
            self.parent.instance_confidences = self.pavement_cell_results.get("confidences", [])
            
            if hasattr(self, 'pavement_df'):
                self._update_state_from_df(self.pavement_df)
            
            self.parent.current_prompt = "pavement-cell"
            self.update_phenotype_display()
            self.parent.update_display()
    
    def display_aperture_results(self):
        if self.aperture_results:
            self.parent.instance_masks = self.aperture_results.get("masks", [])
            self.parent.instance_boxes = self.aperture_results.get("boxes", [])
            self.parent.instance_confidences = self.aperture_results.get("confidences", [])
            
            if hasattr(self, 'aperture_df'):
                self._update_state_from_df(self.aperture_df)
                
            self.parent.current_prompt = "area"
            self.update_phenotype_display()
            self.parent.update_display()
    
    def show_result_selection_dialog(self):
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Select Results to Display")
        dialog.setFixedSize(400, 200)
        
        layout = QVBoxLayout(dialog)
        label = QLabel("Multi-prompt segmentation complete!\nSelect which results to display:")
        label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 10px;")
        layout.addWidget(label)
        
        button_layout = QHBoxLayout()
        stomata_btn = QPushButton("👁️ Show Stomata")
        stomata_btn.clicked.connect(lambda: self.on_select_results("stomata", dialog))
        stomata_btn.setStyleSheet("QPushButton {background-color: #3b82f6; color: white; padding: 10px; border-radius: 5px; font-weight: bold;} QPushButton:hover {background-color: #2563eb;}")
        button_layout.addWidget(stomata_btn)
        
        pavement_btn = QPushButton("🧩 Show Pavement Cells")
        pavement_btn.clicked.connect(lambda: self.on_select_results("pavement", dialog))
        pavement_btn.setStyleSheet("QPushButton {background-color: #10b981; color: white; padding: 10px; border-radius: 5px; font-weight: bold;} QPushButton:hover {background-color: #059669;}")
        button_layout.addWidget(pavement_btn)
        
        aperture_btn = QPushButton("🕳️ Show Apertures")
        aperture_btn.clicked.connect(lambda: self.on_select_results("aperture", dialog))
        aperture_btn.setStyleSheet("QPushButton {background-color: #8b5cf6; color: white; padding: 10px; border-radius: 5px; font-weight: bold;} QPushButton:hover {background-color: #7c3aed;}")
        button_layout.addWidget(aperture_btn)
        
        layout.addLayout(button_layout)
        tip_label = QLabel("Tip: You can switch between results using the new buttons in the left panel")
        tip_label.setStyleSheet("font-size: 12px; color: #64748b; padding: 10px;")
        layout.addWidget(tip_label)
        dialog.exec_()
    
    def on_select_results(self, result_type, dialog):
        dialog.close()
        if result_type == "stomata":
            self.display_stomata_results()
            self.parent.status_label.setText("👁️ Displaying stomata segmentation results")
        elif result_type == "pavement":
            self.display_pavement_cell_results()
            self.parent.status_label.setText("🧩 Displaying pavement cell segmentation results")
        elif result_type == "aperture":
            self.display_aperture_results()
            self.parent.status_label.setText("🕳️ Displaying stomatal aperture results")
    
    # ==================== 导出综合数据 ====================
    def on_export_comprehensive_data(self):
        if (not self.stomata_results and not self.pavement_cell_results and not self.aperture_results):
            QMessageBox.warning(self.parent, "Warning", "No comprehensive data to export")
            return
        
        default_name = "comprehensive_phenotype_data.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Export Comprehensive Data", 
            default_name, 
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                stoma_masks = self.stomata_results.get("masks", [])
                pavement_masks = self.pavement_cell_results.get("masks", [])
                aperture_masks = self.aperture_results.get("masks", [])
                
                stoma_boxes = self.stomata_results.get("boxes", [])
                pavement_boxes = self.pavement_cell_results.get("boxes", [])
                aperture_boxes = self.aperture_results.get("boxes", [])
                
                stoma_confidences = self.stomata_results.get("confidences", [])
                pavement_confidences = self.pavement_cell_results.get("confidences", [])
                aperture_confidences = self.aperture_results.get("confidences", [])
                
                stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_masks, stoma_boxes, stoma_confidences)
                pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_masks, pavement_boxes, pavement_confidences)
                aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_masks, aperture_boxes, aperture_confidences) if aperture_masks else None
                
                export_path = self.phenotype_calculator.export_to_csv(stoma_df, pavement_df, aperture_df, self.comprehensive_report, file_path)
                
                self.parent.status_label.setText(f"✅ Comprehensive data exported to {os.path.basename(export_path)}")
                QMessageBox.information(
                    self.parent, "Success", 
                    f"Comprehensive phenotype data exported successfully!\nSaved to: {export_path}"
                )
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to export comprehensive data:\n{str(e)}")
    
    # ==================== 批量多提示分割 ====================
    def on_batch_multi_prompt_segmentation(self):
        if not self.parent.batch_mode or not self.parent.image_paths:
            QMessageBox.warning(self.parent, "Warning", "Please upload a folder first")
            return
        
        reply = QMessageBox.question(
            self.parent, "Batch Multi-Prompt Segmentation",
            "Segment all images with all three prompts?\n(Stomata, Pavement Cells, Stomatal Apertures)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        self.batch_multi_prompt_processor = BatchMultiPromptProcessor(
            self.parent.sam3_manager.processor,
            self.parent.image_paths,
            ["stoma", "pavement-cell", "area"],
            self.parent.scale_factor
        )
        
        total_steps = len(self.parent.image_paths) * 3
        self.batch_multi_progress = QProgressDialog("Batch multi-prompt segmentation...", "Cancel", 0, total_steps, self.parent)
        self.batch_multi_progress.setWindowTitle("Batch Multi-Prompt Segmentation")
        self.batch_multi_progress.setWindowModality(Qt.WindowModal)
        self.batch_multi_progress.show()
        
        self.batch_multi_prompt_processor.progress.connect(self.on_batch_multi_progress)
        self.batch_multi_prompt_processor.finished.connect(self.on_batch_multi_finished)
        self.batch_multi_prompt_processor.error.connect(self.on_batch_multi_error)
        self.batch_multi_prompt_processor.start()
    
    def on_batch_multi_progress(self, image_index, prompt_index, total_images, total_prompts):
        if self.batch_multi_progress:
            current_step = image_index * total_prompts + prompt_index
            self.batch_multi_progress.setValue(current_step)
            prompts = ["Stomata", "Pavement Cells", "Stomatal Apertures"]
            prompt_name = prompts[prompt_index] if prompt_index < len(prompts) else f"Prompt {prompt_index+1}"
            self.batch_multi_progress.setLabelText(f"Processing image {image_index+1}/{total_images}\nSegmenting {prompt_name}...")
            QCoreApplication.processEvents()
    
    def on_batch_multi_finished(self, all_results):
        if self.batch_multi_progress:
            self.batch_multi_progress.close()
        
        self.batch_multi_results = all_results
        self.generate_batch_comprehensive_report(all_results)
        
        if hasattr(self.parent, 'export_batch_comprehensive_btn'):
            self.parent.export_batch_comprehensive_btn.setEnabled(True)
        
        if self.parent.batch_mode and self.parent.current_image_index < len(self.parent.image_paths):
            current_image_path = self.parent.image_paths[self.parent.current_image_index]
            self.load_multi_prompt_results_for_image(current_image_path)
        
        self.parent.status_label.setText(f"✅ Batch multi-prompt segmentation complete: {len(all_results)} images processed")
        QMessageBox.information(self.parent, "Success", "Batch multi-prompt segmentation completed!")
    
    def on_batch_multi_error(self, error_msg):
        if self.batch_multi_progress:
            self.batch_multi_progress.close()
        QMessageBox.critical(self.parent, "Error", f"Batch multi-prompt segmentation failed:\n{error_msg}")
    
    def generate_batch_comprehensive_report(self, all_results):
        try:
            self.batch_comprehensive_reports = []
            for image_idx, image_results in enumerate(all_results):
                if not image_results: continue
                
                stoma_results = image_results.get("stoma", {})
                pavement_results = image_results.get("pavement-cell", {})
                aperture_results = image_results.get("area", {})
                
                stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_results.get("masks", []), stoma_results.get("boxes", []), stoma_results.get("confidences", []))
                pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_results.get("masks", []), pavement_results.get("boxes", []), pavement_results.get("confidences", []))
                aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_results.get("masks", []), aperture_results.get("boxes", []), aperture_results.get("confidences", [])) if aperture_results else None
                
                if self.parent.current_image_array is not None:
                    height, width = self.parent.current_image_array.shape[:2]
                    image_area_um2 = height * width * (1.0 / self.parent.scale_factor) ** 2
                else:
                    image_area_um2 = 0
                
                image_name = os.path.basename(self.parent.image_paths[image_idx]) if image_idx < len(self.parent.image_paths) else f"image_{image_idx+1}"
                image_path = self.parent.image_paths[image_idx] if image_idx < len(self.parent.image_paths) else "N/A"
                
                report = self.phenotype_calculator.generate_comprehensive_report(stoma_df, pavement_df, aperture_df, image_area_um2, image_name, image_path)
                self.batch_comprehensive_reports.append(report)
        except Exception as e:
            print(f"Error generating batch comprehensive report: {e}")
    
    def on_export_batch_comprehensive_data(self):
        if not hasattr(self, 'batch_multi_results') or not self.batch_multi_results:
            QMessageBox.warning(self.parent, "Warning", "No batch comprehensive data to export")
            return
        
        default_name = "batch_comprehensive_phenotype_data.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(self.parent, "Export Batch Comprehensive Data", default_name, "Excel Files (*.xlsx);;All Files (*.*)")
        
        if file_path:
            try:
                self.batch_export_processor = BatchComprehensiveExportProcessor(self.batch_multi_results, self.parent.image_paths, self.parent.scale_factor, file_path)
                self.batch_export_progress = QProgressDialog("Exporting batch comprehensive data...", "Cancel", 0, len(self.batch_multi_results), self.parent)
                self.batch_export_progress.setWindowTitle("Export Batch Data")
                self.batch_export_progress.setWindowModality(Qt.WindowModal)
                self.batch_export_progress.show()
                
                self.batch_export_processor.progress.connect(self.on_batch_export_progress)
                self.batch_export_processor.finished.connect(self.on_batch_export_finished)
                self.batch_export_processor.error.connect(self.on_batch_export_error)
                self.batch_export_processor.start()
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to export batch comprehensive data:\n{str(e)}")
    
    def on_batch_export_progress(self, index, filename):
        if self.batch_export_progress:
            self.batch_export_progress.setValue(index)
            self.batch_export_progress.setLabelText(f"Exporting {filename}... ({index+1}/{len(self.batch_multi_results)})")
            QCoreApplication.processEvents()
    
    def on_batch_export_finished(self, export_path):
        if self.batch_export_progress:
            self.batch_export_progress.close()
        self.parent.status_label.setText(f"✅ Batch comprehensive data exported to {os.path.basename(export_path)}")
        QMessageBox.information(self.parent, "Success", f"Batch comprehensive data exported successfully!\nSaved to: {export_path}")
    
    def on_batch_export_error(self, error_msg):
        if self.batch_export_progress:
            self.batch_export_progress.close()
        QMessageBox.critical(self.parent, "Error", f"Failed to export batch comprehensive data:\n{error_msg}")
    
    def reset_segmentation_results(self):
        """重置分割结果"""
        self.parent.segmentation_masks = []
        self.parent.segmentation_scores = []
        self.parent.instance_areas_um2 = []
        self.parent.instance_centers = []
        self.parent.instance_indices = []
        self.parent.instance_boxes = []
        self.parent.instance_confidences = []
        self.parent.instance_masks = []
        
        # 清空 DataFrame
        self.current_phenotype_df = None
        
        # 重置轴线数据
        self.parent.instance_major_axes = []
        self.parent.instance_minor_axes = []
        self.parent.clear_axis_lines()
    
    def reset_multi_prompt_results(self):
        self.stomata_results = None
        self.pavement_cell_results = None
        self.aperture_results = None
        self.comprehensive_report = None
        
        # === 禁用标准分析Tab按钮 ===
        if hasattr(self.parent, 'show_stomata_btn'): self.parent.show_stomata_btn.setEnabled(False)
        if hasattr(self.parent, 'show_pavement_btn'): self.parent.show_pavement_btn.setEnabled(False)
        if hasattr(self.parent, 'show_aperture_btn'): self.parent.show_aperture_btn.setEnabled(False)
        if hasattr(self.parent, 'export_comprehensive_btn'): self.parent.export_comprehensive_btn.setEnabled(False)
        if hasattr(self.parent, 'export_batch_comprehensive_btn'): self.parent.export_batch_comprehensive_btn.setEnabled(False)

        # === 禁用批量处理Tab按钮 ===
        if hasattr(self.parent, 'batch_show_stomata_btn'): self.parent.batch_show_stomata_btn.setEnabled(False)
        if hasattr(self.parent, 'batch_show_pavement_btn'): self.parent.batch_show_pavement_btn.setEnabled(False)
        if hasattr(self.parent, 'batch_show_aperture_btn'): self.parent.batch_show_aperture_btn.setEnabled(False)

    def on_process_fov_multi_prompt(self):
        """处理大视野图像的多提示分割"""
        if not hasattr(self.parent, 'fov_manager') or not self.parent.fov_manager.tiles:
            QMessageBox.warning(self.parent, "Warning", "Please generate tiles first in FOV mode")
            return
        
        # 检查是否已加载模型
        if self.parent.sam3_manager.processor is None:
            QMessageBox.warning(self.parent, "Warning", "Please load SAM3 model first")
            return
        
        # 确认处理
        reply = QMessageBox.question(
            self.parent, "FOV Multi-Prompt Processing",
            f"Process {len(self.parent.fov_manager.tiles)} tiles with multi-prompt?\n"
            f"Prompts: {', '.join(self.parent.fov_manager.multi_prompts)}\n"
            f"This may take a while.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # 设置FOV模式
        self.fov_mode = True
        self.fov_manager = self.parent.fov_manager
        
        # 处理多提示分割
        try:
            self.parent.fov_manager.process_all_tiles_multi_prompt()
            self.parent.status_label.setText("✅ FOV multi-prompt processing started")
        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to start FOV multi-prompt processing: {str(e)}")

    def on_merge_fov_multi_results(self):
        """
        合并FOV多提示分析结果 
        """
        # 1. 获取 FOV 管理器
        fov_manager = getattr(self.parent, 'fov_manager', None)
        if not fov_manager:
            QMessageBox.warning(self.parent, "Error", "FOV Manager not initialized.")
            return

        # 2. 检查是否有数据 (内存或磁盘)
        has_memory_data = hasattr(fov_manager, 'tile_multi_results') and bool(fov_manager.tile_multi_results)
        has_disk_data = False
        if hasattr(fov_manager, 'project_dir') and fov_manager.project_dir:
            import glob
            # 简单的检查是否存在对应的npz文件
            has_disk_data = len(list(fov_manager.project_dir.glob("tile_*_multi_data.npz"))) > 0

        if not has_memory_data and not has_disk_data:
            QMessageBox.warning(self.parent, "Warning", "No valid multi-prompt results found.\nPlease run 'Process FOV Multi-Prompt' first.")
            return

        # 3. 直接调用 fov_manager 的合并逻辑
        try:
            # 调用核心合并函数
            fov_manager.merge_tile_multi_results_with_edge_healing()
            
            manager_results = fov_manager.merged_multi_results
            if not manager_results:
                if hasattr(fov_manager, 'load_merged_multi_results_from_disk'):
                    fov_manager.load_merged_multi_results_from_disk()
                    manager_results = fov_manager.merged_multi_results

            if manager_results:
                self.fov_merged_results = {}
                
                # 映射并同步数据
                if 'stoma' in manager_results:
                    self.fov_merged_results['stomata'] = manager_results['stoma']
                
                if 'pavement-cell' in manager_results:
                    self.fov_merged_results['pavement'] = manager_results['pavement-cell']
                
                if 'area' in manager_results:
                    self.fov_merged_results['aperture'] = manager_results['area']
                
                # 5. 更新UI状态
                self._update_fov_display_buttons()
                
                # 自动显示第一个有结果的图层
                if 'stomata' in self.fov_merged_results:
                    self.display_fov_stomata_results()
                elif 'pavement' in self.fov_merged_results:
                    self.display_fov_pavement_results()
                
                # 获取报告引用，以便导出功能使用
                self.fov_comprehensive_report = fov_manager.comprehensive_report

        except Exception as e:
            print(f"Error calling fov_manager merge: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self.parent, "Merge Error", f"An error occurred during merging:\n{str(e)}")

    def _update_fov_display_buttons(self):
        """更新FOV结果显示按钮的可用状态"""
        has_stomata = 'stomata' in self.fov_merged_results
        has_pavement = 'pavement' in self.fov_merged_results
        has_aperture = 'aperture' in self.fov_merged_results
        
        if hasattr(self.parent, 'show_fov_stomata_btn'):
            self.parent.show_fov_stomata_btn.setEnabled(has_stomata)
        if hasattr(self.parent, 'show_fov_pavement_btn'):
            self.parent.show_fov_pavement_btn.setEnabled(has_pavement)
        if hasattr(self.parent, 'show_fov_aperture_btn'):
            self.parent.show_fov_aperture_btn.setEnabled(has_aperture)

    def enable_fov_result_buttons(self):
        """启用FOV结果切换按钮"""
        if hasattr(self.parent, 'show_fov_stomata_btn'):
            self.parent.show_fov_stomata_btn.setEnabled(True)
        if hasattr(self.parent, 'show_fov_pavement_btn'):
            self.parent.show_fov_pavement_btn.setEnabled(True)
        if hasattr(self.parent, 'show_fov_aperture_btn'):
            self.parent.show_fov_aperture_btn.setEnabled(True)
        if hasattr(self.parent, 'export_fov_comprehensive_btn'):
            self.parent.export_fov_comprehensive_btn.setEnabled(True)

    def display_fov_stomata_results(self):
        """
        显示 FOV 气孔合并结果
        """
        if self.fov_merged_results and 'stomata' in self.fov_merged_results:
            self._display_merged_fov_category('stomata', "Stomata")
        elif self.fov_multi_prompt_results and self.fov_multi_prompt_results.get('stomata', {}).get('masks'):
             QMessageBox.warning(self.parent, "Info", "Please click 'Merge FOV Multi-Results' first to see the stitched map.")
        else:
            QMessageBox.warning(self.parent, "Warning", "No Stomata results found.")

    def display_fov_pavement_results(self):
        """
        显示 FOV 表皮细胞合并结果
        """
        if self.fov_merged_results and 'pavement' in self.fov_merged_results:
            self._display_merged_fov_category('pavement', "Pavement Cells")
        elif self.fov_multi_prompt_results and self.fov_multi_prompt_results.get('pavement', {}).get('masks'):
             QMessageBox.warning(self.parent, "Info", "Please click 'Merge FOV Multi-Results' first.")
        else:
            QMessageBox.warning(self.parent, "Warning", "No Pavement Cell results found.")

    def display_fov_aperture_results(self):
        """
        显示 FOV 气孔开口合并结果
        """
        if self.fov_merged_results and 'aperture' in self.fov_merged_results:
            self._display_merged_fov_category('aperture', "Aperture")
        elif self.fov_multi_prompt_results and self.fov_multi_prompt_results.get('aperture', {}).get('masks'):
             QMessageBox.warning(self.parent, "Info", "Please click 'Merge FOV Multi-Results' first.")
        else:
            QMessageBox.warning(self.parent, "Warning", "No Aperture results found.")

    def _display_merged_fov_category(self, category_key, title):
        """
        通用方法：显示特定类别的FOV合并结果
        """
        # 1. 优先检查合并结果 (Merged Results)
        if self.fov_merged_results and category_key in self.fov_merged_results:
            # 获取 FOV 管理器
            fov_manager = getattr(self.parent, 'fov_manager', None)
            if not fov_manager:
                return

            key_map = {
                'stomata': 'stoma',
                'pavement': 'pavement-cell',
                'aperture': 'area'
            }
            
            prompt_key = key_map.get(category_key, category_key)
            
            fov_manager.display_merged_multi_results(prompt_key)
            
        else:
            msg = f"No merged results found for {category_key}. \nPlease run 'Merge FOV Multi-Results' first."
            QMessageBox.information(self.parent, "Info", msg)

    def on_export_fov_comprehensive_data(self):
        """导出FOV综合数据"""
        if not self.fov_comprehensive_report:
            QMessageBox.warning(self.parent, "Warning", "No FOV comprehensive data to export")
            return
        
        default_name = "fov_comprehensive_phenotype_data.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent, "Export FOV Comprehensive Data", 
            default_name, 
            "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                # 调用FOV管理器的导出方法
                if hasattr(self.parent, 'fov_manager'):
                    self.parent.fov_manager.export_comprehensive_data_multi()
                else:
                    # 备选方案：使用现有的表型计算器
                    if self.fov_stomata_results and self.fov_pavement_results:
                        stoma_masks = self.fov_stomata_results.get("masks", [])
                        pavement_masks = self.fov_pavement_results.get("masks", [])
                        aperture_masks = self.fov_aperture_results.get("masks", []) if self.fov_aperture_results else []
                        
                        stoma_boxes = self.fov_stomata_results.get("boxes", [])
                        pavement_boxes = self.fov_pavement_results.get("boxes", [])
                        aperture_boxes = self.fov_aperture_results.get("boxes", []) if self.fov_aperture_results else []
                        
                        stoma_confidences = self.fov_stomata_results.get("scores", [])
                        pavement_confidences = self.fov_pavement_results.get("scores", [])
                        aperture_confidences = self.fov_aperture_results.get("scores", []) if self.fov_aperture_results else []
                        
                        stoma_df = self.phenotype_calculator.calculate_basic_properties(stoma_masks, stoma_boxes, stoma_confidences)
                        pavement_df = self.phenotype_calculator.calculate_basic_properties(pavement_masks, pavement_boxes, pavement_confidences)
                        aperture_df = self.phenotype_calculator.calculate_basic_properties(aperture_masks, aperture_boxes, aperture_confidences) if aperture_masks else None
                        
                        export_path = self.phenotype_calculator.export_to_csv(
                            stoma_df, pavement_df, aperture_df, 
                            self.fov_comprehensive_report, file_path
                        )
                        
                        self.parent.status_label.setText(f"✅ FOV comprehensive data exported to {os.path.basename(export_path)}")
                        QMessageBox.information(
                            self.parent, "Success", 
                            f"FOV comprehensive phenotype data exported successfully!\nSaved to: {export_path}"
                        )
                    
            except Exception as e:
                QMessageBox.critical(self.parent, "Error", f"Failed to export FOV comprehensive data:\n{str(e)}")

    def on_show_fov_result_selection(self):
        """显示FOV结果选择对话框"""
        if not (self.fov_stomata_results or self.fov_pavement_results or self.fov_aperture_results):
            QMessageBox.warning(self.parent, "Warning", "No FOV results available")
            return
        
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
        
        dialog = QDialog(self.parent)
        dialog.setWindowTitle("Select FOV Results to Display")
        dialog.setFixedSize(450, 250)
        
        layout = QVBoxLayout(dialog)
        
        # 标题
        title_label = QLabel("FOV Multi-Prompt Processing Complete!")
        title_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #3b82f6; padding: 10px;")
        layout.addWidget(title_label)
        
        # 统计信息
        stomata_count = len(self.fov_stomata_results.get("masks", [])) if self.fov_stomata_results else 0
        pavement_count = len(self.fov_pavement_results.get("masks", [])) if self.fov_pavement_results else 0
        aperture_count = len(self.fov_aperture_results.get("masks", [])) if self.fov_aperture_results else 0
        
        stats_label = QLabel(
            f"• Stomata: {stomata_count} cells\n"
            f"• Pavement Cells: {pavement_count} cells\n"
            f"• Stomatal Apertures: {aperture_count} cells"
        )
        stats_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f8fafc; border-radius: 5px;")
        layout.addWidget(stats_label)
        
        # 提示
        prompt_label = QLabel("Select which results to display:")
        prompt_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px;")
        layout.addWidget(prompt_label)
        
        # 按钮
        button_layout = QHBoxLayout()
        
        stomata_btn = QPushButton("👁️ Show FOV Stomata")
        stomata_btn.clicked.connect(lambda: self.on_select_fov_results("stomata", dialog))
        stomata_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; 
                color: white; 
                padding: 12px; 
                border-radius: 8px; 
                font-weight: bold;
                font-size: 13px;
            } 
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        button_layout.addWidget(stomata_btn)
        
        pavement_btn = QPushButton("🧩 Show FOV Pavement Cells")
        pavement_btn.clicked.connect(lambda: self.on_select_fov_results("pavement", dialog))
        pavement_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981; 
                color: white; 
                padding: 12px; 
                border-radius: 8px; 
                font-weight: bold;
                font-size: 13px;
            } 
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        button_layout.addWidget(pavement_btn)
        
        aperture_btn = QPushButton("🕳️ Show FOV Apertures")
        aperture_btn.clicked.connect(lambda: self.on_select_fov_results("aperture", dialog))
        aperture_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6; 
                color: white; 
                padding: 12px; 
                border-radius: 8px; 
                font-weight: bold;
                font-size: 13px;
            } 
            QPushButton:hover {
                background-color: #7c3aed;
            }
        """)
        button_layout.addWidget(aperture_btn)
        
        layout.addLayout(button_layout)
        
        # 底部提示
        tip_label = QLabel("Tip: You can also switch between results using the FOV buttons in the left panel")
        tip_label.setStyleSheet("font-size: 12px; color: #64748b; padding: 10px; font-style: italic;")
        layout.addWidget(tip_label)
        
        dialog.exec_()

    def on_select_fov_results(self, result_type, dialog):
        """选择要显示的FOV结果"""
        dialog.close()
        
        if result_type == "stomata":
            self.display_fov_stomata_results()
        elif result_type == "pavement":
            self.display_fov_pavement_results()
        elif result_type == "aperture":
            self.display_fov_aperture_results()
        
        # 显示综合表型报告
        if self.fov_comprehensive_report:
            self.show_fov_comprehensive_report()

    def show_fov_comprehensive_report(self):
        """显示FOV综合表型报告"""
        if not self.fov_comprehensive_report:
            return
        
        # 在表型显示区域显示综合报告
        report = self.fov_comprehensive_report
        
        info_text = "【FOV COMPREHENSIVE PHENOTYPE ANALYSIS】\n"
        info_text += "=" * 70 + "\n\n"
        
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
        
        info_text += "\n💡 Tip: Use FOV buttons to switch between different segmentation results\n"
        info_text += "💡 Tip: Click 'Export FOV Data' to save comprehensive analysis"
        
        self.parent.phenotype_display.setText(info_text)