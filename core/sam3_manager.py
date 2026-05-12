import os
import torch
import gc
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QProgressDialog
from PyQt5.QtCore import QCoreApplication, Qt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Manager:
    def __init__(self, parent):
        self.parent = parent
        self.processor = None
        self.sam3_model = None

    def init_sam3_model(self, checkpoint_path=None):
        try:
            if self.processor is not None:
                reply = QMessageBox.question(
                    self.parent, "Switch SAM3 Model",
                    "A SAM3 model is already loaded. Unload it to free memory before loading new model?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    self.unload_sam3_model()
                else:
                    return

            current_file_path = os.path.abspath(__file__)
            current_dir = os.path.dirname(current_file_path) # .../core
            project_root = os.path.dirname(current_dir)      # .../mllm_new_1231
            
            bpe_path = os.path.join(project_root, "model", "sam3", "bpe_simple_vocab_16e6.txt.gz")
            
            if not os.path.exists(bpe_path):
                bpe_path, _ = QFileDialog.getOpenFileName(
                    self.parent, "Select BPE File (bpe_simple_vocab_16e6.txt.gz)", 
                    "", "GZ Files (*.gz);;Text Files (*.txt);;All Files (*.*)"
                )
                if not bpe_path or not os.path.exists(bpe_path):
                    QMessageBox.critical(self.parent, "Error", 
                        f"BPE file not found at:\n{bpe_path}\n\nPlease check your 'model/sam3' folder structure.")
                    return

            model_kwargs = {"bpe_path": bpe_path}
            if checkpoint_path and checkpoint_path.strip():
                model_kwargs["checkpoint_path"] = checkpoint_path.strip()

            progress = QProgressDialog("Loading SAM3 model...", "Cancel", 0, 100, self.parent)
            progress.setWindowTitle("Loading")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            def update_progress(value, max_value=100):
                progress.setValue(int((value / max_value) * 100))
                QCoreApplication.processEvents()

            update_progress(10)
            model = build_sam3_image_model(**model_kwargs)
            update_progress(60)

            def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
                freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
                t = torch.arange(end, device=freqs.device)
                freqs = torch.outer(t, freqs).float()
                freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
                return freqs_cis

            for block in model.backbone.vision_backbone.trunk.blocks:
                if not hasattr(block.attn, 'freqs_cis') or block.attn.freqs_cis is None:
                    head_dim = block.attn.head_dim if hasattr(block.attn, 'head_dim') else block.attn.dim // block.attn.num_heads
                    max_position = 4096
                    block.attn.freqs_cis = precompute_freqs_cis(head_dim, max_position)

            update_progress(80)
            self.processor = Sam3Processor(model)
            update_progress(100)
            self.sam3_model = model
            progress.close()

            # 启用 Unload 按钮
            if hasattr(self.parent, 'unload_model_btn'):
                self.parent.unload_model_btn.setEnabled(True)
            
            self.parent.status_label.setText("✅ SAM3 Model loaded successfully")
            self.parent.update_status()

        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self.parent, "Error", f"Failed to initialize SAM3 model: {str(e)}")
            self.processor = None
            self.sam3_model = None

    def unload_sam3_model(self):
        if self.processor is not None:
            try:
                if hasattr(self.processor, 'model'):
                    del self.processor.model
                del self.processor
                self.processor = None
                if self.sam3_model is not None:
                    del self.sam3_model
                    self.sam3_model = None
                if self.parent.state is not None:
                    del self.parent.state
                    self.parent.state = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print("SAM3 model memory freed")
                
                # 禁用 Unload 按钮
                if hasattr(self.parent, 'unload_model_btn'):
                    self.parent.unload_model_btn.setEnabled(False)
                
                self.parent.status_label.setText("🗑️ SAM3 Model unloaded")
                self.parent.update_status()  # 更新状态
            except Exception as e:
                print(f"Error unloading SAM3 model: {e}")