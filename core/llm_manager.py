# core/llm_manager.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
import gc
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QFileDialog
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QTextCursor
from ui.widegets import LLMGenerator  
from utils.helpers import escape_html, format_chat_content


class LLMManager:
    def __init__(self, parent):
        self.parent = parent
        self.llm_model = None
        self.llm_tokenizer = None
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        self.project_root = os.path.dirname(current_dir)
        self.default_model_path = os.path.join(self.project_root, "model", "stomata_8B")
        self.llm_model_name = self.default_model_path
        self.llm_chat_history = []
        self.llm_thread = None
        self.is_llm_loading = False
        self.is_llm_generating = False
        self.last_llm_use_time = 0
        self.last_unload_ask_time = 0
        self.llm_load_time = 0
        self.auto_unload_check_interval = 600
        self.initial_unload_check_delay = 900

    def on_model_selected(self, index):
        if index == 0:
            self.parent.llm_path_input.setText(self.default_model_path)
        elif index == 1:
            self.parent.llm_path_input.setText("")
            self.parent.llm_path_input.setFocus()

    def on_browse_llm(self):
        dir_path = QFileDialog.getExistingDirectory(
            self.parent, "Select LLM Model Directory",
            self.parent.llm_path_input.text() or ""
        )
        if dir_path:
            self.parent.llm_path_input.setText(dir_path)

    def thorough_unload_llm(self):
        """彻底卸载 LLM 模型并释放 GPU 显存（核心强化函数，适用于所有场景）"""
        if self.llm_model is None:
            return True

        try:
            # 1. 强制终止正在进行的生成线程（即使正在流式输出）
            if self.llm_thread:
                if self.llm_thread.isRunning():
                    self.llm_thread.terminate()
                    self.llm_thread.wait(5000)  # 增加等待时间至 5 秒，确保线程完全退出
                self.llm_thread = None

            # 2. 删除流式生成中的临时变量（防止引用残留）
            if hasattr(self, '_current_ai_response'):
                del self._current_ai_response
            if hasattr(self, '_current_html_end'):
                del self._current_html_end

            # 3. 删除模型和 tokenizer 的显式引用
            if hasattr(self.llm_model, 'hf_device_map'):
                del self.llm_model.hf_device_map
            del self.llm_model
            del self.llm_tokenizer

            self.llm_model = None
            self.llm_tokenizer = None

            # 4. 强制多次垃圾回收
            gc.collect()
            gc.collect()  # 第二次调用以处理更深层的循环引用

            # 5. 彻底清理 CUDA 资源
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # 再次清空回收碎片

            # 6. 全面重置状态
            self.llm_chat_history = []
            self.is_llm_generating = False
            self.is_llm_loading = False
            self.llm_load_time = 0
            self.last_llm_use_time = 0
            self.last_unload_ask_time = 0

            print("LLM model fully unloaded and GPU memory completely released.")
            return True

        except Exception as e:
            print(f"Error during thorough LLM unloading: {e}")
            # 即使出错，也尽力回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    def load_llm(self):
        new_path = self.parent.llm_path_input.text().strip()
        if not new_path:
            QMessageBox.warning(self.parent, "Warning", "Please enter the LLM model path or HuggingFace repo ID")
            return

        # 路径对比：仅完全一致才跳过卸载
        need_unload = True
        if self.llm_model is not None:
            current_norm = os.path.normpath(self.llm_model_name.strip())
            new_norm = os.path.normpath(new_path.strip())
            if current_norm == new_norm:
                need_unload = False
                QMessageBox.information(self.parent, "Info", "The same model is already loaded. No need to reload.")
            else:
                reply = QMessageBox.question(
                    self.parent, "Switch Model",
                    f"Current model: {os.path.basename(self.llm_model_name)}\n"
                    f"New model: {os.path.basename(new_path)}\n\n"
                    "Unload current model and load the new one?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    return

        # 执行卸载（如需要）
        if need_unload and self.llm_model is not None:
            self.parent.set_loading(True, "Unloading previous model to free GPU memory...")
            unload_progress = QProgressDialog("Unloading previous model...", None, 0, 0, self.parent)
            unload_progress.setWindowModality(Qt.WindowModal)
            unload_progress.show()
            QCoreApplication.processEvents()

            success = self.thorough_unload_llm()
            if not success:
                QMessageBox.warning(self.parent, "Warning", "Partial failure in unloading previous model. Proceeding anyway...")

            unload_progress.close()
            self.parent.set_loading(False)
            time.sleep(2.0)  # 增加缓冲时间，确保显存彻底释放
            self.parent.update_resource_info()

        # 加载新模型
        self.llm_model_name = new_path
        self.parent.set_loading(True, "Loading LLM model, please wait...")
        self.is_llm_loading = True

        progress = QProgressDialog("Loading LLM model...", "Cancel", 0, 100, self.parent)
        progress.setWindowTitle("Loading LLM")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        try:
            progress.setValue(10)
            QCoreApplication.processEvents()
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True,fix_mistral_regex=True)

            progress.setValue(30)
            QCoreApplication.processEvents()
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )

            progress.setValue(95)
            QCoreApplication.processEvents()

            model_name_short = os.path.basename(self.llm_model_name) or self.llm_model_name
            self.parent.llm_status_label.setText("🟢 Status: Loaded and Ready")
            self.parent.load_llm_btn.setEnabled(False)
            self.parent.unload_llm_btn.setEnabled(True)
            self.parent.send_chat_btn.setEnabled(True)
            self.parent.chat_input.setEnabled(True)
            self.parent.clear_chat_btn.setEnabled(True)

            welcome_msg = f"""🌿 **PhenoGuard AI Assistant Loaded Successfully**

**Model:** {model_name_short}  
**Memory Usage:** {self.get_model_memory_usage():.2f} GB  
**Context Length:** 2048 tokens  

**Developed by:** DAAI Team, Zhejiang University  
**Core Expertise:** Panoramic multimodal analysis of plant leaf epidermal cells  
**Capabilities:** Comprehensive in-depth analysis from phenotype quantification to genotype interpretation  

---

🔬 **Specialized Topics**

• Leaf epidermal structures: stomata, guard cells, pavement cells, trichomes (morphology & function)  
• Advanced image segmentation and phenotypic measurement: instance/semantic segmentation techniques and best practices  
• Multimodal deep analysis: integration of phenotypic data with genotypic associations  
• Technical implementation: Python-based image processing, deep learning model training, and PyQt5 GUI development  

📊 **Broad Knowledge Coverage:** Plant biology, computational biology, and any related academic inquiries  

---

**Please feel free to ask your professional questions. I am ready to provide rigorous, evidence-based responses.**  
(Recommendation: In the GUI, implement fade-in animations and a subtle pulsating leaf icon to enhance visual engagement)"""

            self.llm_chat_history = [{"role": "assistant", "content": welcome_msg}]
            self.append_message("assistant", welcome_msg)

            progress.setValue(100)
            self.parent.status_label.setText("✅ LLM model loaded successfully")
            self.last_llm_use_time = time.time()
            self.llm_load_time = time.time()
            self.last_unload_ask_time = 0

        except Exception as e:
            progress.close()
            QMessageBox.critical(self.parent, "Error", f"LLM model loading failed:\n{str(e)}")
            self.parent.llm_status_label.setText("🔴 Status: Loading failed")
            self.thorough_unload_llm()
        finally:
            progress.close()
            self.parent.set_loading(False)
            self.is_llm_loading = False
            self.parent.update_resource_info()

    def get_model_memory_usage(self):
        if self.llm_model is None:
            return 0.0
        param_count = sum(p.numel() for p in self.llm_model.parameters())
        memory_gb = (param_count * 2) / (1024 ** 3)  # float16: 2 bytes per param
        return memory_gb

    def on_unload_llm(self):
        """Unload 按钮点击处理：支持生成中/生成后强制彻底卸载"""
        if self.llm_model is None:
            QMessageBox.information(self.parent, "Info", "No LLM model is currently loaded.")
            return

        # 若正在生成，询问是否强制中断
        if self.is_llm_generating:
            reply = QMessageBox.question(
                self.parent, "Force Unload",
                "LLM is currently generating a response.\n"
                "Force stop generation and unload the model?\n\n"
                "This will terminate the current response immediately.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                return

        self.parent.set_loading(True, "Thoroughly unloading LLM and releasing GPU memory...")

        progress = QProgressDialog("Releasing GPU memory...", None, 0, 0, self.parent)
        progress.setWindowTitle("Unloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QCoreApplication.processEvents()

        try:
            success = self.thorough_unload_llm()

            # UI 全面恢复初始状态
            self.parent.llm_status_label.setText("🟡 Status: Not Loaded")
            self.parent.load_llm_btn.setEnabled(True)
            self.parent.unload_llm_btn.setEnabled(False)
            self.parent.send_chat_btn.setEnabled(False)
            self.parent.chat_input.setEnabled(False)
            self.parent.clear_chat_btn.setEnabled(False)

            self.parent.chat_display.clear()
            self.parent.chat_display.setPlaceholderText("Load LLM and start chatting...")
            self.parent.token_counter.setText("Tokens: --")

            status_msg = "✅ LLM thoroughly unloaded, GPU memory fully released"
            self.parent.status_label.setText(status_msg)
            self.parent.update_resource_info()

            if not success:
                QMessageBox.warning(self.parent, "Partial Failure", "Some resources may not have been fully released.")

        except Exception as e:
            QMessageBox.critical(self.parent, "Error", f"Failed to unload LLM:\n{str(e)}")
        finally:
            progress.close()
            self.parent.set_loading(False)

    # 以下聊天相关方法保持不变（已验证可靠）
    def on_send_chat(self):
        if self.llm_model is None:
            QMessageBox.warning(self.parent, "Warning", "Please load LLM first")
            return
        prompt = self.parent.chat_input.text().strip()
        if not prompt:
            return
        self.parent.chat_input.clear()
        self.parent.send_chat_btn.setEnabled(False)
        self.parent.chat_input.setEnabled(False)
        self.is_llm_generating = True
        self.parent.llm_status_label.setText("🟡 Generating...")
        self.append_message("user", prompt)
        self.llm_chat_history.append({"role": "user", "content": prompt})
        self.llm_thread = LLMGenerator(self.llm_model, self.llm_tokenizer, prompt)
        self.llm_thread.new_token.connect(self.on_new_token)
        self.llm_thread.finished.connect(self.on_llm_finished)
        self.llm_thread.error.connect(self.on_llm_error)
        self.llm_thread.start()
        self.last_llm_use_time = time.time()

    def append_message(self, role, content):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        if role == "user":
            title = "👤 You"
            css_class = "user-message"
        else:
            title = "🤖 AI"
            css_class = "assistant-message"
        formatted_content = format_chat_content(content)
        html = f'''
<div class="message-container">
    <div class="{css_class}">
        <div class="message-title">{title} <span style="font-size:10px; color:#94a3b8;">{timestamp}</span></div>
        <div class="message-content">{formatted_content}</div>
    </div>
</div>
'''
        current_html = self.parent.chat_display.toHtml()
        self.parent.chat_display.setHtml(current_html + html)
        scrollbar = self.parent.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_new_token(self, token):
        if not hasattr(self, '_current_ai_response'):
            self._current_ai_response = ""
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            start_html = f'''
<div class="message-container">
    <div class="assistant-message">
        <div class="message-title">🤖 AI <span style="font-size:10px; color:#94a3b8;">{timestamp}</span></div>
        <div class="message-content">
'''
            self._current_html_end = self.parent.chat_display.toHtml()
            self.parent.chat_display.setHtml(self._current_html_end + start_html)
        self._current_ai_response += token
        escaped_token = escape_html(token)
        cursor = self.parent.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(escaped_token)
        self.parent.chat_display.setTextCursor(cursor)
        token_count = len(self._current_ai_response.split())
        self.parent.token_counter.setText(f"Tokens: {token_count}")
        QCoreApplication.processEvents()

    def on_llm_finished(self):
        self.is_llm_generating = False
        self.parent.send_chat_btn.setEnabled(True)
        self.parent.chat_input.setEnabled(True)
        self.parent.chat_input.setFocus()
        self.parent.llm_status_label.setText("🟢 Ready")
        if hasattr(self, '_current_ai_response'):
            end_html = '''
    </div>
</div>
</div>
'''
            current_html = self.parent.chat_display.toHtml()
            if not current_html.strip().endswith('</div></div></div>'):
                self.parent.chat_display.setHtml(current_html + end_html)
            self.llm_chat_history.append({"role": "assistant", "content": self._current_ai_response})
            del self._current_ai_response
            if hasattr(self, '_current_html_end'):
                del self._current_html_end
        scrollbar = self.parent.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        if len(self.llm_chat_history) > 20:
            self.llm_chat_history = self.llm_chat_history[-20:]
        self.parent.update_resource_info()

    def on_llm_error(self, error):
        self.is_llm_generating = False
        self.parent.send_chat_btn.setEnabled(True)
        self.parent.chat_input.setEnabled(True)
        self.parent.llm_status_label.setText("🔴 Status: Error")
        error_msg = f"\n\n<font color='red'>⚠️ Generation Error: {error}</font>"
        cursor = self.parent.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(error_msg)
        self.parent.chat_display.setTextCursor(cursor)
        QMessageBox.critical(self.parent, "LLM Generation Error", f"Failed to generate response:\n{error}")

    def on_clear_chat(self):
        if self.llm_chat_history:
            reply = QMessageBox.question(self.parent, "Clear Chat",
                                         "Clear all chat history?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.llm_chat_history = []
                self.parent.chat_display.clear()
                self.parent.chat_display.setPlaceholderText("Chat history cleared. Start a new conversation...")
                self.parent.token_counter.setText("Tokens: --")