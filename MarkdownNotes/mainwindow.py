import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                               QMessageBox, QListWidgetItem)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence
from ui_form import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.current_file = None      # 当前打开的文件路径
        self.notes_dir = os.path.join(os.getcwd(), "notes")   # 笔记文件夹

        self._create_menus()
        self._bind_signals()

        self.update_statusbar()

    # ---------- 信号绑定 ----------
    def _bind_signals(self):
        if hasattr(self.ui, "file_list"):
            self.ui.file_list.itemClicked.connect(self.open_from_list)
            self.ui.file_list.itemDoubleClicked.connect(self.open_from_list)

        if hasattr(self.ui, "editor"):
            # 节流定时器：停顿 300ms 才刷新预览，避免每敲一个字符就渲染
            self.preview_timer = QTimer(self)
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(self.update_preview)
            self.ui.editor.textChanged.connect(self.on_text_changed)

        if hasattr(self.ui, "preview"):
            self.ui.preview.setReadOnly(True)   # 代码兜底，就算 Designer 忘了打勾也没事

    # ---------- 菜单栏 ----------
    def _create_menus(self):
        bar = self.menuBar()
        file_menu = bar.addMenu("文件(&F)")            # &F = Alt+F

        act_new = QAction("新建笔记", self)
        act_new.setShortcut(QKeySequence.New)          # Ctrl+N
        act_new.triggered.connect(self.new_note)
        file_menu.addAction(act_new)

        act_open = QAction("打开 .md 文件", self)
        act_open.setShortcut(QKeySequence.Open)        # Ctrl+O
        act_open.triggered.connect(self.open_file_dialog)
        file_menu.addAction(act_open)

        act_save = QAction("保存", self)
        act_save.setShortcut(QKeySequence.Save)        # Ctrl+S
        act_save.triggered.connect(self.save_file)
        file_menu.addAction(act_save)

        act_saveas = QAction("另存为...", self)
        act_saveas.setShortcut(QKeySequence.SaveAs)    # Ctrl+Shift+S
        act_saveas.triggered.connect(self.save_as)
        file_menu.addAction(act_saveas)

        file_menu.addSeparator()
        act_exit = QAction("退出", self)
        act_exit.setShortcut(QKeySequence.Quit)        # Ctrl+Q
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

    # ---------- 实时预览 ----------
    def on_text_changed(self):
        self.preview_timer.start(300)   # 重新计时：停顿 0.3 秒才真刷

    def update_preview(self):
        self.ui.preview.setMarkdown(self.ui.editor.toPlainText())

    # ---------- 新建 ----------
    def new_note(self):
        self.ui.editor.clear()
        self.ui.preview.clear()
        self.current_file = None
        self.update_statusbar()

    # ---------- 打开 ----------
    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 Markdown 文件", "",
            "Markdown 文件 (*.md *.markdown);;所有文件 (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.ui.editor.setPlainText(f.read())
            self.current_file = path
            self.update_preview()
            self.update_statusbar()
        except Exception as e:
            QMessageBox.critical(self, "打开失败", str(e))

    def open_from_list(self, item: QListWidgetItem):
        path = item.data(Qt.UserRole)   # 取出挂在 item 上的真实路径
        if path and os.path.exists(path):
            self.load_file(path)

    # ---------- 左侧文件列表 ----------
    def refresh_file_list(self):
        self.ui.file_list.clear()
        if not os.path.exists(self.notes_dir):
            return
        for name in sorted(os.listdir(self.notes_dir)):
            if name.endswith(".md"):
                full = os.path.join(self.notes_dir, name)
                item = QListWidgetItem(f"📄 {name}")
                item.setData(Qt.UserRole, full)
                item.setToolTip(full)
                self.ui.file_list.addItem(item)

    # ---------- 保存 ----------
    def save_file(self):
        if self.current_file is None:
            self.save_as()
            return
        self._write_to(self.current_file)

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", self.notes_dir,
            "Markdown 文件 (*.md);;所有文件 (*)"
        )
        if path:
            if not path.endswith(".md"):
                path += ".md"
            self._write_to(path)

    def _write_to(self, path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.ui.editor.toPlainText())
            self.current_file = path
            self.update_statusbar()
            self.refresh_file_list()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    # ---------- 关闭前拦截 ----------
    def closeEvent(self, event):
        if self.ui.editor.toPlainText().strip():
            answer = QMessageBox.question(
                self, "未保存",
                "编辑器里还有内容，确定退出吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if answer == QMessageBox.StandardButton.No:
                event.ignore()      # 拒绝关闭
                return
        event.accept()

    # ---------- 状态栏 ----------
    def update_statusbar(self):
        text = self.current_file if self.current_file else "未打开任何文件"
        self.statusBar().showMessage(f"当前文件: {text}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    widget.refresh_file_list()      # 启动时加载左侧列表
    sys.exit(app.exec())
