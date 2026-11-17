import sys
import os
import json
from PySide6.QtWidgets import (QApplication, QWidget, QLineEdit,
                               QMessageBox)
from PySide6.QtCore import Qt
from ui_form import Ui_Widget   # ← 以 ui_form.py 里 class Ui_??? 的实际名字为准

DATA_FILE = "passwords.json"   # 数据文件，跑起来后会出现在项目目录


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.data = {}   # 形如 {"GitHub": {"user": "alice", "pwd": "123"}}

        # 密码输入框设为掩码模式（输入显示 ●●●●）
        if hasattr(self.ui, "input_pwd"):
            self.ui.input_pwd.setEchoMode(QLineEdit.Password)

        # 绑定按钮
        if hasattr(self.ui, "btn_add"):
            self.ui.btn_add.clicked.connect(self.add_entry)
        if hasattr(self.ui, "btn_copy_user"):
            self.ui.btn_copy_user.clicked.connect(self.copy_username)
        if hasattr(self.ui, "btn_copy_pwd"):
            self.ui.btn_copy_pwd.clicked.connect(self.copy_password)
        if hasattr(self.ui, "btn_delete"):
            self.ui.btn_delete.clicked.connect(self.delete_entry)
        if hasattr(self.ui, "btn_clear"):
            self.ui.btn_clear.clicked.connect(self.clear_all)

        # 选中列表项变化时刷新详情（信号自带 item 参数）
        if hasattr(self.ui, "entry_list"):
            self.ui.entry_list.currentItemChanged.connect(self.show_detail)

        # 网站输入框按回车也能添加
        if hasattr(self.ui, "input_site"):
            self.ui.input_site.returnPressed.connect(self.add_entry)

        self.load_data()      # 启动时读档
        self.refresh_list()   # 列表渲染

    # ---------- 数据读写 ----------
    def load_data(self):
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}   # 文件损坏时优雅降级为空

    def save_data(self):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def closeEvent(self, event):
        """窗口关闭前自动调用——保存数据的好时机"""
        self.save_data()
        event.accept()   # 接受关闭请求

    # ---------- 列表渲染 ----------
    def refresh_list(self):
        self.ui.entry_list.clear()
        for site, info in self.data.items():
            item_text = f"{site}  —  {info['user']}"
            item = self.ui.entry_list.addItem(item_text)
            # 取回刚添加的那一项，把完整数据"挂在"它背后
            last = self.ui.entry_list.item(self.ui.entry_list.count() - 1)
            last.setData(Qt.UserRole, {"site": site, **info})

    # ---------- 添加条目 ----------
    def add_entry(self):
        site = self.ui.input_site.text().strip()
        user = self.ui.input_user.text().strip()
        pwd = self.ui.input_pwd.text().strip()

        if not site or not user or not pwd:
            QMessageBox.warning(self, "提示", "网站 / 用户名 / 密码都要填哦")
            return
        if site in self.data:
            QMessageBox.warning(self, "提示", f"{site} 已存在，如需修改请先删除再添加")
            return

        self.data[site] = {"user": user, "pwd": pwd}
        self.save_data()          # 添加后立刻存盘，即使突然关机也不丢
        self.refresh_list()
        self.ui.input_site.clear()
        self.ui.input_user.clear()
        self.ui.input_pwd.clear()
        self.ui.input_site.setFocus()

    # ---------- 取当前选中项背后的数据 ----------
    def _current_entry(self):
        item = self.ui.entry_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)   # 取出 setData 塞进去的字典

    # ---------- 复制到剪贴板 ----------
    def copy_username(self):
        entry = self._current_entry()
        if entry is None:
            return
        QApplication.clipboard().setText(entry["user"])
        self.ui.label_detail.setText(f"✅ 已复制用户名：{entry['user']}")

    def copy_password(self):
        entry = self._current_entry()
        if entry is None:
            return
        QApplication.clipboard().setText(entry["pwd"])
        self.ui.label_detail.setText(f"✅ 已复制 {entry['site']} 的密码，去粘贴吧")

    # ---------- 删除选中 ----------
    def delete_entry(self):
        entry = self._current_entry()
        if entry is None:
            return
        confirm = QMessageBox.question(
            self, "确认删除",
            f"确定删除 {entry['site']} 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.data.pop(entry["site"])
            self.save_data()
            self.refresh_list()
            self.ui.label_detail.setText("已删除")

    # ---------- 清空全部（二次确认）----------
    def clear_all(self):
        confirm = QMessageBox.question(
            self, "危险操作",
            f"将删除全部 {len(self.data)} 条记录，无法恢复！确定吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.data.clear()
            self.save_data()
            self.refresh_list()
            self.ui.label_detail.setText("已清空全部")

    # ---------- 选中变化时显示详情 ----------
    def show_detail(self, current, previous):
        if current is None:
            self.ui.label_detail.setText("点击条目查看详情")
            return
        info = current.data(Qt.UserRole)
        self.ui.label_detail.setText(
            f"选中：{info['site']} / {info['user']} / {'•' * len(info['pwd'])}"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
