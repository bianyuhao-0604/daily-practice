import sys
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QFont, QColor
from ui_form import Ui_Widget   # ← 以 ui_form.py 里 class Ui_??? 的实际名字为准


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()   # ← 两处类名必须一致
        self.ui.setupUi(self)

        # 绑定按钮（hasattr 保护：没画的按钮自动跳过）
        if hasattr(self.ui, "btn_add"):
            self.ui.btn_add.clicked.connect(self.add_task)
        if hasattr(self.ui, "btn_done"):
            self.ui.btn_done.clicked.connect(self.toggle_done)
        if hasattr(self.ui, "btn_delete"):
            self.ui.btn_delete.clicked.connect(self.delete_task)
        if hasattr(self.ui, "btn_clear"):
            self.ui.btn_clear.clicked.connect(self.clear_done)

        # 输入框里按回车也能添加任务
        if hasattr(self.ui, "input_task"):
            self.ui.input_task.returnPressed.connect(self.add_task)

    def add_task(self):
        text = self.ui.input_task.text().strip()
        if not text:
            return
        self.ui.task_list.addItem(text)
        self.ui.input_task.clear()
        self.ui.input_task.setFocus()

    def toggle_done(self):
        item = self.ui.task_list.currentItem()
        if item is None:
            return
        font = QFont(item.font())
        if item.font().strikeOut():
            font.setStrikeOut(False)
            item.setFont(font)
            item.setForeground(QColor("black"))
        else:
            font.setStrikeOut(True)
            item.setFont(font)
            item.setForeground(QColor("gray"))

    def delete_task(self):
        item = self.ui.task_list.currentItem()
        if item is not None:
            self.ui.task_list.takeItem(self.ui.task_list.row(item))

    def clear_done(self):
        # 倒序遍历删除，避免删除时行号变化跳项
        for i in range(self.ui.task_list.count() - 1, -1, -1):
            item = self.ui.task_list.item(i)
            if item.font().strikeOut():
                self.ui.task_list.takeItem(i)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
