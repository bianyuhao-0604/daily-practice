from PySide6.QtWidgets import QApplication, QMainWindow
from ui_form import Ui_MainWindow
import sys

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 所有控件都挂在 self.ui 上 ↓↓↓
        for name in ["btn_0", "btn_1", "btn_2", "btn_3", "btn_4",
                     "btn_5", "btn_6", "btn_7", "btn_8", "btn_9", "btn_dot"]:
            if hasattr(self.ui, name):          # 保护：没画的按钮跳过
                getattr(self.ui, name).clicked.connect(self.on_digit)

        for op in ["btn_add", "btn_sub", "btn_mul", "btn_div"]:
            if hasattr(self.ui, op):
                getattr(self.ui, op).clicked.connect(self.on_operator)

        if hasattr(self.ui, "btn_eq"):
            self.ui.btn_eq.clicked.connect(self.on_equals)
        if hasattr(self.ui, "btn_clear"):
            self.ui.btn_clear.clicked.connect(self.on_clear)
        if hasattr(self.ui, "btn_back"):
            self.ui.btn_back.clicked.connect(self.on_back)

    def on_digit(self):
        self.ui.display.insert(self.sender().text())     # display 也要加 ui.

    def on_operator(self):
        mapping = {"÷": "/", "×": "*", "-": "-", "+": "+"}
        self.ui.display.insert(" " + mapping[self.sender().text()] + " ")

    def on_equals(self):
        try:
            expr = self.ui.display.text().replace("×", "*").replace("÷", "/")
            self.ui.display.setText(str(eval(expr)))
        except Exception:
            self.ui.display.setText("错误")

    def on_clear(self):
        self.ui.display.clear()

    def on_back(self):
        self.ui.display.backspace()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())




