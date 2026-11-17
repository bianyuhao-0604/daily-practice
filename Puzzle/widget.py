import sys
import random
from PySide6.QtWidgets import (QApplication, QWidget, QPushButton,
                               QGridLayout, QMessageBox)
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from ui_form import Ui_Widget   # ← 以 ui_form.py 里 class Ui_??? 的实际名字为准

N = 3   # 拼图规格：3×3（以后改成 4 就是 4×4）


class MainWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        self.buttons = []      # 9 个按钮，按位置编号 0~8
        self.board = []        # 棋盘数据：board[i] = 该位置的数字，0 表示空格
        self.moves = 0
        self.seconds = 0
        self.won = False

        self._create_buttons()

        # 新知识：定时器，每 1000 毫秒触发一次 on_tick
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        if hasattr(self.ui, "btn_restart"):
            self.ui.btn_restart.clicked.connect(self.restart)

        self.restart()

    # ---------- 用代码动态创建 N×N 按钮网格 ----------
    def _create_buttons(self):
        grid = getattr(self.ui, "gridLayout", None)   # Designer 里的空网格
        if grid is None:
            return
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        for i in range(N * N):
            btn = QPushButton()
            btn.setMinimumSize(70, 70)
            btn.setFont(font)
            btn.clicked.connect(self.on_tile_clicked)   # 9 个按钮共用一个槽
            grid.addWidget(btn, i // N, i % N)          # 按行列放进网格
            self.buttons.append(btn)

    # ---------- 求与位置 i 相邻的位置 ----------
    def _neighbors(self, i):
        row, col = i // N, i % N
        result = []
        if row > 0:     result.append(i - N)   # 上
        if row < N - 1: result.append(i + N)   # 下
        if col > 0:     result.append(i - 1)   # 左
        if col < N - 1: result.append(i + 1)   # 右
        return result

    # ---------- 新开一局 ----------
    def restart(self):
        self.board = list(range(1, N * N)) + [0]   # 复原状态 [1,2,...,8,0]
        for _ in range(100):                        # 随机走 100 步打乱 → 保证有解
            empty = self.board.index(0)
            target = random.choice(self._neighbors(empty))
            self.board[empty], self.board[target] = self.board[target], self.board[empty]

        self.moves = 0
        self.seconds = 0
        self.won = False
        self.timer.start(1000)
        self.refresh()

    # ---------- 点击数字块 ----------
    def on_tile_clicked(self):
        if self.won:
            return
        i = self.buttons.index(self.sender())      # 点的是哪个位置
        empty = self.board.index(0)
        if i in self._neighbors(empty):            # 只有挨着空格的块能滑
            self.board[i], self.board[empty] = self.board[empty], self.board[i]
            self.moves += 1
            self.refresh()
            self.check_win()

    # ---------- 把棋盘数据画到界面上 ----------
    def refresh(self):
        for i in range(N * N):
            n = self.board[i]
            self.buttons[i].setText("" if n == 0 else str(n))
        self._update_status()

    def _update_status(self):
        m, s = divmod(self.seconds, 60)
        if hasattr(self.ui, "label_time"):
            self.ui.label_time.setText(f"⏱ {m:02d}:{s:02d}")
        if hasattr(self.ui, "label_moves"):
            self.ui.label_moves.setText(f"🔢 {self.moves} 步")

    # ---------- 每秒触发一次 ----------
    def on_tick(self):
        self.seconds += 1
        self._update_status()

    # ---------- 胜利判定 ----------
    def check_win(self):
        if self.board == list(range(1, N * N)) + [0]:
            self.won = True
            self.timer.stop()
            m, s = divmod(self.seconds, 60)
            QMessageBox.information(
                self, "🎉 恭喜通关",
                f"用时 {m:02d}:{s:02d}，共 {self.moves} 步\n点「重新开始」再来一局！"
            )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())
