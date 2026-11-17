# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.12.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(800, 600)
        self.verticalLayout = QVBoxLayout(Widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label_time = QLabel(Widget)
        self.label_time.setObjectName(u"label_time")

        self.horizontalLayout.addWidget(self.label_time)

        self.label_moves = QLabel(Widget)
        self.label_moves.setObjectName(u"label_moves")

        self.horizontalLayout.addWidget(self.label_moves)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")

        self.verticalLayout.addLayout(self.gridLayout)

        self.btn_restart = QPushButton(Widget)
        self.btn_restart.setObjectName(u"btn_restart")

        self.verticalLayout.addWidget(self.btn_restart)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.label_time.setText(QCoreApplication.translate("Widget", u"\u23f1 00:00", None))
        self.label_moves.setText(QCoreApplication.translate("Widget", u"\U0001f522 0 \U00006b65", None))
        self.btn_restart.setText(QCoreApplication.translate("Widget", u"\U0001f504 \U000091cd\U000065b0\U00005f00\U000059cb", None))
    # retranslateUi

