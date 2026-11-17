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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QSizePolicy, QVBoxLayout,
    QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(577, 600)
        self.verticalLayout = QVBoxLayout(Widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_task = QLineEdit(Widget)
        self.input_task.setObjectName(u"input_task")

        self.horizontalLayout.addWidget(self.input_task)

        self.btn_add = QPushButton(Widget)
        self.btn_add.setObjectName(u"btn_add")

        self.horizontalLayout.addWidget(self.btn_add)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.task_list = QListWidget(Widget)
        QListWidgetItem(self.task_list)
        self.task_list.setObjectName(u"task_list")

        self.verticalLayout.addWidget(self.task_list)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_done = QPushButton(Widget)
        self.btn_done.setObjectName(u"btn_done")

        self.horizontalLayout_2.addWidget(self.btn_done)

        self.btn_delete = QPushButton(Widget)
        self.btn_delete.setObjectName(u"btn_delete")

        self.horizontalLayout_2.addWidget(self.btn_delete)

        self.btn_clear = QPushButton(Widget)
        self.btn_clear.setObjectName(u"btn_clear")

        self.horizontalLayout_2.addWidget(self.btn_clear)


        self.verticalLayout.addLayout(self.horizontalLayout_2)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.input_task.setText(QCoreApplication.translate("Widget", u"(\u6e05\u7a7a\u5360\u4f4d\u6587\u5b57)", None))
        self.input_task.setPlaceholderText(QCoreApplication.translate("Widget", u"\u8f93\u5165\u65b0\u4efb\u52a1\uff0c\u56de\u8f66\u6216\u70b9\u6dfb\u52a0", None))
        self.btn_add.setText(QCoreApplication.translate("Widget", u"\u6dfb\u52a0", None))

        __sortingEnabled = self.task_list.isSortingEnabled()
        self.task_list.setSortingEnabled(False)
        ___qlistwidgetitem = self.task_list.item(0)
        ___qlistwidgetitem.setText(QCoreApplication.translate("Widget", u"____", None))
        self.task_list.setSortingEnabled(__sortingEnabled)

        self.btn_done.setText(QCoreApplication.translate("Widget", u"\u2714 \u5b8c\u6210/\u53d6\u6d88", None))
        self.btn_delete.setText(QCoreApplication.translate("Widget", u"\U0001f5d1 \U00005220\U00009664", None))
        self.btn_clear.setText(QCoreApplication.translate("Widget", u"\u6e05\u9664\u5df2\u5b8c\u6210", None))
    # retranslateUi

