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
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(800, 600)
        self.verticalLayout = QVBoxLayout(Widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.input_site = QLineEdit(Widget)
        self.input_site.setObjectName(u"input_site")

        self.horizontalLayout.addWidget(self.input_site)

        self.input_user = QLineEdit(Widget)
        self.input_user.setObjectName(u"input_user")

        self.horizontalLayout.addWidget(self.input_user)

        self.input_pwd = QLineEdit(Widget)
        self.input_pwd.setObjectName(u"input_pwd")

        self.horizontalLayout.addWidget(self.input_pwd)

        self.btn_add = QPushButton(Widget)
        self.btn_add.setObjectName(u"btn_add")

        self.horizontalLayout.addWidget(self.btn_add)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.entry_list = QListWidget(Widget)
        QListWidgetItem(self.entry_list)
        self.entry_list.setObjectName(u"entry_list")

        self.verticalLayout.addWidget(self.entry_list)

        self.label_detail = QLabel(Widget)
        self.label_detail.setObjectName(u"label_detail")

        self.verticalLayout.addWidget(self.label_detail)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.btn_copy_user = QPushButton(Widget)
        self.btn_copy_user.setObjectName(u"btn_copy_user")

        self.horizontalLayout_2.addWidget(self.btn_copy_user)

        self.btn_copy_pwd = QPushButton(Widget)
        self.btn_copy_pwd.setObjectName(u"btn_copy_pwd")

        self.horizontalLayout_2.addWidget(self.btn_copy_pwd)

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
        self.input_site.setText(QCoreApplication.translate("Widget", u"\u7f51\u7ad9", None))
        self.input_site.setPlaceholderText(QCoreApplication.translate("Widget", u"\u7f51\u7ad9", None))
        self.input_user.setText(QCoreApplication.translate("Widget", u"\u7528\u6237\u540d", None))
        self.input_user.setPlaceholderText(QCoreApplication.translate("Widget", u"\u7528\u6237\u540d", None))
        self.input_pwd.setText(QCoreApplication.translate("Widget", u"\u5bc6\u7801", None))
        self.input_pwd.setPlaceholderText(QCoreApplication.translate("Widget", u"\u5bc6\u7801", None))
        self.btn_add.setText(QCoreApplication.translate("Widget", u"\u2795 \u6dfb\u52a0", None))

        __sortingEnabled = self.entry_list.isSortingEnabled()
        self.entry_list.setSortingEnabled(False)
        ___qlistwidgetitem = self.entry_list.item(0)
        ___qlistwidgetitem.setText(QCoreApplication.translate("Widget", u"____", None))
        self.entry_list.setSortingEnabled(__sortingEnabled)

        self.label_detail.setText(QCoreApplication.translate("Widget", u"\u70b9\u51fb\u6761\u76ee\u67e5\u770b\u8be6\u60c5", None))
        self.btn_copy_user.setText(QCoreApplication.translate("Widget", u"\U0001f4cb \U0000590d\U00005236\U00007528\U00006237\U0000540d", None))
        self.btn_copy_pwd.setText(QCoreApplication.translate("Widget", u"\U0001f511 \U0000590d\U00005236\U00005bc6\U00007801", None))
        self.btn_delete.setText(QCoreApplication.translate("Widget", u"\U0001f5d1 \U00005220\U00009664", None))
        self.btn_clear.setText(QCoreApplication.translate("Widget", u"\U0001f9f9 \U00006e05\U00007a7a\U00005168\U000090e8", None))
    # retranslateUi

