# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(884, 493)
        MainWindow.setMinimumSize(QtCore.QSize(800, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(10, 20, 889, 449))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_filters = QtWidgets.QWidget()
        self.tab_filters.setObjectName("tab_filters")
        self.label_filters_input = QtWidgets.QLabel(self.tab_filters)
        self.label_filters_input.setGeometry(QtCore.QRect(180, 30, 331, 231))
        self.label_filters_input.setAutoFillBackground(True)
        self.label_filters_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_input.setScaledContents(True)
        self.label_filters_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_input.setObjectName("label_filters_input")
        self.label_filters_output = QtWidgets.QLabel(self.tab_filters)
        self.label_filters_output.setGeometry(QtCore.QRect(520, 30, 331, 231))
        self.label_filters_output.setAutoFillBackground(True)
        self.label_filters_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_filters_output.setTextFormat(QtCore.Qt.PlainText)
        self.label_filters_output.setScaledContents(True)
        self.label_filters_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_filters_output.setObjectName("label_filters_output")
        self.pushButton_filters_load = QtWidgets.QPushButton(self.tab_filters)
        self.pushButton_filters_load.setGeometry(QtCore.QRect(20, 30, 121, 81))
        self.pushButton_filters_load.setObjectName("pushButton_filters_load")
        self.label = QtWidgets.QLabel(self.tab_filters)
        self.label.setGeometry(QtCore.QRect(30, 140, 71, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tab_filters)
        self.label_2.setGeometry(QtCore.QRect(30, 170, 71, 21))
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(self.tab_filters)
        self.groupBox.setGeometry(QtCore.QRect(30, 280, 751, 111))
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 311, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout.addWidget(self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_2.setGeometry(QtCore.QRect(380, 30, 311, 71))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(40, 20, 152, 39))
        self.label_5.setObjectName("label_5")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit.setGeometry(QtCore.QRect(210, 20, 51, 41))
        self.textEdit.setObjectName("textEdit")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 60, 311, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.comboBox_2 = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.horizontalLayout_2.addWidget(self.comboBox_2)
        self.tabWidget.addTab(self.tab_filters, "")
        self.tab_histograms = QtWidgets.QWidget()
        self.tab_histograms.setObjectName("tab_histograms")
        self.label_histograms_houtput = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_houtput.setGeometry(QtCore.QRect(450, 200, 261, 191))
        self.label_histograms_houtput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_houtput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_houtput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_houtput.setObjectName("label_histograms_houtput")
        self.pushButton_histograms_load = QtWidgets.QPushButton(self.tab_histograms)
        self.pushButton_histograms_load.setGeometry(QtCore.QRect(40, 20, 121, 81))
        self.pushButton_histograms_load.setObjectName("pushButton_histograms_load")
        self.label_histograms_input = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_input.setGeometry(QtCore.QRect(200, 20, 241, 171))
        self.label_histograms_input.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input.setObjectName("label_histograms_input")
        self.label_histograms_output = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_output.setGeometry(QtCore.QRect(450, 20, 261, 171))
        self.label_histograms_output.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output.setObjectName("label_histograms_output")
        self.label_10 = QtWidgets.QLabel(self.tab_histograms)
        self.label_10.setGeometry(QtCore.QRect(50, 160, 71, 21))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.tab_histograms)
        self.label_11.setGeometry(QtCore.QRect(50, 130, 71, 21))
        self.label_11.setObjectName("label_11")
        self.label_histograms_hinput = QtWidgets.QLabel(self.tab_histograms)
        self.label_histograms_hinput.setGeometry(QtCore.QRect(200, 200, 241, 191))
        self.label_histograms_hinput.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput.setObjectName("label_histograms_hinput")
        self.tabWidget.addTab(self.tab_histograms, "")
        self.tab_hybrid = QtWidgets.QWidget()
        self.tab_hybrid.setObjectName("tab_hybrid")
        self.label_histograms_input_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_input_2.setGeometry(QtCore.QRect(180, 20, 241, 171))
        self.label_histograms_input_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_input_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_input_2.setObjectName("label_histograms_input_2")
        self.label_histograms_hinput_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_hinput_2.setGeometry(QtCore.QRect(180, 200, 241, 191))
        self.label_histograms_hinput_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_hinput_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_histograms_hinput_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_hinput_2.setObjectName("label_histograms_hinput_2")
        self.label_histograms_output_2 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_histograms_output_2.setGeometry(QtCore.QRect(430, 20, 431, 371))
        self.label_histograms_output_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_histograms_output_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_histograms_output_2.setObjectName("label_histograms_output_2")
        self.label_12 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_12.setGeometry(QtCore.QRect(30, 110, 71, 21))
        self.label_12.setObjectName("label_12")
        self.pushButton_histograms_load_2 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_2.setGeometry(QtCore.QRect(20, 20, 121, 81))
        self.pushButton_histograms_load_2.setObjectName("pushButton_histograms_load_2")
        self.label_13 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_13.setGeometry(QtCore.QRect(30, 130, 71, 21))
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_14.setGeometry(QtCore.QRect(30, 310, 71, 21))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.tab_hybrid)
        self.label_15.setGeometry(QtCore.QRect(30, 290, 71, 21))
        self.label_15.setObjectName("label_15")
        self.pushButton_histograms_load_3 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_3.setGeometry(QtCore.QRect(20, 200, 121, 81))
        self.pushButton_histograms_load_3.setObjectName("pushButton_histograms_load_3")
        self.pushButton_histograms_load_4 = QtWidgets.QPushButton(self.tab_hybrid)
        self.pushButton_histograms_load_4.setGeometry(QtCore.QRect(20, 350, 121, 41))
        self.pushButton_histograms_load_4.setObjectName("pushButton_histograms_load_4")
        self.tabWidget.addTab(self.tab_hybrid, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 884, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_filters_input.setText(_translate("MainWindow", "Input image"))
        self.label_filters_output.setText(_translate("MainWindow", "Output image"))
        self.pushButton_filters_load.setText(_translate("MainWindow", "Load Image"))
        self.label.setText(_translate("MainWindow", "Name:"))
        self.label_2.setText(_translate("MainWindow", "Size:"))
        self.groupBox.setTitle(_translate("MainWindow", "Filter Settings"))
        self.label_3.setText(_translate("MainWindow", "       Add noise"))
        self.comboBox.setCurrentText(_translate("MainWindow", "Gaussian"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Gaussian"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Uniform"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Salt-papper"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Additional Parameters"))
        self.label_5.setText(_translate("MainWindow", "                          Mask size"))
        self.label_4.setText(_translate("MainWindow", "      Select Filter"))
        self.comboBox_2.setCurrentText(_translate("MainWindow", "Gaussian"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Gaussian"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Mean"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Median"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Sobel"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "Roberts"))
        self.comboBox_2.setItemText(5, _translate("MainWindow", "Prewitt"))
        self.comboBox_2.setItemText(6, _translate("MainWindow", "Canny"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_filters), _translate("MainWindow", "Filters"))
        self.label_histograms_houtput.setText(_translate("MainWindow", "Output Histogram"))
        self.pushButton_histograms_load.setText(_translate("MainWindow", "Load image"))
        self.label_histograms_input.setText(_translate("MainWindow", "Input image"))
        self.label_histograms_output.setText(_translate("MainWindow", "Output image"))
        self.label_10.setText(_translate("MainWindow", "Size:"))
        self.label_11.setText(_translate("MainWindow", "Name:"))
        self.label_histograms_hinput.setText(_translate("MainWindow", "Input Histogram"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_histograms), _translate("MainWindow", "Histograms"))
        self.label_histograms_input_2.setText(_translate("MainWindow", "Input image A"))
        self.label_histograms_hinput_2.setText(_translate("MainWindow", "Input image B"))
        self.label_histograms_output_2.setText(_translate("MainWindow", "Output image"))
        self.label_12.setText(_translate("MainWindow", "Name:"))
        self.pushButton_histograms_load_2.setText(_translate("MainWindow", "Load image A"))
        self.label_13.setText(_translate("MainWindow", "Size:"))
        self.label_14.setText(_translate("MainWindow", "Size:"))
        self.label_15.setText(_translate("MainWindow", "Name:"))
        self.pushButton_histograms_load_3.setText(_translate("MainWindow", "Load image B"))
        self.pushButton_histograms_load_4.setText(_translate("MainWindow", "Make Hybrid"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_hybrid), _translate("MainWindow", "Hybrid"))
