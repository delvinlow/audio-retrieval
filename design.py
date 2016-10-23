# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AudioSearch.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1280, 751)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("assets/ic_window_icon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.btn_picker = QtGui.QPushButton(self.centralwidget)
        self.btn_picker.setMinimumSize(QtCore.QSize(147, 32))
        self.btn_picker.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.btn_picker.setFocusPolicy(QtCore.Qt.TabFocus)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8("assets/ic_video.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_picker.setIcon(icon1)
        self.btn_picker.setObjectName(_fromUtf8("btn_picker"))
        self.horizontalLayout.addWidget(self.btn_picker)
        self.btn_search = QtGui.QPushButton(self.centralwidget)
        self.btn_search.setMinimumSize(QtCore.QSize(104, 32))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8("../../ImageSearch/assets/ic_search.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_search.setIcon(icon2)
        self.btn_search.setObjectName(_fromUtf8("btn_search"))
        self.horizontalLayout.addWidget(self.btn_search)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setText(_fromUtf8(""))
        self.label.setPixmap(QtGui.QPixmap(_fromUtf8("assets/ic_music.png")))
        self.label.setScaledContents(False)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_2.addWidget(self.label)
        self.label_3 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_2.addWidget(self.label_3)
        self.checkBox_energy = QtGui.QCheckBox(self.centralwidget)
        self.checkBox_energy.setChecked(True)
        self.checkBox_energy.setObjectName(_fromUtf8("checkBox_energy"))
        self.horizontalLayout_2.addWidget(self.checkBox_energy)
        self.doubleSpinBoxEnergy = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxEnergy.setDecimals(1)
        self.doubleSpinBoxEnergy.setProperty("value", 1.0)
        self.doubleSpinBoxEnergy.setObjectName(_fromUtf8("doubleSpinBoxEnergy"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxEnergy)
        self.checkBox_zerocrossing = QtGui.QCheckBox(self.centralwidget)
        self.checkBox_zerocrossing.setChecked(True)
        self.checkBox_zerocrossing.setObjectName(_fromUtf8("checkBox_zerocrossing"))
        self.horizontalLayout_2.addWidget(self.checkBox_zerocrossing)
        self.doubleSpinBoxZeroCrossing = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxZeroCrossing.setDecimals(1)
        self.doubleSpinBoxZeroCrossing.setProperty("value", 1.0)
        self.doubleSpinBoxZeroCrossing.setObjectName(_fromUtf8("doubleSpinBoxZeroCrossing"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxZeroCrossing)
        self.checkBox_spect = QtGui.QCheckBox(self.centralwidget)
        self.checkBox_spect.setChecked(True)
        self.checkBox_spect.setObjectName(_fromUtf8("checkBox_spect"))
        self.horizontalLayout_2.addWidget(self.checkBox_spect)
        self.doubleSpinBoxSpect = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxSpect.setDecimals(1)
        self.doubleSpinBoxSpect.setProperty("value", 1.0)
        self.doubleSpinBoxSpect.setObjectName(_fromUtf8("doubleSpinBoxSpect"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxSpect)
        self.checkBox_mfcc = QtGui.QCheckBox(self.centralwidget)
        self.checkBox_mfcc.setChecked(True)
        self.checkBox_mfcc.setObjectName(_fromUtf8("checkBox_mfcc"))
        self.horizontalLayout_2.addWidget(self.checkBox_mfcc)
        self.doubleSpinBoxMFCC = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxMFCC.setDecimals(1)
        self.doubleSpinBoxMFCC.setProperty("value", 1.0)
        self.doubleSpinBoxMFCC.setObjectName(_fromUtf8("doubleSpinBoxMFCC"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxMFCC)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_4 = QtGui.QLabel(self.centralwidget)
        self.label_4.setText(_fromUtf8(""))
        self.label_4.setPixmap(QtGui.QPixmap(_fromUtf8("assets/ic_camera.png")))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.horizontalLayout_2.addWidget(self.label_4)
        self.label_visual = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_visual.setFont(font)
        self.label_visual.setObjectName(_fromUtf8("label_visual"))
        self.horizontalLayout_2.addWidget(self.label_visual)
        self.checkBoxColorHist = QtGui.QCheckBox(self.centralwidget)
        self.checkBoxColorHist.setEnabled(True)
        self.checkBoxColorHist.setChecked(True)
        self.checkBoxColorHist.setObjectName(_fromUtf8("checkBoxColorHist"))
        self.horizontalLayout_2.addWidget(self.checkBoxColorHist)
        self.doubleSpinBoxColorHist = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxColorHist.setDecimals(1)
        self.doubleSpinBoxColorHist.setSingleStep(0.5)
        self.doubleSpinBoxColorHist.setProperty("value", 3.0)
        self.doubleSpinBoxColorHist.setObjectName(_fromUtf8("doubleSpinBoxColorHist"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxColorHist)
        self.checkBoxVisualKeyword = QtGui.QCheckBox(self.centralwidget)
        self.checkBoxVisualKeyword.setEnabled(True)
        self.checkBoxVisualKeyword.setFocusPolicy(QtCore.Qt.TabFocus)
        self.checkBoxVisualKeyword.setChecked(True)
        self.checkBoxVisualKeyword.setObjectName(_fromUtf8("checkBoxVisualKeyword"))
        self.horizontalLayout_2.addWidget(self.checkBoxVisualKeyword)
        self.doubleSpinBoxVisualKeyword = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxVisualKeyword.setDecimals(1)
        self.doubleSpinBoxVisualKeyword.setSingleStep(0.5)
        self.doubleSpinBoxVisualKeyword.setProperty("value", 1.0)
        self.doubleSpinBoxVisualKeyword.setObjectName(_fromUtf8("doubleSpinBoxVisualKeyword"))
        self.horizontalLayout_2.addWidget(self.doubleSpinBoxVisualKeyword)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_5 = QtGui.QLabel(self.centralwidget)
        self.label_5.setText(_fromUtf8(""))
        self.label_5.setPixmap(QtGui.QPixmap(_fromUtf8("assets/ic_letter.png")))
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.horizontalLayout_3.addWidget(self.label_5)
        self.label_2 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Helvetica"))
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.tags_search = QtGui.QLineEdit(self.centralwidget)
        self.tags_search.setObjectName(_fromUtf8("tags_search"))
        self.horizontalLayout_3.addWidget(self.tags_search)
        self.doubleSpinBoxText = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBoxText.setDecimals(1)
        self.doubleSpinBoxText.setSingleStep(0.5)
        self.doubleSpinBoxText.setProperty("value", 1.0)
        self.doubleSpinBoxText.setObjectName(_fromUtf8("doubleSpinBoxText"))
        self.horizontalLayout_3.addWidget(self.doubleSpinBoxText)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.listWidgetResults = QtGui.QListWidget(self.centralwidget)
        self.listWidgetResults.setEnabled(True)
        self.listWidgetResults.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.listWidgetResults.setFocusPolicy(QtCore.Qt.NoFocus)
        self.listWidgetResults.setDragDropMode(QtGui.QAbstractItemView.DragDrop)
        self.listWidgetResults.setAlternatingRowColors(False)
        self.listWidgetResults.setSelectionMode(QtGui.QAbstractItemView.NoSelection)
        self.listWidgetResults.setIconSize(QtCore.QSize(140, 140))
        self.listWidgetResults.setResizeMode(QtGui.QListView.Adjust)
        self.listWidgetResults.setViewMode(QtGui.QListView.IconMode)
        self.listWidgetResults.setUniformItemSizes(True)
        self.listWidgetResults.setObjectName(_fromUtf8("listWidgetResults"))
        self.verticalLayout.addWidget(self.listWidgetResults)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.btn_reset = QtGui.QPushButton(self.centralwidget)
        self.btn_reset.setMinimumSize(QtCore.QSize(66, 32))
        self.btn_reset.setObjectName(_fromUtf8("btn_reset"))
        self.horizontalLayout_5.addWidget(self.btn_reset)
        self.btn_quit = QtGui.QPushButton(self.centralwidget)
        self.btn_quit.setMinimumSize(QtCore.QSize(66, 32))
        self.btn_quit.setObjectName(_fromUtf8("btn_quit"))
        self.horizontalLayout_5.addWidget(self.btn_quit)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.listWidgetResults.setCurrentRow(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.btn_reset, self.btn_quit)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "CS2108 Audio Search", None))
        self.btn_picker.setText(_translate("MainWindow", "Choose Video", None))
        self.btn_search.setText(_translate("MainWindow", "Estimate Venue", None))
        self.label_3.setText(_translate("MainWindow", "Acoustic:", None))
        self.checkBox_energy.setText(_translate("MainWindow", "Energy", None))
        self.checkBox_zerocrossing.setText(_translate("MainWindow", "Zero-Crossing", None))
        self.checkBox_spect.setText(_translate("MainWindow", "Mag Spectrum", None))
        self.checkBox_mfcc.setText(_translate("MainWindow", "MFCC", None))
        self.label_visual.setText(_translate("MainWindow", "Visual:", None))
        self.checkBoxColorHist.setText(_translate("MainWindow", "Color Histogram", None))
        self.checkBoxVisualKeyword.setText(_translate("MainWindow", "Visual Keyword", None))
        self.label_2.setText(_translate("MainWindow", "Text:", None))
        self.btn_reset.setText(_translate("MainWindow", "Clear", None))
        self.btn_quit.setText(_translate("MainWindow", "Quit", None))
