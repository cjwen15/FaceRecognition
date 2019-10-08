import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import cv2
import tensorflow as tf
import numpy as np
from scipy import misc

from src import facenet, align
from src.common import face_preprocess

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import pickle
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import copy
import argparse
# import facenet
from src import face, align, facenet
import src.align.detect_face
from src import Images_face_recognition

import random
from PIL import Image, ImageDraw, ImageFont

from os.path import join as pjoin
import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals import joblib


class picture_Window(QDialog):

    def __init__(self):
        super(picture_Window, self).__init__()
        self.set_ui()

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.choose = QtWidgets.QPushButton(u'选择图片')
        self.start = QtWidgets.QPushButton(u'开始识别')
        self.display1 = QtWidgets.QLabel()

        self.choose.setMinimumHeight(40)
        self.start.setMinimumHeight(40)

        self.display1.setFixedSize(600, 400)
        self.display1.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.choose)
        self.__layout_fun_button.addWidget(self.start)
        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.display1)

        self.setLayout(self.__layout_main)
        self.setWindowTitle(u'图片识别')

        # 添加按钮响应
        self.choose.clicked.connect(self.loadFile)
        self.start.clicked.connect(self.button_start_click)

    # 这个函数是用来打开电脑的资源管理器选择照片
    def loadFile(self):
        global fname
        # QFileDialog就是系统对话框的那个类第一个参数是上下文，第二个参数是弹框的名字，第三个参数是开始打开的路径，第四个参数是需要的格式
        fname, ftype = QFileDialog.getOpenFileName(self, '选择图片', 'c:\\', 'Image files(*.jpg *.gif *.png)')
        self.display1.setPixmap(QtGui.QPixmap(fname))

    def function(self):
        image = cv2.imread(fname)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_recognition = face.Recognition()
        faces = face_recognition.identify(frame)
        self.add_overlays(image, faces)
        newfname = os.path.splitext(os.path.split(fname)[1])[0]
        cv2.imwrite('../images/' + newfname + '-result.jpg', image)
        self.display1.setPixmap(QtGui.QPixmap('../images/' + newfname + '-result.jpg'))

    def add_overlays(self, image, faces):
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(image,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              (0, 255, 0), 2)
                if face.name is not None:
                    if face.name == 'unknown':
                        cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                                    thickness=2, lineType=2)
                    else:
                        cv2.putText(image, face.name, (face_bb[0], face_bb[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                                    thickness=2, lineType=2)

    # “开始识别”按钮功能
    def button_start_click(self):
        self.function()
