# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import os
# import face
import sys
import facegui

from src import facenet, align, face
from src.common import face_preprocess
# import facenet
import align.detect_face
import pickle  # pickle提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上
# python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化
from scipy import misc
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.childWindow import picture_Window

sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
add_name = ''
# import face_preprocess
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QDialog


class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # self.face_recognition = face.Recognition()
        self.face_detection = Detection()
        self.face_detection_capture = face.Detection()
        self.timer_camera = QtCore.QTimer()
        self.timer_camera_capture = QtCore.QTimer()
        self.cap = cv2.VideoCapture()  # VideoCapture()里面参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0

    # 设置界面
    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.opencamera = QtWidgets.QPushButton(u'人脸识别')
        self.addface = QtWidgets.QPushButton(u'建库')
        self.captureface = QtWidgets.QPushButton(u'采集人脸')
        self.saveface = QtWidgets.QPushButton(u'保存人脸')
        self.introduce = QtWidgets.QPushButton(u'使用说明')
        self.picture = QtWidgets.QPushButton(u'图片识别')

        # 设置按钮大小
        self.opencamera.setMinimumHeight(40)
        self.addface.setMinimumHeight(40)
        self.captureface.setMinimumHeight(40)
        self.saveface.setMinimumHeight(40)
        self.introduce.setMinimumHeight(40)
        self.picture.setMinimumHeight(40)


        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.textChanged.connect(self.text_changed)# 实时打印文本框的内容
        self.lineEdit.setMinimumHeight(40)
        self.lineEdit.move(13, 240)

        # 信息显示
        self.showcamera = QtWidgets.QLabel()
        self.lineEdit.setFixedSize(72, 20)
        self.showcamera.setFixedSize(641, 481)
        self.showcamera.setAutoFillBackground(False)

        # 将按钮、显示框置放在对应布局框
        self.__layout_fun_button.addWidget(self.opencamera)
        self.__layout_fun_button.addWidget(self.addface)
        self.__layout_fun_button.addWidget(self.captureface)
        self.__layout_fun_button.addWidget(self.saveface)
        self.__layout_fun_button.addWidget(self.introduce)
        self.__layout_fun_button.addWidget(self.picture)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.showcamera)
        self.setLayout(self.__layout_main)
        self.setWindowTitle(u'人脸识别系统-designed by 温超杰')

    # 添加按钮响应
    def slot_init(self):
        self.opencamera.clicked.connect(self.button_open_camera_click)
        self.addface.clicked.connect(self.button_add_face_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera_capture.timeout.connect(self.capture_camera)
        self.captureface.clicked.connect(self.button_capture_face_click)
        self.saveface.clicked.connect(self.save_face_click)
        self.introduce.clicked.connect(self.introduce_detail)
        self.picture.clicked.connect(self.showPicWin)

    # 实时打印文本框的内容
    def text_changed(self):
        global add_name
        add_name = self.lineEdit.text()
        print(u'文本框此刻输入的内容是：%s' % add_name)

    # 人脸识别
    def button_open_camera_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        self.face_recognition = face.Recognition()
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)

                self.opencamera.setText(u'关闭识别')
        else:
            self.timer_camera.stop()
            self.cap.release()
            self.showcamera.clear()
            self.opencamera.setText(u'人脸识别')

    # 显示即时图像（人脸识别）
    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))  # 图像尺寸大小
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 然后改变图像色域
        faces = self.face_recognition.identify(show)
        if faces is not None:
            if faces is not None:
                img_PIL = Image.fromarray(show)  # PIL.Image 数据是 uinit8 型的，范围是0-255，所以要进行转换
                font = ImageFont.truetype('simsun.ttc', 40)  # （新宋体）从指定的文件加载了一个字体对象，并且为指定大小的字体创建了字体对象
                # 字体颜色
                fillColor1 = (0, 255, 0)  # “人名”为绿色
                fillColor2 = (255, 0, 0)  # “陌生人”为红色
                draw = ImageDraw.Draw(img_PIL)
                for face in faces:
                    face_bb = face.bounding_box.astype(int)  # 转换数组的数据类型
                    draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
                    draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
                    draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
                    draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
                    if face.name is not None:
                        if face.name == 'unknown':
                            draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                        else:
                            draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
            show = np.asarray(img_PIL)  # 将结构数据转化为ndarray
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 添加人脸（建库）
    def button_add_face_click(self):
        self.timer_camera_capture.stop()
        self.cap.release()
        self.showcamera.clear()
        model = "20190128-123456/3001w-train.pb"
        traindata_path = "../data/gump"
        feature_files = []
        face_label = []
        with tf.Graph().as_default():  # 创建一个默认会话，当上下文管理器退出时会话没有关闭，还可以通过调用会话进行run()和eval()操作
            with tf.Session() as sess:

                # 加载模型
                facenet.load_model(model)

                # 获取输入和输出的tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # get_default_graph() 获取当前默认计算图
                for images in os.listdir(traindata_path):
                    print(images)
                    filename = os.path.splitext(os.path.split(images)[1])[0]

                    # os.path.split()按照路径将文件名和路径分割开
                    # os.path.splitext()分离文件名与扩展名
                    image_path = traindata_path + "/" + images
                    images = self.face_detection.find_faces(image_path)
                    if images is not None:
                        face_label.append(filename)
                        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                        emb = sess.run(embeddings, feed_dict=feed_dict)
                        print(emb)
                        feature_files.append(emb)
                    else:
                        print('未找到人脸')
                write_file = open('20190128-123456/knn_classifier.pkl', 'wb')  # 建库
                pickle.dump(feature_files, write_file, -1)  # 将对象feature_files保存到文件write_file中去
                pickle.dump(face_label, write_file, -1)
                write_file.close()
        reply = QMessageBox.information(self,  # 使用information信息框
                                        "建库",
                                        "建库完成",
                                        QMessageBox.Yes | QMessageBox.No)

    # 采集人脸
    def button_capture_face_click(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.timer_camera_capture.start(30)

    # 采集人脸实现
    def capture_camera(self):
        flag, self.images = self.cap.read()
        self.images = cv2.cvtColor(self.images, cv2.COLOR_BGR2RGB)
        show_images = self.images
        faces = self.face_detection_capture.find_faces(show_images)
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)

                # cv2.rectangle(原图, 矩形的左上点坐标, 矩形的右下点坐标, 画线对应的RGB颜色, 画的线的宽度) 画出矩形
                # 画人脸矩形框并标示类别
                cv2.rectangle(show_images,
                              (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                              (0, 255, 0), 2)
        show_images = np.asarray(show_images)
        showImage = QtGui.QImage(show_images.data, show_images.shape[1], show_images.shape[0],
                                 QtGui.QImage.Format_RGB888)
        self.showcamera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 保存人脸
    def save_face_click(self):
        global add_name
        imagepath = os.sep.join(['../data/gump/', add_name + '.jpg'])  # 存放待识别的人脸
        print('faceID is:', add_name)
        if add_name == '':
            reply = QMessageBox.information(self,  # 使用information信息框
                                            "人脸ID",
                                            "请在文本框输入人脸的ID",
                                            QMessageBox.Yes | QMessageBox.No)
        else:
            self.images = cv2.cvtColor(self.images, cv2.COLOR_RGB2BGR)
            cv2.imencode(add_name + '.jpg', self.images)[1].tofile(imagepath)
            # cv2.imwrite('../data/gump/' + '张三' + '.jpg', self.images)

    # 主窗口关闭提示
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()

    # 使用说明
    def introduce_detail(self):
        QMessageBox.information(self, "使用说明", "\n1、点击“采集人脸”进行人脸采集\n\n"
                                              "2、在文本框中输入人脸名称\n\n"
                                              "3、点击“保存人脸”\n\n"
                                              "4、添加新的人脸后需要进行建库\n\n"
                                              "4.1、点击“建库”\n\n"
                                              "5、点击“人脸识别”进行识别\n\n"
                                              "5.1、可同时识别多个人脸\n\n"
                                              "6、点击“图片识别”进行静态人脸识别\n\n"
                                              "6.1、可同时识别多个人脸\n\n",
                                QMessageBox.Ok)

    # # 设置背景
    # def background(self):
    #     bkgImg = QtGui.QPalette(self)
    #     bkgImg.setBrush(self.backgroundRole(),QtGui.QBrush(QtGui.QPixmap(r"C:\Users\KlayWen\Desktop\基于Python的人脸识别系统\python-FaceRec2-Enhance-GUI\src\Image.jpg")))
    #     self.setPalette(bkgImg)

    # 图片识别窗口
    def showPicWin(self):
        picWin = picture_Window()
        picWin.show()
        picWin.exec()


def add_overlays(frame, faces):
    if faces is not None:
        img_PIL = Image.fromarray(frame)
        font = ImageFont.truetype('simsun.ttc', 40)  # 加载字体“新宋体”
        # 字体颜色
        fillColor1 = (0, 255, 0)  # “人名”的颜色为绿色
        fillColor2 = (255, 0, 0)  # “陌生人”的颜色为红色
        draw = ImageDraw.Draw(img_PIL)
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            draw.line([face_bb[0], face_bb[1], face_bb[2], face_bb[1]], "green")
            draw.line([face_bb[0], face_bb[1], face_bb[0], face_bb[3]], fill=128)
            draw.line([face_bb[0], face_bb[3], face_bb[2], face_bb[3]], "yellow")
            draw.line([face_bb[2], face_bb[1], face_bb[2], face_bb[3]], "black")
            if face.name is not None:
                if face.name == 'unknown':
                    draw.text((face_bb[0], face_bb[1]), '陌生人', font=font, fill=fillColor2)
                else:
                    draw.text((face_bb[0], face_bb[1]), face.name, font=font, fill=fillColor1)
        frame = np.asarray(img_PIL)
        return frame


# 人脸识别
class Detection:
    minsize = 40  # minimum size of face
    threshold = [0.8, 0.9, 0.9]  # three steps's threshold 阈值
    factor = 0.709  # scale factor  比例因子

    def __init__(self, face_crop_size=112, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            # 默认tensorflow是使用GPU尽可能多的显存,这里分配给tensorflow的GPU显存大小为：GPU实际显存*0.7
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image_paths):
        img = misc.imread(os.path.expanduser(image_paths), mode='RGB')
        _bbox = None
        _landmark = None

        # 检测出人脸框和5个特征点
        bounding_boxes, points = align.detect_face.detect_face(img, self.minsize, self.pnet, self.rnet, self.onet,
                                                               self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]
        img_list = []
        max_Aera = 0
        if nrof_faces > 0:
            if nrof_faces == 1:
                bindex = 0
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T  # 矩阵转置
                warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)  # prewhiten()人脸图片
                img_list.append(prewhitened)
            else:
                for i in range(nrof_faces):
                    _bbox = bounding_boxes[i, 0:4]
                    if _bbox[2] * _bbox[3] > max_Aera:
                        max_Aera = _bbox[2] * _bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
