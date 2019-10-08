"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
#
# Copyright (c) 2016 David Sandberg
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

from __future__ import absolute_import  # 绝对引入，从指明顶层中导入模块
from __future__ import division  # 精确除法 如3/4=0.75
from __future__ import print_function  # 即使在python2.X，使用print就得像python3.X那样加括号使用
# python2.X中print不需要括号，而在python3.X中则需要

import pickle
from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import facenet
import align.detect_face

from src import align, facenet

# os.path.dirname()去掉文件名，返回目录
# os.path.join()用于路径拼接文件路径
# sys.path返回的是一个列表。对于模块和自己写的脚本不在同一个目录下，在脚本开头加sys.path.append(‘xxx’)
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

import face_preprocess

gpu_memory_fraction = 0.3


# Tensor(张量)意味着N维数组，Flow(流)意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。
# TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。

def main():
    model = "20190128-123456/3001w-train.pb"  # 导入模型
    traindata_path = "../data/gump"  # 训练样本的路径
    feature_files = []
    face_label = []
    face_detection = Detection()

    # 表示将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default():

        # 会话的作用是处理内存分配和优化，使我们能够实际执行由图形指定的计算
        with tf.Session() as sess:

            # 加载模型
            facenet.load_model(model)

            # 获取输入和输出的tensors(张量)
            # get_tensor_by_name 根据名字获取某个tensor

            # 输入图像占位符
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")

            # 卷积网络最后输出的"特征"
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # 训练
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            for images in os.listdir(traindata_path):
                print(images)
                filename = os.path.splitext(os.path.split(images)[1])[0]
                image_path = traindata_path + "/" + images
                images = face_detection.find_faces(image_path)
                if images is not None:
                    face_label.append(filename)

                    # Run forward pass to calculate embeddings
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb = sess.run(embeddings, feed_dict=feed_dict)
                    print(emb)
                    feature_files.append(emb)
                else:
                    print('no find face')

            # .pkl 在python中用于保存文件
            write_file = open('20190128-123456/knn_classifier.pkl', 'wb')

            # pickle.dump() 序列化对象，并将结果数据流写入到文件对象中
            pickle.dump(feature_files, write_file, -1)
            pickle.dump(face_label, write_file, -1)
            write_file.close()
            num_count = 0
            for i in range(len(feature_files)):
                for j in range(len(feature_files)):
                    num = np.dot(feature_files[i], feature_files[j].T)
                    sim = 0.5 + 0.5 * num  # 归一化，，余弦距离
                    if sim > 0.82 and sim < 0.99:
                        print(face_label[i], ' ', face_label[j], ' ', sim)
                        num_count = num_count + 1
            print(num_count / 2)
            print('total_num:', len(os.listdir(traindata_path)))
            print('align_num:', len(face_label))
            print('End')


class Detection:
    minsize = 40  # 人脸的最小值
    threshold = [0.8, 0.9, 0.9]  # three steps's threshold 阈值
    factor = 0.709  # scale factor 比例因子

    # 创建P-Net,R-Net,O-Net网络，并加载参数
    def __init__(self, face_crop_size=112, face_crop_margin=0):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
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

                # .T 矩阵的转置
                _landmark = points[:, bindex].reshape((2, 5)).T
                warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
            else:
                for i in range(nrof_faces):
                    _bbox = bounding_boxes[i, 0:4]
                    if _bbox[2] * _bbox[3] > max_Aera:
                        max_Aera = _bbox[2] * _bbox[3]
                        _landmark = points[:, i].reshape((2, 5)).T
                        warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size='112,112')

                # 归一化处理
                prewhitened = facenet.prewhiten(warped)
                img_list.append(prewhitened)
        else:
            return None
        images = np.stack(img_list)
        return images


if __name__ == '__main__':
    main()
