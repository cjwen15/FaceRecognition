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
import time
import cv2
import face


# 用 OpenCV 标注 bounding_box (用一个最小的矩形，把找到的形状包起来) 主要用到下面两个工具——cv2.rectangle() 和 cv2.putText()

# cv2.rectangle() # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
# cv2.rectangle(img, (x,y), (x+w,y+h), (B,G,R), Thickness)

# cv2.putText() # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
# cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)

def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            # astype 修改数据类型  区别：dtype 数据元素的类型
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main():
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100/Streaming/Channels/1") # rtsp实时流传输协议，是TCP/IP协议体系中的一个应用层协议
    # video_capture = cv2.VideoCapture("rtsp://admin:12345678hu@192.168.0.100:80/h264/ch1/main/av_stream")
    face_recognition = face.Recognition()

    # VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
    video_capture = cv2.VideoCapture(0)
    start_time = time.time()
    while True:

        # 逐帧捕获
        ret, frame = video_capture.read()

        # 颜色空间转换
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.identify(frame)

        # 查看当前fps
        end_time = time.time()
        frame_rate = float('%.2f' % (1 / (end_time - start_time)))
        start_time = time.time()
        add_overlays(frame, faces, frame_rate)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 完成后释放捕获
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
