#!/usr/bin/env python3
# Author: huangh12 <he.huang1@outlook.com>

import cv2

from PyQt5.QtCore import QTimer, QRegExp, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon, QIntValidator, QRegExpValidator, QTextCursor
from PyQt5.QtWidgets import QDialog, QApplication, QWidget, QMessageBox
from PyQt5.uic import loadUi

import queue
import threading
import sqlite3
import os
import sys
import numpy as np

from datetime import datetime
import tensorflow as tf
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
from process import alignImages


class CreateDatabase(QWidget):
    receiveLogSignal = pyqtSignal(str)

    def __init__(self,
                 det_model='./models/mtcnn', 
                 recog_model='./models/arcface',
                 database='./database/feat.libsvm',):

        super(CreateDatabase, self).__init__()
        loadUi('./ui/CreateDatabase.ui', self)
        self.setFixedSize(1011, 601)

        # OpenCV
        self.cap = cv2.VideoCapture()

        # start webcam
        self.startWebcamButton.toggled.connect(self.startWebcam)
        self.startWebcamButton.setCheckable(True)

        # timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)

        # load the model
        config = tf.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4)
        config.gpu_options.allow_growth = True

        det_graph = tf.Graph()
        recog_graph = tf.Graph()

        self.det_sess = tf.Session(graph=det_graph, config=config)
        self.recog_sess = tf.Session(graph=recog_graph, config=config)

        # load det model
        with self.det_sess.as_default():
            with det_graph.as_default():
                meta_graph_def = tf.saved_model.loader.load(
                        self.det_sess,
                        [tf.saved_model.tag_constants.SERVING],
                        det_model)
                signature = meta_graph_def.signature_def

            inputs_name = signature['predict'].inputs['inputs'].name
            boxes_name = signature['predict'].outputs['boxes'].name
            lmks_name = signature['predict'].outputs['landmarks'].name
            scores_name = signature['predict'].outputs['scores'].name

            self.det_inputs = self.det_sess.graph.get_tensor_by_name(inputs_name)
            self.det_boxes = self.det_sess.graph.get_tensor_by_name(boxes_name)
            self.det_lmks = self.det_sess.graph.get_tensor_by_name(lmks_name)
            self.det_scores = self.det_sess.graph.get_tensor_by_name(scores_name)

        # load recog model
        with self.recog_sess.as_default():
            with recog_graph.as_default():
                meta_graph_def = tf.saved_model.loader.load(
                        self.recog_sess,
                        [tf.saved_model.tag_constants.SERVING],
                        recog_model)
                signature = meta_graph_def.signature_def
                    
            inputs_name = signature['predict'].inputs['inputs'].name
            outputs_name = signature['predict'].outputs['outputs'].name

            self.recog_inputs = self.recog_sess.graph.get_tensor_by_name(inputs_name)
            self.recog_outputs = self.recog_sess.graph.get_tensor_by_name(outputs_name)

        self.isFaceDetectEnabled = False
        self.enableFaceDetectButton.toggled.connect(self.enableFaceDetect)
        self.enableFaceDetectButton.setCheckable(True)

        # capture frame
        self.isCaptureFrameEnabled = False
        self.captureFrameButton.clicked.connect(self.enableCaptureFrame)
        self.releaseFrameButton.clicked.connect(self.disableCaptureFrame)
        self.captureFrameButton.setEnabled(True)
        self.releaseFrameButton.setEnabled(False)

        # data entry
        self.BoxIDLineEdit.textChanged.connect(self.setBoxID)
        self.PersonNameLineEdit.textChanged.connect(self.setPersonName)
        self.BoxIDLineEdit.setEnabled(False)
        self.PersonNameLineEdit.setEnabled(False)
        self.BoxIDLineEdit.setValidator(QIntValidator(0, 65535))

        # import
        self.ImportToDatabaseButton.clicked.connect(self.importToDatabase)
        self.ImportToDatabaseButton.setEnabled(False)
        self.database = database


    def importToDatabase(self):
        self.ImportToDatabaseButton.setEnabled(False)
        try:
            if int(self.BoxID) < len(self.pred_boxes):
                idx = int(self.BoxID)
                box = self.pred_boxes[idx]
                lmk = self.pred_lmks[idx]
                aligned_img = alignImages(self.current_frame, box, lmk)
                aligned_img = aligned_img.reshape((1, 112, 112, 3))
                res = self.recog_sess.run(self.recog_outputs, feed_dict={self.recog_inputs: aligned_img})
                res = res[0]

                write_str_i = [self.PersonName]
                for j in range(1, 513):
                    write_str_i.append('%d:%f' %(j, res[j-1]))

                with open(self.database, 'a', encoding="utf-8") as f1:
                    f1.write(' '.join(write_str_i)+'\n')

                self.isCaptureFrameEnabled = False
                self.captureFrameButton.setEnabled(True)
                self.releaseFrameButton.setEnabled(False)

                self.BoxIDLineEdit.setEnabled(False)
                self.PersonNameLineEdit.setEnabled(False)
            
            else:
                text = 'Fail to import'
                informativeText = '<b>Please check the validity of Box id and Person name.</b>'
                CreateDatabase.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
                self.ImportToDatabaseButton.setEnabled(True)

        except:
            text = 'Fail to import'
            informativeText = '<b>Please check the validity of Box id and Person name.</b>'
            CreateDatabase.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
            self.ImportToDatabaseButton.setEnabled(True)


    def setBoxID(self):
        self.BoxID = self.BoxIDLineEdit.text()


    def setPersonName(self):
        self.PersonName = self.PersonNameLineEdit.text()


    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False


    def startWebcam(self, status):
        if status:
            camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()

            if not ret:
                self.cap.release()
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
                self.startWebcamButton.setChecked(False)
            else:
                self.startWebcamButton.setText('Close the camera')
                self.enableFaceDetectButton.setEnabled(True)
                self.timer.start(5)
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
        else:
            if self.cap.isOpened():
                if self.timer.isActive():
                    self.timer.stop()
                self.cap.release()
                self.faceDetectCaptureLabel.clear()
                self.faceDetectCaptureLabel.setText('<font color=red>The camera is closed.</font>')
                self.startWebcamButton.setText('Open the camera')
                self.enableFaceDetectButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    def enableFaceDetect(self, status):
        if self.cap.isOpened():
            if status:
                self.enableFaceDetectButton.setText('Disable face detection')
                self.isFaceDetectEnabled = True
            else:
                self.enableFaceDetectButton.setText('Enable face detection')
                self.isFaceDetectEnabled = False

    # capture frame
    def enableCaptureFrame(self):
        if not self.isFaceDetectEnabled:
            text = 'Please enable face detection before capturing current frame.'
            informativeText = '<b></b>'
            CreateDatabase.callDialog(QMessageBox.Information, text, informativeText, QMessageBox.Ok)
            return

        self.isCaptureFrameEnabled = True
        self.captureFrameButton.setEnabled(False)
        self.releaseFrameButton.setEnabled(True)

        self.BoxIDLineEdit.setEnabled(True)
        self.PersonNameLineEdit.setEnabled(True)
        self.ImportToDatabaseButton.setEnabled(True)     

    # release frame
    def disableCaptureFrame(self):
        self.isCaptureFrameEnabled = False
        self.captureFrameButton.setEnabled(True)
        self.releaseFrameButton.setEnabled(False)

        self.BoxIDLineEdit.setEnabled(False)
        self.PersonNameLineEdit.setEnabled(False)
        self.ImportToDatabaseButton.setEnabled(False)


    # timer
    def updateFrame(self):
        if self.isCaptureFrameEnabled:
            ret, frame = self.current_ret, self.current_frame
        else:
            ret, frame = self.cap.read()

            if ret:
                self.displayImage(frame)
                if self.isFaceDetectEnabled:
                    detected_frame = self.detectFace(frame.copy())
                    self.displayImage(detected_frame)
                else:
                    self.displayImage(frame)

        self.current_frame = frame
        self.current_ret = ret


    def detectFace(self, frame):
        self.pred_boxes = np.zeros((0, 4))
        self.pred_lmks = np.zeros((0, 5, 2))

        pred_boxes, pred_lmks = self.det_sess.run([self.det_boxes, self.det_lmks], {self.det_inputs: frame[None,:,:]})
        if pred_boxes.shape[0] == 0:
            return frame

        # y1,x1,y2,x2 -> x1,y1,x2,y2
        pred_boxes[:, [0,1]], pred_boxes[:, [2,3]] = \
                pred_boxes[:, [1,0]], pred_boxes[:, [3,2]]
        # y1,...,y5,x1,...,x5 -> x1,y1,x2,y2...x5,y5
        pred_lmks = pred_lmks.reshape((-1, 2, 5)).transpose((0, 2, 1))
        pred_lmks[..., [0]], pred_lmks[..., [1]] = \
            pred_lmks[..., [1]], pred_lmks[..., [0]]

        self.pred_boxes = pred_boxes
        self.pred_lmks = pred_lmks

        idx = 0
        for box, lmk in zip(pred_boxes, pred_lmks):
            box = box.astype('int32')
            frame = cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
            img = cv2.putText(frame, 'Box id: %d' %idx, (box[0], box[1]), 0, 1, (0,255,255), 1)
            idx += 1
            
        return frame


    def displayImage(self, img):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.faceDetectCaptureLabel.setPixmap(QPixmap.fromImage(outImage))
        self.faceDetectCaptureLabel.setScaledContents(True)


    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowTitle('py_face_recognition DataRecord')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CreateDatabase()
    window.show()
    sys.exit(app.exec())
