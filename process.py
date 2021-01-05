#!/usr/bin/env python3
# Author: huangh12 <he.huang1@outlook.com>

import tensorflow as tf
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
import numpy as np
from skimage import transform as trans
import cv2
import os


def alignImages(img, box, landmark, str_image_size='112,112'):

    M = None
    image_size = []
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert image_size[0]==112 or image_size[1]==96
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]

    if M is None:
        if bbox is None:#use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = 44
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))

    else: #do align using landmark
        assert len(image_size)==2
        ret = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue = 0.0)

    return ret


class Processor(object):
    def __init__(self, 
                 det_model='./models/mtcnn', 
                 recog_model='./models/arcface',
                 database='./database/feat.libsvm',
                 det_thresh=0.9,
                 threshold=0.6):

        self.det_model = det_model
        self.recog_model = recog_model
        self.det_thresh = det_thresh
        self.threshold = threshold

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

        # load the database
        self.IDs = []
        self.feats = []
        try:
            with open(database, 'r', encoding="utf-8") as f:
                line = f.readline()
                while line:
                    id, feat = line.split(' ', 1)
                    feat = [float(_.split(':')[-1]) for _ in feat.split(' ')]
                    self.IDs.append(id)
                    self.feats.append(feat)
                    line = f.readline()

        except FileNotFoundError:
            print('Database file %s not found. Please create the database first!' %database)
            exit()

        self.feats = np.array(self.feats, dtype=np.float32)


    def RecogAndDraw(self, img):  

        # feed BGR for face det
        pred_boxes, pred_lmks, pred_scores = self.det_sess.run([self.det_boxes, self.det_lmks, self.det_scores],
                                                               {self.det_inputs: img[None,:,:]})

        keep = np.where(pred_scores > self.det_thresh)[0]
        pred_boxes = pred_boxes[keep]
        pred_lmks = pred_lmks[keep]

        if pred_boxes.shape[0] == 0:
            return img

        # y1,x1,y2,x2 -> x1,y1,x2,y2
        pred_boxes[:, [0,1]], pred_boxes[:, [2,3]] = \
                pred_boxes[:, [1,0]], pred_boxes[:, [3,2]]

        # y1,...,y5,x1,...,x5 -> x1,y1,x2,y2...x5,y5
        pred_lmks = pred_lmks.reshape((-1, 2, 5)).transpose((0, 2, 1))
        pred_lmks[..., [0]], pred_lmks[..., [1]] = \
            pred_lmks[..., [1]], pred_lmks[..., [0]]


        # face align & face recog
        names = []
        for box, lmk in zip(pred_boxes, pred_lmks):
            aligned_img = alignImages(img, box, lmk)

            # resize to target size
            # if not aligned_img.size == (112, 112):
            #     aligned_img = cv2.resize(alignImages, (112, 112), interpolation=cv2.INTER_LINEAR)
            aligned_img = aligned_img.reshape((1, 112, 112, 3))

            res = self.recog_sess.run(self.recog_outputs, feed_dict={self.recog_inputs: aligned_img})
            res = res[0]

            if self.IDs != []:
                similarity = np.matmul(self.feats, res)
                idx = np.argmax(similarity)
                if similarity[idx] > self.threshold:
                    names.append(self.IDs[idx])
                else:
                    names.append('unknown')
            else:
                names.append('unknown')

        # draw
        for name, box, pts in zip(names, pred_boxes, pred_lmks):
            box = box.astype('int32')
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
            img = cv2.putText(img, name, (box[0], box[1]), 0, 1, (0,255,255), 1)
            pts = pts.astype('int32')
            for i in range(5):
                img = cv2.circle(img, (pts[i,0], pts[i,1]), 1, (0, 255, 0), 2)

        return img
