import tensorflow as tf
from ops import sub_net, pred2bbox
import numpy as np
from PIL import Image
from draw_bbox import draw_bbox
from vgg16 import vgg16

import os

ANCHORS = [[10, 13], [16, 30], [33, 23], [30,61], [62, 45], [59, 119], [116, 90],
           [156, 198], [373, 326]]
OBJECT_NAMES = ["tvmonitor", "train", "sofa", "sheep", "cat", "chair", "bottle", "motorbike", "boat", "bird",
                   "person", "aeroplane", "dog", "pottedplant", "cow", "bus", "diningtable", "horse", "bicycle", "car"]

def recover_bbox(row, col, anchor_wh, bbox):
    p_w, p_h = anchor_wh[0], anchor_wh[1]
    t_x, t_y, t_h, t_w = bbox[0], bbox[1], bbox[2], bbox[3]
    h, w = t_h * p_h, t_w * p_w
    c_x = (col + t_x) * 32.0
    c_y = (row + t_y) * 32.0
    x1, y1 = c_x - w / 2, c_y - h / 2
    x2, y2 = c_x + w / 2, c_y + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def select_bbox_again(best_idx_store, score_store, bboxes):
    max_score = np.max(score_store)
    if max_score < 0.2:
        score_store[score_store < max_score*0.9] = 0
    else:
        score_store[score_store < 0.2] = 0
    n_nms = best_idx_store.shape[-1]
    pair_class_bbox = []
    for i in range(20):
        for j in range(n_nms):
            if score_store[i, best_idx_store[i, j]] > 0:
                pair_class_bbox.append([OBJECT_NAMES[i], bboxes[best_idx_store[i, j]]])
    return pair_class_bbox


class yolo2:
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        features = vgg16(self.inputs)
        self.prediction = sub_net(features)  # 1 x 13 x 13 x 9 x 25
        bboxes_xy = self.prediction[:, :, :, :, :2]
        bboxes_hw = tf.square(self.prediction[:, :, :, :, 2:4])
        bboxes = tf.concat([bboxes_xy, bboxes_hw], axis=-1)
        bboxes = pred2bbox(bboxes)
        self.bboxes = tf.reshape(bboxes, [-1, 4])
        confidences = self.prediction[:, :, :, :, 4:5]
        class_score = self.prediction[:, :, :, :, 5:]
        class_specific_conf = confidences * class_score  # 1 x 13 x 13 x 9 x 20
        split_class = tf.split(class_specific_conf, num_or_size_splits=20, axis=-1)
        best_idx_store = []
        score_store = []
        for score in split_class:
            score = tf.reshape(score, [-1])
            score_store.append(score)
            best_idx = tf.image.non_max_suppression(self.bboxes, score, 10)
            best_idx_store.append(best_idx)
        self.best_idx_store = tf.stack(best_idx_store)
        self.score_store = tf.stack(score_store)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, "./save_para/model.ckpt")

    def detect(self, img_path):
        img = np.array(Image.open(img_path).resize([416, 416]))
        [BEST_IDX, SCORE, BBOX] = self.sess.run([self.best_idx_store, self.score_store, self.bboxes],
                                           feed_dict={self.inputs: img[np.newaxis, :, :, :]})
        PAIR_DATA = select_bbox_again(BEST_IDX, SCORE, BBOX)
        for item in PAIR_DATA:
            object_name = item[0]
            bbox = np.int32(item[1])
            img = draw_bbox(img, bbox, object_name)
        Image.fromarray(np.uint8(img)).show()

    def detect_and_save(self, img_pathes, save_path):
        img_list = os.listdir(img_pathes)
        for img_name in img_list:
            img = np.array(Image.open(img_pathes + img_name).resize([416, 416]))
            [BEST_IDX, SCORE, BBOX] = self.sess.run([self.best_idx_store, self.score_store, self.bboxes],
                                               feed_dict={self.inputs: img[np.newaxis, :, :, :]})
            PAIR_DATA = select_bbox_again(BEST_IDX, SCORE, BBOX)
            for item in PAIR_DATA:
                object_name = item[0]
                bbox = np.int32(item[1])
                img = draw_bbox(img, bbox, object_name)
            Image.fromarray(np.uint8(img)).save(save_path + img_name)

if __name__ == "__main__":
    yolo = yolo2()
    yolo.detect("C:/Users/gmt/Desktop/30.jpg")
    yolo.detect_and_save("E:/数据集/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/", "C:/Users/gmt/Desktop/detect_img/")
    # test("E:/数据集/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/001144.jpg")