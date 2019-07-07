import xml.etree.cElementTree as ET
import numpy as np
import os
from PIL import Image
import scipy.io as sio


OBJECT_NAMES = ["tvmonitor", "train", "sofa", "sheep", "cat", "chair", "bottle", "motorbike", "boat", "bird",
                   "person", "aeroplane", "dog", "pottedplant", "cow", "bus", "diningtable", "horse", "bicycle", "car"]
ANCHORS = [[10, 13], [16, 30], [33, 23], [30,61], [62, 45], [59, 119], [116, 90],
           [156, 198], [373, 326]]

NUMS_ANCHOR = 9
NUMS_CELL = 13
NUMS_CLASS = 20
EPSILON = 1e-8

def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall("object")
    imgname = root.find("filename").text
    gt_bbox = np.zeros([objects.__len__(), 4], dtype=np.int32)
    name_bbox = []
    for i, obj in enumerate(objects):
        objectname = obj.find("name").text
        bbox = np.zeros([4], dtype=np.int32)
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        bbox[0], bbox[1], bbox[2], bbox[3] = xmin, ymin, xmax, ymax
        name_bbox.append(objectname)
        gt_bbox[i, :] = bbox

    return imgname, gt_bbox, name_bbox

def cal_iou(bbox1, bbox2):
    #bbox = [x1, y1, x2, y2]
    x1, y1, x1_, y1_ = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, x2_, y2_ = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    x0 = max(x1, x2)
    y0 = max(y1, y2)
    x0_ = min(x1_, x2_)
    y0_ = min(y1_, y2_)

    if x0 >= x0_ or y0 >= y0_:
        iou = 0
    else:
        inter_area = (x0_ - x0) * (y0_ - y0)
        bbox1_area = (x1_ - x1) * (y1_ - y1)
        bbox2_area = (x2_ - x2) * (y2_ - y2)
        union_area = bbox1_area + bbox2_area - inter_area

        iou = inter_area / union_area
    return iou

def cal_iou_gtbbox_anchors(bbox, anchors):
    #bbox: [x1, y1, x2, y2], anchors: [[x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
    bbox = bbox[np.newaxis, :]
    bboxes = np.tile(bbox, reps=[NUMS_ANCHOR, 1])#[x1, y1, x2, y2] -> [[x1, y1, x2, y2], [x1, y1, x2, y2], ..., [x1, y1, x2, y2]]
    x1, x1_ = bboxes[:, 0], anchors[:, 0]
    x2, x2_ = bboxes[:, 2], anchors[:, 2]
    y1, y1_ = bboxes[:, 1], anchors[:, 1]
    y2, y2_ = bboxes[:, 3], anchors[:, 3]
    x_inter1 = np.maximum(x1, x1_)
    x_inter2 = np.minimum(x2, x2_)
    y_inter1 = np.maximum(y1, y1_)
    y_inter2 = np.minimum(y2, y2_)
    inter_area = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
    union_area = (x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - inter_area
    iou = inter_area / union_area
    return iou

def ToScaleImg(img, tar_h, tar_w, raw_bboxes):
    h, w = img.shape[0], img.shape[1]
    nums_bbox = raw_bboxes.shape[0]
    tar_bboxes = np.zeros_like(raw_bboxes)
    for i in range(nums_bbox):
        bbox = raw_bboxes[i]
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        x0 = tar_w / w * x0
        x1 = tar_w / w * x1
        y0 = tar_h / h * y0
        y1 = tar_h / h * y1
        tar_bboxes[i, 0], tar_bboxes[i, 1] = x0, y0
        tar_bboxes[i, 2], tar_bboxes[i, 3] = x1, y1
    scaled_img = np.array(Image.fromarray(np.uint8(img)).resize([tar_w, tar_h]))#misc.imresize(img, [tar_h, tar_w])
    return scaled_img, tar_bboxes

def test(img, anchors, idx):
    bbox = anchors[idx]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    return img[y1:y2, x1:x2]



def read_batch(img_path, xml_path, batch_size, img_h=416, img_w=416):
    xml_lists = os.listdir(xml_path)
    nums = xml_lists.__len__()
    rand_idx = np.random.randint(0, nums, [batch_size])
    batch_bboxes = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, 4])#bboxes: 13 x 13 x 9 x 4
    batch_classes = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, NUMS_CLASS])#classes: 13 x 13 x 9 x 20
    batch_confidence = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, 1])#confidences: 13 x 13 x 9
    batch_img = np.zeros([batch_size, img_h, img_w, 3])
    cell_h = img_h / NUMS_CELL
    cell_w = img_w / NUMS_CELL
    for j in range(batch_size):
        imgname, gt_bbox, name_bbox = read_xml(xml_path + xml_lists[rand_idx[j]])
        img = np.array(Image.open(img_path + imgname))
        scaled_img, scaled_bbox = ToScaleImg(img, img_h, img_w, gt_bbox)
        batch_img[j, :, :, :] = scaled_img
        for i in range(scaled_bbox.shape[0]):
            c_x = (scaled_bbox[i, 0] + scaled_bbox[i, 2]) / 2
            anchor_x = (c_x // cell_w + 0.5) * cell_w
            c_y = (scaled_bbox[i, 1] + scaled_bbox[i, 3]) / 2
            anchor_y = (c_y // cell_h + 0.5) * cell_h
            anchors_wh = np.array(ANCHORS)
            w, h = anchors_wh[:, 0], anchors_wh[:, 1]
            x1, x2 = anchor_x - w/2, anchor_x + w/2
            y1, y2 = anchor_y - h/2, anchor_y + h/2
            anchors = np.stack((x1, y1, x2, y2), axis=1)
            iou = cal_iou_gtbbox_anchors(scaled_bbox[i, :], anchors)
            argmax_iou = np.argmax(iou)
            confidence = np.max(iou)
            anchor_w, anchor_h = anchors_wh[argmax_iou, 0], anchors_wh[argmax_iou, 1]
            bbox_w, bbox_h = scaled_bbox[i, 2] - scaled_bbox[i, 0], scaled_bbox[i, 3] - scaled_bbox[i, 1]
            col = int(c_x // cell_w)
            row = int(c_y // cell_h)
            t_x = c_x / cell_w - col
            t_y = c_y / cell_h - row
            t_h = np.sqrt(bbox_h / anchor_h)
            t_w = np.sqrt(bbox_w / anchor_w)
            batch_bboxes[j, row, col, argmax_iou, 0], batch_bboxes[j, row, col, argmax_iou, 1] = t_x, t_y
            batch_bboxes[j, row, col, argmax_iou, 2], batch_bboxes[j, row, col, argmax_iou, 3] = t_h, t_w
            batch_confidence[j, row, col, argmax_iou, 0] = confidence
            index = OBJECT_NAMES.index(name_bbox[i])
            batch_classes[j, row, col, argmax_iou, index] = 1
    batch_labels = np.concatenate((batch_bboxes, batch_confidence, batch_classes), axis=-1)
    return batch_img, batch_labels

def read_colab_batch(imgs_mat, batch_size, img_h=416, img_w=416):
    imgs = imgs_mat["imgs"]
    bboxes = imgs_mat["bboxes"]
    class_name = imgs_mat["class"]
    nums = imgs.shape[0]
    rand_idx = np.random.randint(0, nums, [batch_size])
    batch_bboxes = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, 4])#bboxes: 13 x 13 x 9 x 4
    batch_classes = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, NUMS_CLASS])#classes: 13 x 13 x 9 x 20
    batch_confidence = np.zeros([batch_size, NUMS_CELL, NUMS_CELL, NUMS_ANCHOR, 1])#confidences: 13 x 13 x 9
    batch_img = np.zeros([batch_size, img_h, img_w, 3])
    cell_h = img_h / NUMS_CELL
    cell_w = img_w / NUMS_CELL
    for j in range(batch_size):
        scaled_img, scaled_bbox, name_bbox = imgs[rand_idx[j]], bboxes[0, rand_idx[j]], class_name[0, rand_idx[j]]
        scaled_img, scaled_bbox = ToScaleImg(scaled_img, img_h, img_w, scaled_bbox)
        batch_img[j, :, :, :] = scaled_img
        for i in range(scaled_bbox.shape[0]):
            c_x = (scaled_bbox[i, 0] + scaled_bbox[i, 2]) / 2
            anchor_x = (c_x // cell_w + 0.5) * cell_w
            c_y = (scaled_bbox[i, 1] + scaled_bbox[i, 3]) / 2
            anchor_y = (c_y // cell_h + 0.5) * cell_h
            anchors_wh = np.array(ANCHORS)
            w, h = anchors_wh[:, 0], anchors_wh[:, 1]
            x1, x2 = anchor_x - w/2, anchor_x + w/2
            y1, y2 = anchor_y - h/2, anchor_y + h/2
            anchors = np.stack((x1, y1, x2, y2), axis=1)
            iou = cal_iou_gtbbox_anchors(scaled_bbox[i, :], anchors)
            argmax_iou = np.argmax(iou)
            confidence = np.max(iou)
            anchor_w, anchor_h = anchors_wh[argmax_iou, 0], anchors_wh[argmax_iou, 1]
            bbox_w, bbox_h = scaled_bbox[i, 2] - scaled_bbox[i, 0], scaled_bbox[i, 3] - scaled_bbox[i, 1]
            col = int(c_x // cell_w)
            row = int(c_y // cell_h)
            t_x = c_x / cell_w - col
            t_y = c_y / cell_h - row
            t_h = np.sqrt(bbox_h / anchor_h)
            t_w = np.sqrt(bbox_w / anchor_w)
            batch_bboxes[j, row, col, argmax_iou, 0], batch_bboxes[j, row, col, argmax_iou, 1] = t_x, t_y
            batch_bboxes[j, row, col, argmax_iou, 2], batch_bboxes[j, row, col, argmax_iou, 3] = t_h, t_w
            batch_confidence[j, row, col, argmax_iou, 0] = confidence
            index = OBJECT_NAMES.index(name_bbox[i].replace(" ", ""))
            batch_classes[j, row, col, argmax_iou, index] = 1
    batch_labels = np.concatenate((batch_bboxes, batch_confidence, batch_classes), axis=-1)
    return batch_img, batch_labels

def restore_bbox(row, col, w, h, norm_bbox):
    norm_x, norm_y, norm_h, norm_w = norm_bbox[0], norm_bbox[1], norm_bbox[2], norm_bbox[3]
    c_x = (col + norm_x) * 32
    c_y = (row + norm_y) * 32
    w = np.exp(norm_w) * w
    h = np.exp(norm_h) * h
    x1, x2 = c_x - w / 2, c_x + w / 2
    y1, y2 = c_y - h / 2, c_y + h / 2
    return np.array([x1, y1, x2, y2])


def img2mat(imgpath, xmlpath):
    filenames = os.listdir(xmlpath)
    nums = filenames.__len__()
    imgs = np.zeros([nums, 448, 448, 3], dtype=np.uint8)
    xml = []
    class_name = []
    for idx, filename in enumerate(filenames):
        imgname, gt_bbox, name_bbox = read_xml(xmlpath + filename)
        img = np.array(Image.open(imgpath + imgname))
        scaled_img, scaled_bbox = ToScaleImg(img, 448, 448, gt_bbox)
        imgs[idx, :, :, :] = scaled_img
        xml.append(scaled_bbox)
        class_name.append(name_bbox)
        print(idx)
    sio.savemat("pascal.mat", {"imgs": imgs, "bboxes": xml, "class": class_name})

