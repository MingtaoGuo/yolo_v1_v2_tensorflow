import tensorflow as tf

ANCHORS = [[10, 13], [16, 30], [33, 23], [30,61], [62, 45], [59, 119], [116, 90],
           [156, 198], [373, 326]]


def conv(name, inputs, k_size, strides, nums_out):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def preprocess(img):
    img = tf.stack([img[:, :, :, 0] - 123.68, img[:, :, :, 1] - 116.779,
              img[:, :, :, 2] - 103.939], axis=-1)
    return img * 0.017

def sub_net(inputs):
    inputs = conv("logits", inputs, 1, 1, 9 * 25)
    inputs = tf.reshape(inputs, [-1, 13, 13, 9, 25])
    bboxes_xy = tf.nn.sigmoid(inputs[:, :, :, :, :2])
    bboxes_hw = tf.sqrt(tf.exp(inputs[:, :, :, :, 2:4]))
    confidences = tf.nn.sigmoid(inputs[:, :, :, :, 4:5])
    classes = tf.nn.softmax(inputs[:, :, :, :, 5:])
    return tf.concat([bboxes_xy, bboxes_hw, confidences, classes], axis=-1)

def pred2bbox(pred):
    #pred: 1 x 13 x 13 x 9 x 4
    anchors = tf.constant(ANCHORS, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]#9 x 2 -> 1 x 1 x 1 x 9 x 2
    anchors = tf.tile(anchors, multiples=[1, 13, 13, 1, 1])#1 x 1 x 1 x 9 x 2 -> 1 x 13 x 13 x 9 x 2
    offset_col = tf.range(13, dtype=tf.float32)[tf.newaxis, :]#[[0 1 2 3 4 5 6 7 8 9 10 11 12]]
    offset_col = tf.tile(offset_col, multiples=[13, 1])[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
    offset_row = tf.transpose(offset_col, perm=[0, 2, 1, 3, 4])
    x, y = pred[:, :, :, :, 0:1], pred[:, :, :, :, 1:2]
    h, w = pred[:, :, :, :, 2:3], pred[:, :, :, :, 3:4]
    anchor_h, anchor_w = anchors[:, :, :, :, 1:2], anchors[:, :, :, :, 0:1]
    x = (x + offset_col) * 32.0
    y = (y + offset_row) * 32.0
    h = h * anchor_h
    w = w * anchor_w
    x1, x2 = x - w / 2, x + w / 2
    y1, y2 = y - h / 2, y + h / 2
    return tf.concat([x1, y1, x2, y2], axis=-1)


def cal_iou(prediction, labels):
    anchors_hw = tf.constant(ANCHORS, dtype=tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis, :, :]#1 x 1 x 1 x 9 x 2
    pred_xy = prediction[:, :, :, :, :2]
    pred_hw = prediction[:, :, :, :, 2:4] * anchors_hw
    label_xy = labels[:, :, :, :, :2]
    label_hw = labels[:, :, :, :, 2:4] * anchors_hw
    pred_h, pred_w = pred_hw[:, :, :, :, 0], pred_hw[:, :, :, :, 1]
    label_h, label_w = label_hw[:, :, :, :, 0], label_hw[:, :, :, :, 1]
    pred_x, pred_y = pred_xy[:, :, :, :, 0], pred_xy[:, :, :, :, 1]
    label_x, label_y = label_xy[:, :, :, :, 0], label_xy[:, :, :, :, 1]
    x1, x1_ = pred_x - pred_w/2, label_x - label_w/2
    x2, x2_ = pred_x + pred_w/2, label_x + label_w/2
    y1, y1_ = pred_y - pred_h/2, label_y - label_h/2
    y2, y2_ = pred_y + pred_h/2, label_y + label_h/2
    inter_x1 = tf.maximum(x1, x1_)
    inter_x2 = tf.minimum(x2, x2_)
    inter_y1 = tf.maximum(y1, y1_)
    inter_y2 = tf.minimum(y2, y2_)
    inter_area = tf.maximum(0., inter_x2 - inter_x1) * tf.maximum(0., inter_y2 - inter_y1)
    union_area = pred_h * pred_w + label_h * label_w - inter_area
    iou = inter_area / (union_area + 1e-10)
    return iou


def yolo_loss(prediction, labels):
    pred_bboxes = prediction[:, :, :, :, :4]
    pred_confidences = prediction[:, :, :, :, 4:5]
    pred_classes = prediction[:, :, :, :, 5:]
    labels_bboxes = labels[:, :, :, :, :4]
    labels_confidences = labels[:, :, :, :, 4:5]
    labels_classes = labels[:, :, :, :, 5:]
    mask_obj = tf.cast(tf.greater(labels_confidences, 0.), dtype=tf.float32)
    loss_bbox = tf.reduce_mean(tf.reduce_sum(tf.square(pred_bboxes - labels_bboxes) * mask_obj, axis=[1, 2, 3, 4]))
    loss_conf_obj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidences - 1) * mask_obj , axis=[1, 2, 3, 4]))
    loss_conf_noobj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidences) * (1 - mask_obj), axis=[1, 2, 3, 4]))
    loss_classes = tf.reduce_mean(tf.reduce_sum(tf.square(pred_classes - labels_classes) * mask_obj, axis=[1, 2, 3, 4]))
    return loss_bbox * 5.0 + loss_conf_obj + loss_conf_noobj * 0.5 + loss_classes, loss_bbox, loss_conf_obj, loss_conf_noobj, loss_classes


