import tensorflow as tf


def yolo_loss(prediction, labels, H=448, W=448, S=7):
    #prediction: batchsize x 7 x 7 x 30, bboxes: batchsize x 7 x 7 x (0:8), confidences: batchsize x 7 x 7 x (8:10), class: batchsize x 7 x 7 x (10:30)
    #labels: batchsize x 7 x 7 x 25, response_mask: batchsize x 7 x 7 x (0:1), bbox: batchsize x 7 x 7 x (1:5), class: batchsize: batchsize x 7 x 7 x (5:25)
    pred_bboxes = prediction[:, :, :, 0:8]
    pred_bboxes = tf.reshape(pred_bboxes, [-1, 7, 7, 2, 4])
    pred_confidence = prediction[:, :, :, 8:10]
    pred_class = prediction[:, :, :, 10:]
    #groundtruth
    label_mask = labels[:, :, :, 0:1]
    label_bboxes = labels[:, :, :, 1:5]
    label_bboxes = tf.tile(label_bboxes, multiples=[1, 1, 1, 2])
    label_bboxes = tf.reshape(label_bboxes, [-1, 7, 7, 2, 4])
    label_class = labels[:, :, :, 5:]
    #prediction normalized offset -> [c_x, c_y, h, w]
    cell_h = H / S
    cell_w = W / S
    temp = tf.constant([[0., 1., 2., 3., 4., 5., 6.]])
    temp = tf.tile(temp, multiples=[7, 1])
    col = temp[tf.newaxis, :, :, tf.newaxis, tf.newaxis]#dimension 7 x 7 -> 1 x 7 x 7 x 1 x 1
    row = tf.transpose(col, perm=[0, 2, 1, 3, 4])
    pred_bboxes_original = tf.concat([(pred_bboxes[:, :, :, :, 0:1] + col) * cell_w,
                                      (pred_bboxes[:, :, :, :, 1:2] + row) * cell_h,
                                       tf.square(pred_bboxes[:, :, :, :, 2:3]) * H,
                                       tf.square(pred_bboxes[:, :, :, :, 3:4]) * W], axis=-1)
    label_bboxes_original = tf.concat([(label_bboxes[:, :, :, :, 0:1] + col) * cell_w,
                                      (label_bboxes[:, :, :, :, 1:2] + row) * cell_h,
                                      tf.square(label_bboxes[:, :, :, :, 2:3]) * H,
                                      tf.square(label_bboxes[:, :, :, :, 3:4]) * W], axis=-1)
    iou = cal_iou(pred_bboxes_original, label_bboxes_original)#input: batchsize x 7 x 7 x 2 x 4, output: batchsize x 7 x 7 x 2
    max_iou = tf.reduce_max(iou, axis=-1, keep_dims=True)#output: batchsize x 7 x 7 x 1
    mask_obj = tf.cast(tf.greater_equal(iou, max_iou), dtype=tf.float32) * label_mask
    loss_bboxes = tf.reduce_mean(tf.reduce_sum(tf.square(pred_bboxes - label_bboxes) * mask_obj[:, :, :, :, tf.newaxis], axis=[1, 2, 3, 4]))
    loss_confidence_obj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidence - 1) * mask_obj, axis=[1, 2, 3]))
    loss_confidence_noobj = tf.reduce_mean(tf.reduce_sum(tf.square(pred_confidence) * (1 - mask_obj), axis=[1, 2, 3]))
    loss_class = tf.reduce_mean(tf.reduce_sum(tf.square(pred_class - label_class) * label_mask, axis=[1, 2, 3]))
    loss = 5 * loss_bboxes + loss_confidence_obj + 0.5 * loss_confidence_noobj + loss_class
    return loss, loss_bboxes, loss_confidence_obj, loss_confidence_noobj, loss_class

def cal_iou(bboxes1, bboxes2):
    #bboxes: [batchsize, 7, 7, 2, 4] with [center_x, center_y, h, w]
    cx, cx_ = bboxes1[:, :, :, :, 0], bboxes2[:, :, :, :, 0]
    cy, cy_ = bboxes1[:, :, :, :, 1], bboxes2[:, :, :, :, 1]
    h, h_ = bboxes1[:, :, :, :, 2], bboxes2[:, :, :, :, 2]
    w, w_ = bboxes1[:, :, :, :, 3], bboxes2[:, :, :, :, 3]
    x1, x1_ = cx - w / 2, cx_ - w_ / 2
    x2, x2_ = cx + w / 2, cx_ + w_ / 2
    y1, y1_ = cy - h / 2, cy_ - h_ / 2
    y2, y2_ = cy + h / 2, cy_ + h_ / 2
    x_inter1 = tf.maximum(x1, x1_)
    x_inter2 = tf.minimum(x2, x2_)
    y_inter1 = tf.maximum(y1, y1_)
    y_inter2 = tf.minimum(y2, y2_)
    h_inter = tf.maximum(0., y_inter2 - y_inter1)
    w_inter = tf.maximum(0., x_inter2 - x_inter1)
    area_inter = h_inter * w_inter
    area_union = h * w + h_ * w_ - area_inter
    iou = area_inter / area_union
    return iou

if __name__ == "__main__":
    prediction = tf.placeholder(tf.float32, [None, 7, 7, 30])
    labels = tf.placeholder(tf.float32, [None, 7, 7, 25])
    yolo_loss(prediction, labels)