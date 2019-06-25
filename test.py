from networks import vgg16
from utils import *
from ops import *
from draw_bbox import draw_bbox

#
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BATCH_SIZE = 1
img_path = "./VOCdevkit/VOC2007/JPEGImages/"
xml_path = "./VOCdevkit/VOC2007/Annotations/"
OBJECT_NAMES = ["tvmonitor", "train", "sofa", "sheep", "cat", "chair", "bottle", "motorbike", "boat", "bird",
                   "person", "aeroplane", "dog", "pottedplant", "cow", "bus", "diningtable", "horse", "bicycle", "car"]

def norm2bbox(norm_bbox, H=448, W=448, S=7):
    temp = tf.constant([[0., 1., 2., 3., 4., 5., 6.]])
    col = tf.tile(temp, multiples=[7, 1])[tf.newaxis, :, :, tf.newaxis, tf.newaxis]
    row = tf.transpose(col, perm=[0, 2, 1, 3, 4])
    x, y = norm_bbox[:, :, :, :, 0:1], norm_bbox[:, :, :, :, 1:2]
    h, w = norm_bbox[:, :, :, :, 2:3], norm_bbox[:, :, :, :, 3:4]
    x = (x + col) * (W / S)
    y = (y + row) * (H / S)
    h = tf.square(h) * H
    w = tf.square(w) * W
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    coord = tf.concat([x1, y1, x2, y2], axis=-1)
    return coord

def select_bbox_again(indx_class, scores_class, pred_bboxes):
    #indx_class: the index after non-maximun suppression, [20, 5]
    #scores_class: confidence * class_confidence, [20, 98]
    #pred_bboxes: [98, 4]
    mask = np.zeros_like(scores_class)
    for i in range(20):
        for j in range(5):
            mask[i, indx_class[i, j]] = 1
    scores_class[scores_class < 0.2] = 0
    scores_class *= mask
    max_score = np.max(scores_class, axis=0)
    indx_bboxes = np.arange(0, 98)
    indx_bboxes = indx_bboxes[max_score > 0]
    class_indx = np.argmax(scores_class, axis=0)
    bbox_indx = class_indx[max_score > 0]
    return pred_bboxes[indx_bboxes], bbox_indx

def test(img):
    inputs = tf.placeholder(tf.float32, [1, 448, 448, 3])
    prediction = vgg16(inputs)
    pred_bboxes = prediction[:, :, :, :8]#[1, 7, 7, 8]
    pred_bboxes = tf.reshape(pred_bboxes, [1, 7, 7, 2, 4])#[1, 7, 7, 2, 4]
    pred_confidence = prediction[:, :, :, 8:10]#[1, 7, 7, 2]
    pred_class = prediction[:, :, :, 10:]#[1, 7, 7, 20]
    pred_bboxes = norm2bbox(pred_bboxes)#[1, 7, 7, 2, 4]
    class_confidences = tf.split(pred_class, num_or_size_splits=20, axis=-1)
    pred_bboxes = tf.reshape(pred_bboxes, [-1, 4])
    indx_class = []
    scores_class = []
    for class_confidence in class_confidences:
        scores = pred_confidence * class_confidence#[1, 7, 7, 2]
        scores = tf.reshape(scores, [-1])
        indx = tf.image.non_max_suppression(pred_bboxes, scores, 5)
        indx_class.append(indx[tf.newaxis, :])
        scores_class.append(scores[tf.newaxis, :])
    indx_class = tf.concat(indx_class, axis=0)
    scores_class = tf.concat(scores_class, axis=0)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./save_para/.\\model.ckpt")

    INDX_CLASS, PRED_BBOXES, SCORES_CLASS = sess.run([indx_class, pred_bboxes, scores_class], feed_dict={inputs: img[np.newaxis, :, :, :]})
    bboxes, class_num = select_bbox_again(INDX_CLASS, SCORES_CLASS, PRED_BBOXES)
    for i in range(bboxes.shape[0]):
        try:
            img = draw_bbox(img, np.int32(bboxes[i]), OBJECT_NAMES[class_num[i]])
        except:
            continue
    Image.fromarray(np.uint8(img)).show()



if __name__ == "__main__":
    img = np.array(Image.open("./ironman.jpg").resize([448, 448]))
    test(img)
