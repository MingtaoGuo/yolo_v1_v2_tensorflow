import tensorflow as tf
from ops import yolo_loss, sub_net
from utils import read_batch
from vgg16 import vgg16


BATCH_SIZE = 32
img_path = "./VOCdevkit/VOC2007/JPEGImages/"
xml_path = "./VOCdevkit/VOC2007/Annotations/"


def train():
    inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
    labels = tf.placeholder(tf.float32, [None, 13, 13, 9, 25])
    lr = tf.placeholder(tf.float32)
    features = vgg16(inputs)
    prediction = sub_net(features)
    loss, loss_bbox, loss_conf_obj, loss_conf_noobj, loss_classes = yolo_loss(prediction, labels)
    Opt = tf.train.AdamOptimizer(lr).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    LR = 0
    for i in range(20000):
        if i < 10000:
            LR = 1e-4
        if i >= 10000 and i < 15000:
            LR = 1e-5
        if i >= 15000:
            LR = 1e-6
        batch_img, batch_labels = read_batch(img_path, xml_path, BATCH_SIZE)
        sess.run(Opt, feed_dict={inputs: batch_img, labels: batch_labels, lr: LR})
        [LOSS, b, c, n, cl] = sess.run([loss, loss_bbox, loss_conf_obj, loss_conf_noobj, loss_classes], feed_dict={inputs: batch_img, labels: batch_labels})
        print("Iteration: %d, Loss: %f, bbox loss: %f, conf obj loss: %f, conf no obj loss: %f, classes loss: %f"%(i, LOSS, b, c, n, cl))
        if i % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")


if __name__ == "__main__":
    train()
