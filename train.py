from networks import vgg16
from utils import *
from ops import *


BATCH_SIZE = 32
img_path = "./VOCdevkit/VOC2007/JPEGImages/"
xml_path = "./VOCdevkit/VOC2007/Annotations/"


def train():
    inputs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    labels = tf.placeholder(tf.float32, [None, 7, 7, 25])
    prediction = vgg16(inputs)
    loss, loss_bboxes, loss_confidence_obj, loss_confidence_noobj, loss_class = yolo_loss(prediction, labels)
    Opt = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # saver.restore(sess, "./save_para/.\\model.ckpt")
    for i in range(10000):
        batch_img, batch_labels = read_batch(img_path, xml_path, BATCH_SIZE)
        sess.run(Opt, feed_dict={inputs: batch_img, labels: batch_labels})
        [LOSS, b, o, n, c] = sess.run([loss, loss_bboxes, loss_confidence_obj, loss_confidence_noobj, loss_class], feed_dict={inputs: batch_img, labels: batch_labels})
        print("Iteration: %d, Loss: %f, loss_bbox: %f, loss_confi_obj: %f, loss_confi_noobj: %f, loss_class: %f"%(i, LOSS, b, o, n, c))
        if i % 500 == 0:
            saver.save(sess, "./save_para/model.ckpt")


if __name__ == "__main__":
    train()