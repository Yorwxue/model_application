import importlib
import time

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import skimage
import skimage.io
import skimage.transform

from model_functions import name_in_checkpoint

try:
    from data.imagenet_classes import *
except Exception as e:
    raise Exception("{} / download the file from: https://github.com/zsdonghao/tensorlayer/tree/master/example/data".format(e))


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 299, 299
    resized_img = skimage.transform.resize(crop_img, (299, 299))
    return resized_img


def print_prob(prob):
    synset = class_names
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1


# model define
# -------------------------------------------------------------------------
image_batch = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
network = importlib.import_module('models.inception_resnet_v2')
scope = network.inception_resnet_v2_arg_scope(
            weight_decay=0.0,
            batch_norm_decay=0.995,
            batch_norm_epsilon=0.001,
            activation_fn=tf.nn.relu
        )
with slim.arg_scope(scope):
    prelogits, _ = network.inception_resnet_v2(
        image_batch, is_training=False, dropout_keep_prob=1.0,
        num_classes=1001, reuse=None)

probs = tf.nn.softmax(prelogits)
# -------------------------------------------------------------------------

img1 = load_image("data/tiger.jpeg")
img1 = img1.reshape((1, 299, 299, 3))

# model_path = "./pretrained/inception_resnet_v2_2016_08_30.ckpt"
model_path = "./pretrained/tmp/model.ckpt"

variables_to_restore = slim.get_variables_to_restore()

# Restoring models with different variable names
# scope_name = "InceptionRestionResnetV2"
# variables_to_restore = {name_in_checkpoint(scope_name, var): var for var in variables_to_restore}

# Restore only the convolutional layers:
# variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
# network.init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

restorer = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    # Initialize variables
    # sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    # print("Model Initialized")

    restorer.restore(sess, model_path)

    # show variables
    for v in variables_to_restore:
        v_ = sess.run(v)
        print(v_)

    print("Model Restored")

    prob = sess.run(probs, feed_dict={image_batch: img1})
    print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing
