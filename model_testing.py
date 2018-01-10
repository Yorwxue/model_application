import importlib
import time
from datetime import datetime

import numpy as np
from scipy import misc
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
from tensorflow.python.framework import graph_util
from models.inception_resnet_v2 import inception_resnet_v2_arg_scope

from model_functions import load_model

subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
print('subdir: ', subdir)

# model_path = "./pretrained/InceptionResnetV2_facenet/model.ckpt"
model_path = "./pretrained/davidsandberg_facenet/model.ckpt"
# model_path = "./pretrained/tmp/model.ckpt"

# dataset
face_img = misc.imread("data/Aaron_Eckhart_0001.png")
face_img = face_img.reshape((1, 160, 160, 3))

network = importlib.import_module('models.inception_resnet_v1')
# network = importlib.import_module('models.inception_resnet_v2')


with tf.Graph().as_default():
    # define model
    # -------------------------------------------------------------------------
    image_batch = tf.placeholder(tf.float32, shape=(None, 160, 160, 3))
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    # inception resnet v1
    prelogits, _ = network.inference(image_batch, 1.0, phase_train=phase_train_placeholder,
                                     bottleneck_layer_size=128, weight_decay=0.0)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # inception resnet v2
    # scope = network.inception_resnet_v2_arg_scope(
    #     weight_decay=0.0,
    #     batch_norm_decay=0.995,
    #     batch_norm_epsilon=0.001,
    #     activation_fn=tf.nn.relu
    # )
    # with slim.arg_scope(scope):
    #     prelogits, _ = network.inception_resnet_v2(
    #         image_batch, is_training=phase_train_placeholder, dropout_keep_prob=1.0,
    #         num_classes=128, reuse=None)
    # embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    # -------------------------------------------------------------------------

    variables_to_restore = slim.get_variables_to_restore()

    # show variables
    # for v in variables_to_restore:
    #     print(v)
    # exit()

    # Partially Restoring Models
    # variables_to_restore = slim.get_variables_to_restore(exclude=["v1"])

    saver = tf.train.Saver(variables_to_restore)
    restorer = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        # Initialize variables
        # sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        # print("Model Initialized")

        print("loadind model from %s" % model_path)
        restorer.restore(sess, model_path)
        print("Model Restored")

        # forward pass
        feed_dict = {image_batch: face_img, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        print(emb)

        # collect variables
        # tf.add_to_collection('vars', tf.global_variables())
        # all_vars = tf.get_collection('vars')

        # show variables
        # for v in all_vars:
        #     v_ = sess.run(v)
        #     print(v_)

        # save model
        # save_path = "./pretrained/tmp/model.ckpt"
        # saver.save(sess, "./pretrained/tmp/model.ckpt")
        # print("Model saved in file: %s" % save_path)

# ----

with tf.Graph().as_default():
    with tf.Session() as sess:
        load_model('./pretrained/davidsandberg_facenet/20170512-110547.pb')

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: face_img, phase_train_placeholder: False}
        emb_array = sess.run(embeddings, feed_dict=feed_dict)

print(emb_array)

print('Finish')
