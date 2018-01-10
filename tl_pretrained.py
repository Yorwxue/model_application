import os
import tensorflow as tf
import tensorlayer as tl
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.alexnet import alexnet_v2
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
# from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import resnet_v2_152
# from tensorflow.contrib.slim.python.slim.nets.vgg import vgg_16
import skimage
import skimage.io
import skimage.transform
import time, os
import numpy as np

# model in .ckpt form
# ----------------------------------------------------------------------------------------------------------------------
# >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', save_dir='model', printable=True)
# - Save specific parameters.
# >>> tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', printable=True)
# - Load latest ckpt.
# >>> tl.files.load_ckpt(sess=sess, var_list=net.all_params, save_dir='model', printable=True)
# - Load specific ckpt.
# >>> tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=net.all_params, save_dir='model', is_latest=False, printable=True)
# ----------------------------------------------------------------------------------------------------------------------

# model in .npz form
# ----------------------------------------------------------------------------------------------------------------------
# >>> tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
# - Load model from npz (Method 1)
# >>> load_params = tl.files.load_npz(name='model.npz')
# >>> tl.files.assign_params(sess, load_params, network)
# - Load model from npz (Method 2)
# >>> tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)
# ----------------------------------------------------------------------------------------------------------------------

src_path, _ = os.path.split(os.path.realpath(__file__))
model_path = os.path.join(src_path, 'pretrained_model/inception_v3.ckpt')
npz_model_path = os.path.join(src_path, 'pretrained_model/inception_v3.npz')

# --------
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


## InceptionV3 / All TF-Slim nets can be merged into TensorLayer
x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
net_in = tl.layers.InputLayer(x, name='input_layer')
with slim.arg_scope(inception_v3_arg_scope()):
    ## Alternatively, you should implement inception_v3 without TensorLayer as follow.
    # logits, end_points = inception_v3(X, num_classes=1001,
    #                                   is_training=False)
    network = tl.layers.SlimNetsLayer(layer=net_in, slim_layer=inception_v3,
                                      slim_args={
                                             'num_classes': 1001,
                                             'is_training': False,
                                             #  'dropout_keep_prob' : 0.8,       # for training
                                             #  'min_depth' : 16,
                                             #  'depth_multiplier' : 1.0,
                                             #  'prediction_fn' : slim.softmax,
                                             #  'spatial_squeeze' : True,
                                             #  'reuse' : None,
                                             #  'scope' : 'InceptionV3'
                                             },
                                      name='InceptionV3'  # <-- the name should be the same with the ckpt model
                                      )

sess = tf.InteractiveSession()

network.print_params(False)

saver = tf.train.Saver()
if not os.path.isfile(model_path):
    print("Please download inception_v3 ckpt from : https://github.com/tensorflow/models/tree/master/research/slim")
    exit()
try:    # TF12+
    saver.restore(sess, model_path)
except:  # TF11
    saver.restore(sess, "inception_v3.ckpt")
print("Model Restored")

from scipy.misc import imread, imresize
y = network.outputs
probs = tf.nn.softmax(y)
# img1 = load_image("data/puzzle.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
img1 = load_image("data/tiger.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
# img1 = load_image("data/laska.png")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
img1 = img1.reshape((1, 299, 299, 3))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: img1})
print("End time : %.5ss" % (time.time() - start_time))
print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing


## You can save the model into npz file
tl.files.save_npz(network.all_params, name=npz_model_path)

# restore model from .npz (method 1)
load_params = tl.files.load_npz(name=npz_model_path)
tl.files.assign_params(sess, load_params, network)

print("Model Restored")

from scipy.misc import imread, imresize
y = network.outputs
probs = tf.nn.softmax(y)
# img1 = load_image("data/puzzle.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
img1 = load_image("data/tiger.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
# img1 = load_image("data/laska.png")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
img1 = img1.reshape((1, 299, 299, 3))

start_time = time.time()
prob = sess.run(probs, feed_dict={x: img1})
print("End time : %.5ss" % (time.time() - start_time))
print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing