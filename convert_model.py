import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl
slim = tf.contrib.slim
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base, inception_v3, inception_v3_arg_scope
from models.inception_resnet_v1 import *
import skimage
import skimage.io
import skimage.transform

# -------------------------
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


# some parameter need to change
# inception v3
# ------------------------------------------------------
def ckpt_to_npz(ckpt_model_path, npz_model_path):
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
    if not os.path.isfile(ckpt_model_path):
        print("Please download inception_v3 ckpt from : https://github.com/tensorflow/models/tree/master/research/slim")
        exit()
    try:  # TF12+
        saver.restore(sess, ckpt_model_path)
    except:  # TF11
        saver.restore(sess, "inception_v3.ckpt")
    print("Model Restored")

    # evaluate
    y = network.outputs
    probs = tf.nn.softmax(y)
    # img1 = load_image("data/puzzle.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    img1 = load_image(
        "data/tiger.jpeg")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    # img1 = load_image("data/laska.png")  # test data in github: https://github.com/zsdonghao/tensorlayer/tree/master/example/data
    img1 = img1.reshape((1, 299, 299, 3))

    start_time = time.time()
    prob = sess.run(probs, feed_dict={x: img1})
    print("End time : %.5ss" % (time.time() - start_time))
    print_prob(prob[0][1:])  # Note : as it have 1001 outputs, the 1st output is nothing

    # save the model into npz file
    tl.files.save_npz(network.all_params, name=npz_model_path)

# ------------------------------------------------------


# can't work
def pb_to_ckpt(pb_model_path, ckpt_model_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # saver=tf.train.Saver()
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="prefix")
            sess = tf.Session(graph=graph)
            saver = tf.train.Saver(tf.global_variables())
            saver.save(sess, ckpt_model_path)
            print("Model saved to ckpt format")


# not sure
def ckpt_to_pb():
    # Get the current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Current directory : ", dir_path)
    save_dir = os.path.join(dir_path, 'Protobufs')

    graph = tf.get_default_graph()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    print("Restoring the model to the default graph ...")
    saver = tf.train.import_meta_graph(dir_path + '/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(dir_path))
    print("Restoring Done .. ")

    print("Saving the model to Protobuf format: ", save_dir)

    # Save the model to protobuf  (pb and pbtxt) file.
    tf.train.write_graph(sess.graph_def, save_dir, "Binary_Protobuf.pb", False)
    tf.train.write_graph(sess.graph_def, save_dir, "Text_Protobuf.pbtxt", True)
    print("Saving Done .. ")


if __name__ == '__main__':
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # pb_model_path = os.path.join(src_path, 'pretrained/inception_resnet_of_facenet.pb')
    ckpt_model_path = os.path.join(src_path, 'pretrained/inception_v3.ckpt')
    npz_model_path = os.path.join(src_path, 'pretrained/inception_v3.npz')
    # pb_to_ckpt(pb_model_path, ckpt_model_path)
    ckpt_to_npz(ckpt_model_path, npz_model_path)
