#!/usr/bin/python
#coding:utf-8

import sys
import tensorflow as tf
import numpy as np
import scipy
from scipy import ndimage


file_path = sys.argv[1]

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(tf.float32, [1, n_H0, n_W0, n_C0], name = "X")
    Y = tf.placeholder(tf.float32, [1, n_y], name = "Y")
    ### END CODE HERE ###

    return X, Y

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1, 8, 8, 1], strides = [1,8, 8, 1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1, 4, 4, 1], strides = [1, 4, 4, 1], padding = 'SAME')
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
    ### END CODE HERE ###
    return Z3

def predict(X, parameters, sess):

    # X = tf.placeholder(tf.float32, [1, 64, 64, 3], name = "X")
    Z3 = forward_propagation(X, parameters)

    # print("Z3 = " + str(a))


if __name__ == "__main__":
    # image = np.array(ndimage.imread(file_path, flatten = False))
    # my_image = scipy.misc.imresize(image, size = (64, 64), mode = 'RGB')
    # # my_image = my_image / 255.
    # my_image_work = np.expand_dims(my_image, 0)
    # print("Using a picture of shape", my_image_work.shape, "for the prediction")
    # print (my_image.shape)
    image = np.array(ndimage.imread(file_path, flatten=False))
    image = image/255.
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
    # img = np.asarray(file_path, dtype='float32') / 256.
    # image = image.astype('float32')
    # my_image = image.reshape(1, 64, 64, 3)
    # tf.reset_default_graph()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        graph = tf.get_default_graph()
        new_saver = tf.train.import_meta_graph('my_test_model.meta')
        # new_saver.restore(sess, 'my_test_model')
        new_saver.restore(sess, 'my_test_model')
        predict_op = tf.get_collection('predict_op')[0]
        dataFlowGraph = tf.get_default_graph()
        x = dataFlowGraph.get_tensor_by_name("X:0")
        prediction = sess.run(predict_op, feed_dict = {x: my_image})
        print("\nThe predicted image class is:", str(np.squeeze(prediction)))
        # predict(my_image_work, parameters, sess)
