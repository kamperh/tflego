"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy.testing as npt

from blocks import *


def test_feedforward():

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 28*28
    n_data = 3
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Model parameters
    n_classes = 10
    n_hiddens = [256, 256]

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_input])
    keep_prob = tf.placeholder(TF_DTYPE)
    ff = build_feedforward(x, n_hiddens, keep_prob)
    with tf.variable_scope("ff_layer_final"):
        ff = build_linear(ff, n_classes)
    with tf.variable_scope("ff_layer_0", reuse=True):
        W_0 = tf.get_variable("W", dtype=TF_DTYPE)
        b_0 = tf.get_variable("b", dtype=TF_DTYPE)
    with tf.variable_scope("ff_layer_1", reuse=True):
        W_1 = tf.get_variable("W", dtype=TF_DTYPE)
        b_1 = tf.get_variable("b", dtype=TF_DTYPE)
    with tf.variable_scope("ff_layer_final", reuse=True):
        W_2 = tf.get_variable("W", dtype=TF_DTYPE)
        b_2 = tf.get_variable("b", dtype=TF_DTYPE)

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_output = ff.eval({x: test_data, keep_prob: 1.0})
        
        # Weights
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()
        W_2 = W_2.eval()
        b_2 = b_2.eval()

    # Numpy model
    np_output = np_linear(np_feedforward(test_data, [W_0, W_1], [b_0, b_1]), W_2, b_2)

    npt.assert_almost_equal(np_output, tf_output, decimal=5)


def test_cnn():

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 28*28
    n_data = 3
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Model parameters
    input_shape = [-1, 28, 28, 1] # [n_data, height, width, d_in]
    filter_shapes = [
        [5, 5, 1, 32],
        [5, 5, 32, 64]
        ]
    pool_shapes = [
        [2, 2], 
        [2, 2]
        ]

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_input])
    cnn = build_cnn(x, input_shape, filter_shapes, pool_shapes, padding="VALID")
    cnn = tf.contrib.layers.flatten(cnn)
    with tf.variable_scope("cnn_layer_0", reuse=True):
        W_0 = tf.get_variable("W")
        b_0 = tf.get_variable("b")
    with tf.variable_scope("cnn_layer_1", reuse=True):
        W_1 = tf.get_variable("W")
        b_1 = tf.get_variable("b")

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        
        # Output
        tf_output = cnn.eval({x: test_data})
        
        # Parameters
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()

    # Numpy model
    np_output = np_cnn(test_data, input_shape, [W_0, W_1], [b_0, b_1], pool_shapes)
    np_output = np_output.reshape(np_output.shape[0], -1)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_rnn():

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 10
    n_data = 11
    n_maxlength = 12
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in xrange(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 13
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    rnn_outputs, rnn_states = build_rnn(x, x_lengths, n_hidden, rnn_type=rnn_type)
    with tf.variable_scope("RNN/BasicRNNCell/Linear", reuse=True):
        W = tf.get_variable("Matrix")
        b = tf.get_variable("Bias")

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        
        # Output
        tf_output = rnn_outputs.eval({x: test_data, x_lengths: lengths})
        
        # Weights
        W = W.eval()
        b = b.eval()

    # Numpy model
    np_output = np_rnn(test_data, lengths, W, b, n_maxlength)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)


def test_multi_rnn():

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 10
    n_data = 11
    n_maxlength = 12
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in xrange(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hiddens = [13, 14]
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    rnn_outputs, rnn_states = build_multi_rnn(x, x_lengths, n_hiddens, rnn_type=rnn_type)
    with tf.variable_scope("rnn_layer_0/RNN/BasicRNNCell/Linear", reuse=True) as vs:
        W_0 = tf.get_variable("Matrix")
        b_0 = tf.get_variable("Bias")
    with tf.variable_scope("rnn_layer_1/RNN/BasicRNNCell/Linear", reuse=True) as vs:
        W_1 = tf.get_variable("Matrix")
        b_1 = tf.get_variable("Bias")

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_output = rnn_outputs.eval({x: test_data, x_lengths: lengths})

        # Weights
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()

    # Numpy model
    np_output = np_multi_rnn(test_data, lengths, [W_0, W_1], [b_0, b_1], n_maxlength)

    npt.assert_almost_equal(tf_output, np_output, decimal=5)

