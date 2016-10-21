"""
Functions for building the basic components of neural networks.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy as np
import tensorflow as tf

NP_DTYPE = np.float32
TF_DTYPE = tf.float32
NP_ITYPE = np.int32
TF_ITYPE = tf.int32


#-----------------------------------------------------------------------------#
#                           FEEDFORWARD LEGO BLOCKS                           #
#-----------------------------------------------------------------------------#

def build_linear(x, n_output):
    n_input = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        "W", [n_input, n_output], dtype=TF_DTYPE, initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [n_output], dtype=TF_DTYPE, initializer=tf.random_normal_initializer()
        )
    return tf.matmul(x, W) + b


def build_feedforward(x, n_hiddens, keep_prob=1.):
    """
    Build a feedforward neural network.
    
    Parameters
    ----------
    n_hiddens : list
        Hidden units in each of the layers.
    """
    for i_layer, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope("ff_layer_{}".format(i_layer)):
            x = build_linear(x, n_hidden)
            x = tf.nn.relu(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def np_relu(x):
    return np.maximum(0., x)


def np_linear(x, W, b):
    return np.dot(x, W) + b


def np_feedforward(x, weights, biases):
    """
    Push the input `x` through the feedforward neural network. The `weights`
    and `biases` should be lists of the parameters of each layer.
    """
    for W, b in zip(weights, biases):
        x = np_relu(np_linear(x, W, b))
    return x


#-----------------------------------------------------------------------------#
#                          CONVOLUTIONAL LEGO BLOCKS                          #
#-----------------------------------------------------------------------------#

def build_conv2d_relu(x, filter_shape, stride=1, padding="VALID"):
    """Single convolutional layer with bias and ReLU activation."""
    W = tf.get_variable(
        "W", filter_shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()
        )
    b = tf.get_variable(
        "b", [filter_shape[-1]], dtype=tf.float32, initializer=tf.random_normal_initializer()
        )
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def build_maxpool2d(x, pool_shape, padding="VALID"):
    """Max pool over `x` using a `pool_shape` of [pool_height, pool_width]."""
    ksize = [1,] + pool_shape + [1,]
    return tf.nn.max_pool(x, ksize=ksize, strides=ksize, padding=padding)


def build_cnn(x, input_shape, filter_shapes, pool_shapes, padding="VALID"):
    """
    Build a convolutional neural network (CNN).
    
    As an example, a CNN with single-channel [28, 28] shaped input with two
    convolutional layers can be constructud using:
    
        x = tf.placeholder(tf.float32, [None, 28*28])
        input_shape = [-1, 28, 28, 1] # [n_data, height, width, d_in]
        filter_shapes = [
            [5, 5, 1, 32],  # filter shape of first layer
            [5, 5, 32, 64]  # filter shape of second layer
            ]   
        pool_shapes = [
            [2, 2],         # pool shape of first layer
            [2, 2]          # pool shape of second layer
            ]
        cnn = build_cnn(x, input_shape, filter_shapes, pool_shapes)
    
    Parameters
    ----------
    x : Tensor [n_data, n_input]
        Input to the CNN, which is reshaped to match `input_shape`.
    input_shape : list
        The shape of the input to the CNN as [n_data, height, width, d_in].
    filter_shapes : list of list
        The filter shape of each layer as [height, width, d_in, d_out].
    pool_shape : list of list
        The pool shape of each layer as [height, width].
    """
    assert len(filter_shapes) == len(pool_shapes)
    x = tf.reshape(x, input_shape)
    cnn = x
    for i_layer, (filter_shape, pool_shape) in enumerate(zip(filter_shapes, pool_shapes)):
        with tf.variable_scope("cnn_layer_{}".format(i_layer)):
            cnn = build_conv2d_relu(cnn, filter_shape, padding=padding)
            cnn = build_maxpool2d(cnn, pool_shape, padding=padding)
    return cnn


def np_conv2d(x, filters, padding="valid"):
    """
    Calculate the convolution of `x` using `filters`.
    
    A useful tutorial: http://www.robots.ox.ac.uk/~vgg/practicals/cnn/.
    
    Parameters
    ----------
    x : matrix [n_data, height, width, d_in]
    filters : matrix [filter_height, filter_width, d_in, d_out]
    """

    import scipy.signal

    # Dimensions
    n_data, height, width, d_in = x.shape
    filter_height, filter_width, _, d_out = filters.shape
    assert d_in == _
    
    # Loop over data
    conv_over_data = []
    for i_data in xrange(n_data):
        # Loop over output channels
        conv_over_channels = []
        for i_out_channel in xrange(d_out):
            conv_result = 0.
            # Loop over input channels
            for i_in_channel in xrange(d_in):
                conv_result += scipy.signal.correlate(
                    x[i_data, :, :, i_in_channel], filters[:, :, i_in_channel,
                    i_out_channel], mode=padding
                    )
            conv_over_channels.append(conv_result)
        conv_over_data.append(np.transpose(np.array(conv_over_channels), (1, 2, 0)))
    
    return np.array(conv_over_data)


def np_maxpool2d(x, pool_shape, ignore_border=False):
    """
    Performs max pooling on `x`.
    
    Parameters
    ----------
    x : matrix [n_data, height, width, d_in]
        Input over which pooling is performed.
    pool_shape : list
        Gives the pooling shape as (pool_height, pool_width).
    """
    
    # Dimensions
    n_data, height, width, d_in = x.shape
    pool_height, pool_width = pool_shape
    round_func = np.floor if ignore_border else np.ceil
    output_height = int(round_func(1.*height/pool_height))
    output_width = int(round_func(1.*width/pool_width))

    # Max pool
    max_pool = np.zeros((n_data, output_height, output_width, d_in))
    for i_data in xrange(n_data):
        for i_channel in xrange(d_in):
            for i in xrange(output_height):
                for j in xrange(output_width):
                    max_pool[i_data, i, j, i_channel] = np.max(x[
                        i_data,
                        i*pool_height:i*pool_height + pool_height,
                        j*pool_width:j*pool_width + pool_width,
                        i_channel
                        ])
    
    return max_pool


def np_cnn(x, input_shape, weights, biases, pool_shapes):
    """
    Push the input `x` through the CNN with `cnn_specs` matching the parameters
    passed to `build_cnn`, `weights` and `biases` the parameters of each
    convolutional layer.
    """
    cnn = x.reshape(input_shape)
    for W, b, pool_shape in zip(weights, biases, pool_shapes):
        cnn = np_relu(np_maxpool2d(np_conv2d(cnn, W) + b, pool_shape))
    return cnn


#-----------------------------------------------------------------------------#
#                            RECURRENT LEGO BLOCKS                            #
#-----------------------------------------------------------------------------#

def build_rnn_cell(n_hidden, rnn_type="lstm", **kwargs):
    """
    The `kwargs` parameters are passed directly to the constructor of the cell
    class, e.g. peephole connections can be used by adding `use_peepholes=True`
    when `rnn_type` is "lstm".
    """
    if rnn_type == "lstm":
        cell_args = {"state_is_tuple": True}  # default LSTM parameters
        cell_args.update(kwargs)
        cell = tf.nn.rnn_cell.LSTMCell(n_hidden, **cell_args)
    elif rnn_type == "gru" or rnn_type == "rnn":
        cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden, **kwargs)
    else:
        assert False, "Invalid RNN type: {}".format(rnn_type)
    return cell


def build_rnn(x, x_lengths, n_hidden, rnn_type="lstm", keep_prob=1., scope=None, **kwargs):
    """
    Build a recurrent neural network (RNN) with architecture `rnn_type`.
    
    The RNN is dynamic, with `x_lengths` giving the lengths as a Tensor with
    shape [n_data]. The input `x` should be padded to have shape [n_data,
    n_padded, d_in].
    
    Parameters
    ----------
    rnn_type : str
        Can be "lstm", "gru" or "rnn".
    kwargs : dict
        These are passed directly to the constructor of the cell class, e.g.
        peephole connections can be used by adding `use_peepholes=True` when
        `rnn_type` is "lstm".
    """
    
    # RNN cell
    cell = build_rnn_cell(n_hidden, rnn_type)
    
    # Dropout
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1., output_keep_prob=keep_prob)
    
    # Dynamic RNN
    return tf.nn.dynamic_rnn(cell, x, sequence_length=x_lengths, dtype=TF_DTYPE, scope=scope)


def build_multi_rnn(x, x_lengths, n_hiddens, rnn_type="lstm", keep_prob=1., **kwargs):
    """
    Build a multi-layer RNN.
    
    Apart from those below, parameters are similar to that of `build_rnn`.

    Parameters
    ----------
    n_hiddens : list
        Hidden units in each of the layers.
    """
    for i_layer, n_hidden in enumerate(n_hiddens):
        with tf.variable_scope("rnn_layer_{}".format(i_layer)):
            outputs, states = build_rnn(x, x_lengths, n_hidden, rnn_type, keep_prob, **kwargs)
            x = outputs
    return outputs, states


def np_rnn(x, x_lengths, W, b, maxlength=None):
    """Calculates the output for a basic RNN."""
    if maxlength is None:
        maxlength = max(x_lengths)
    outputs = np.zeros((x.shape[0], maxlength, W.shape[1]))
    for i_data in xrange(x.shape[0]):
        cur_x_sequence = x[i_data, :x_lengths[i_data], :]
        prev_state = np.zeros(W.shape[1])
        for i_step, cur_x in enumerate(cur_x_sequence):
            cur_state = np.tanh(np.dot(np.hstack((cur_x, prev_state)), W) + b)
            outputs[i_data, i_step, :] = cur_state
            prev_state = cur_state
    return outputs


def np_multi_rnn(x, x_lengths, weights, biases, maxlength=None):
    """
    Push the input `x` through the RNN. The `weights`
    and `biases` should be lists of the parameters of each layer.
    """
    for W, b in zip(weights, biases):
        x = np_rnn(x, x_lengths, W, b, maxlength)
    return x


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import numpy.testing as npt


    # # RNN

    # tf.reset_default_graph()

    # # Random seed
    # np.random.seed(2)
    # tf.set_random_seed(2)

    # # Test data
    # n_input = 4
    # n_data = 3
    # n_maxlength = 5
    # test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    # lengths = []
    # for i_data in xrange(n_data):
    #     length = np.random.randint(1, n_maxlength + 1)
    #     lengths.append(length)
    #     test_data[i_data, :length, :] = np.random.randn(length, n_input)
    # lengths = np.array(lengths, dtype=NP_ITYPE)

    # # Model parameters
    # n_hidden = 6
    # rnn_type = "rnn"

    # # TensorFlow model
    # x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    # x_lengths = tf.placeholder(TF_DTYPE, [None])
    # rnn_outputs, rnn_states = build_rnn(x, x_lengths, n_hidden, rnn_type=rnn_type)
    # with tf.variable_scope("RNN/BasicRNNCell/Linear", reuse=True):
    #     W = tf.get_variable("Matrix")
    #     b = tf.get_variable("Bias")

    # # TensorFlow graph
    # init = tf.initialize_all_variables()
    # with tf.Session() as session:
    #     session.run(init)
        
    #     # Output
    #     tf_output = rnn_outputs.eval({x: test_data, x_lengths: lengths})
        
    #     # Weights
    #     W = W.eval()
    #     b = b.eval()

    # # Numpy model
    # np_output = np_rnn(test_data, lengths, W, b, n_maxlength)

    # npt.assert_almost_equal(tf_output, np_output, decimal=5)


    # CNN

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


    # # Feedforward neural network

    # # Random seed
    # np.random.seed(1)
    # tf.set_random_seed(1)

    # # Test data
    # n_input = 28*28
    # n_data = 3
    # test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # # Model parameters
    # n_classes = 10
    # n_hiddens = [256, 256]

    # # TensorFlow model
    # x = tf.placeholder(TF_DTYPE, [None, n_input])
    # keep_prob = tf.placeholder(TF_DTYPE)
    # ff = build_feedforward(x, n_hiddens, keep_prob)
    # with tf.variable_scope("ff_layer_final"):
    #     ff = build_linear(ff, n_classes)
    # with tf.variable_scope("ff_layer_0", reuse=True):
    #     W_0 = tf.get_variable("W", dtype=TF_DTYPE)
    #     b_0 = tf.get_variable("b", dtype=TF_DTYPE)
    # with tf.variable_scope("ff_layer_1", reuse=True):
    #     W_1 = tf.get_variable("W", dtype=TF_DTYPE)
    #     b_1 = tf.get_variable("b", dtype=TF_DTYPE)
    # with tf.variable_scope("ff_layer_final", reuse=True):
    #     W_2 = tf.get_variable("W", dtype=TF_DTYPE)
    #     b_2 = tf.get_variable("b", dtype=TF_DTYPE)

    # # TensorFlow graph
    # init = tf.initialize_all_variables()
    # with tf.Session() as session:
    #     session.run(init)

    #     # Output
    #     tf_output = ff.eval({x: test_data, keep_prob: 1.0})
        
    #     # Weights
    #     W_0 = W_0.eval()
    #     b_0 = b_0.eval()
    #     W_1 = W_1.eval()
    #     b_1 = b_1.eval()
    #     W_2 = W_2.eval()
    #     b_2 = b_2.eval()

    # # Numpy model
    # np_output = np_linear(np_feedforward(test_data, [W_0, W_1], [b_0, b_1]), W_2, b_2)

    # import numpy.testing as npt
    # npt.assert_almost_equal(tf_output, np_output, decimal=5)


if __name__ == "__main__":
    main()
