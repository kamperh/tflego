"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy.testing as npt

from blocks import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE
from seq2seq import *


def test_encdec_lazydynamic():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 4
    n_data = 3
    n_maxlength = 5
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in xrange(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 6
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    network_tuple = build_encdec_lazydynamic(x, x_lengths, n_hidden, rnn_type=rnn_type)
    encoder_states = network_tuple[0]
    decoder_output = network_tuple[1]
    with tf.variable_scope("rnn_encoder/BasicRNNCell/Linear", reuse=True):
        W_encoder = tf.get_variable("Matrix")
        b_encoder = tf.get_variable("Bias")
    with tf.variable_scope("rnn_decoder/BasicRNNCell/Linear", reuse=True):
        W_decoder = tf.get_variable("Matrix")
        b_decoder = tf.get_variable("Bias")
    with tf.variable_scope("rnn_decoder/linear_output", reuse=True):
        W_output = tf.get_variable("W")
        b_output = tf.get_variable("b")

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_encoder_states = encoder_states.eval({x: test_data, x_lengths: lengths})
        tf_decoder_output = decoder_output.eval({x: test_data, x_lengths: lengths})

        # Weights
        W_encoder = W_encoder.eval()
        b_encoder = b_encoder.eval()
        W_decoder = W_decoder.eval()
        b_decoder = b_decoder.eval()
        W_output = W_output.eval()
        b_output = b_output.eval()

    # print tf_encoder_states
    # print tf_decoder_output

    np_encoder_states, np_decoder_output = np_encdec_lazydynamic(
        test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, n_maxlength
        )

    # print np_encoder_states
    # print np_decoder_output

    npt.assert_almost_equal(tf_encoder_states, np_encoder_states, decimal=5)
    npt.assert_almost_equal(tf_decoder_output, np_decoder_output, decimal=5)


def test_encdec_outback():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_input = 4
    n_data = 3
    n_maxlength = 5
    test_data = np.zeros((n_data, n_maxlength, n_input), dtype=NP_DTYPE)
    lengths = []
    for i_data in xrange(n_data):
        length = np.random.randint(1, n_maxlength + 1)
        lengths.append(length)
        test_data[i_data, :length, :] = np.random.randn(length, n_input)
    lengths = np.array(lengths, dtype=NP_ITYPE)

    # Model parameters
    n_hidden = 6
    rnn_type = "rnn"

    # TensorFlow model
    x = tf.placeholder(TF_DTYPE, [None, n_maxlength, n_input])
    x_lengths = tf.placeholder(TF_DTYPE, [None])
    network_tuple = build_encdec_outback(x, x_lengths, n_hidden, rnn_type=rnn_type)
    encoder_states = network_tuple[0]
    decoder_output = network_tuple[1]
    with tf.variable_scope("rnn_encoder/BasicRNNCell/Linear", reuse=True):
        W_encoder = tf.get_variable("Matrix")
        b_encoder = tf.get_variable("Bias")
    with tf.variable_scope("rnn_decoder/BasicRNNCell/Linear", reuse=True):
        W_decoder = tf.get_variable("Matrix")
        b_decoder = tf.get_variable("Bias")
    with tf.variable_scope("rnn_decoder/linear_output", reuse=True):
        W_output = tf.get_variable("W")
        b_output = tf.get_variable("b")

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        tf_encoder_states = encoder_states.eval({x: test_data, x_lengths: lengths})
        tf_decoder_output = decoder_output.eval({x: test_data, x_lengths: lengths})

        # Weights
        W_encoder = W_encoder.eval()
        b_encoder = b_encoder.eval()
        W_decoder = W_decoder.eval()
        b_decoder = b_decoder.eval()
        W_output = W_output.eval()
        b_output = b_output.eval()

    # print tf_encoder_states
    # print tf_decoder_output

    np_encoder_states, np_decoder_output = np_encdec_outback(
        test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, n_maxlength
        )

    # print np_encoder_states
    # print np_decoder_output

    npt.assert_almost_equal(tf_encoder_states, np_encoder_states, decimal=5)
    npt.assert_almost_equal(tf_decoder_output, np_decoder_output, decimal=5)
