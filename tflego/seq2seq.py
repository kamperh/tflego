"""
Functions for building sequence-to-sequence models.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy as np
import tensorflow as tf

import blocks


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))


def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])


#-----------------------------------------------------------------------------#
#                                 SEQ2SEQ LEGO                                #
#-----------------------------------------------------------------------------#

def build_encdec_lazydynamic(x, x_lengths, n_hidden, rnn_type="lstm",
        keep_prob=1., **kwargs):
    """
    Encoder-decoder with the encoder state fed in at each decoding time step.

    The function name refers to the simple implementation essentially using
    `tf.nn.dynamic_rnn` for both the encoder and decoder. Since the encoding
    state is used as input at each decoding time step, the output of the
    decoder is never used. As in `build_encdec_outback`, a linear
    transformation is applied to the output of the decoder such that the final
    output dimensionality matches that of the input `x`.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]

    Return
    ------
    (encoder_states, decoder_output) : (Tensor, Tensor)
    """

    maxlength = x.get_shape().as_list()[-2]
    n_output = x.get_shape().as_list()[-1]

    # Encoder
    encoder_output, encoder_states = blocks.build_rnn(
        x, x_lengths, n_hidden, rnn_type, keep_prob, scope="rnn_encoder", **kwargs
        )
    if rnn_type == "lstm":
        encoder_states = encoder_states.h

    # Decoder

    # Repeat encoder states
    decoder_input = tf.reshape(
        tf.tile(encoder_states, [1, maxlength]), [-1, maxlength, n_hidden]
        )

    # Decoding RNN
    decoder_output, decoder_states = blocks.build_rnn(
        decoder_input, x_lengths, n_hidden, rnn_type, scope="rnn_decoder", **kwargs
        )

    # Final linear layer
    with tf.variable_scope("rnn_decoder/linear_output"):
        decoder_output = tf.reshape(decoder_output, [-1, n_hidden])
        decoder_output = blocks.build_linear(decoder_output, n_output)
        decoder_output = tf.reshape(decoder_output, [-1, maxlength, n_output])

    return (encoder_states, decoder_output)


def build_encdec_outback(x, x_lengths, n_hidden, rnn_type="lstm", keep_prob=1.,
        **kwargs):
    """
    Encoder-decoder where the output is fed back in the decoder.

    A linear transformation is applied to the decoder outputs. The last state
    of the encoder is given as the initial state of the decoder. Both the
    encoder and decoder consists of single recurrent layers.

    Parameters
    ----------
    x : Tensor [n_data, maxlength, d_in]

    Return
    ------
    (encoder_states, decoder_output) : (Tensor, Tensor)
    """

    maxlength = x.get_shape().as_list()[-2] 
    n_output = x.get_shape().as_list()[-1]

    # Encoder
    encoder_output, encoder_states = blocks.build_rnn(
        x, x_lengths, n_hidden, rnn_type, keep_prob, scope="rnn_encoder", **kwargs
        )

    # Decoder
    decoder_output = build_rnn_decoder_outback(
        encoder_states, n_hidden, n_output, maxlength, rnn_type, dtype=x.dtype, **kwargs
        )

    return (encoder_states, decoder_output)


def build_rnn_decoder_outback(initial_state, n_hidden, n_output, maxlength,
        rnn_type="lstm", initial_prev_output=None, dtype=None, **kwargs):
    """The output is fed back as input in this decoder."""

    with tf.variable_scope("rnn_decoder") as scope:
    
        # RNN decoder cell
        cell = blocks.build_rnn_cell(n_hidden, rnn_type, **kwargs)

        # Initial state and input
        state = initial_state
        if rnn_type == "lstm":
            batch_size = tf.shape(initial_state[0])[0]  # cell and hidden states together in tuple
        elif rnn_type == "gru" or rnn_type == "rnn":
            batch_size = tf.shape(initial_state)[0]
        if initial_prev_output is not None:
            prev_output = initial_prev_output
        else:
            if not dtype:
                raise ValueError("If no initial_prev_output is provided, dtype must be.")
            prev_output = tf.zeros([batch_size, n_output], dtype)
            prev_output.set_shape([None, n_output])
        
        # Decode over timesteps
        outputs = []
        for i_time in xrange(maxlength):
            if i_time > 0:
                scope.reuse_variables()
            output, state = cell(prev_output, state)
            with tf.variable_scope("linear_output"):
                output = blocks.build_linear(output, n_output)
            outputs.append(output)
            prev_output = output

        return pack_sequence(outputs)


def np_encdec_lazydynamic(x, x_lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, maxlength=None):

    if maxlength is None:
        maxlength = max(x_lengths)

    # Encoder
    encoder_output = blocks.np_rnn(x, x_lengths, W_encoder, b_encoder, maxlength)
    encoder_states = []
    for i_data, l in enumerate(x_lengths):
        encoder_states.append(encoder_output[i_data, l - 1, :])
    encoder_states = np.array(encoder_states)

    # Decoder

    # Repeat encoder states
    n_hidden = W_encoder.shape[-1]
    decoder_input = np.reshape(np.repeat(encoder_states, maxlength, axis=0), [-1, maxlength, n_hidden])
    
    # Decoding RNN
    decoder_output = blocks.np_rnn(decoder_input, x_lengths, W_decoder, b_decoder, maxlength)

    # Final linear layer
    decoder_output_linear =  np.zeros(x.shape)
    for i_data in xrange(x.shape[0]):
        cur_decoder_sequence = decoder_output[i_data, :, :]
        for i_step, cur_decoder in enumerate(cur_decoder_sequence):
            decoder_output_linear[i_data, i_step, :] = blocks.np_linear(
                cur_decoder, W_output, b_output
                )
    decoder_output = decoder_output_linear

    return encoder_states, decoder_output


def np_encdec_outback(x, x_lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, maxlength=None):

    if maxlength is None:
        maxlength = max(x_lengths)
    
    # Encoder
    encoder_output = blocks.np_rnn(x, x_lengths, W_encoder, b_encoder, maxlength)
    encoder_states = []
    for i_data, l in enumerate(x_lengths):
        encoder_states.append(encoder_output[i_data, l - 1, :])
    encoder_states = np.array(encoder_states)

    # Decoder
    decoder_output = np_rnn_decoder_outback(
        encoder_states, W_decoder, b_decoder, W_output, b_output, maxlength
        )
    return encoder_states, decoder_output


def np_rnn_decoder_outback(initial_states, W_decoder, b_decoder, W_output,
        b_output, max_length):
    """
    In this decoder the output is fed back as input.
    
    Parameters
    ----------
    initial_hiddens : matrix [batch_size, n_hidden]
    """
    outputs = []
    for i_data in xrange(initial_states.shape[0]):
        output = []
        prev_output = np.zeros(W_output.shape[1])
        prev_state = initial_states[i_data]
        for i_time in xrange(max_length):
            cur_state = np.tanh(np.dot(np.hstack((prev_output, prev_state)), W_decoder) + b_decoder)
            cur_output = np.dot(cur_state, W_output) + b_output
            prev_state = cur_state
            prev_output = cur_output
            output.append(cur_output)
        outputs.append(output)
    return np.array(outputs)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import numpy.testing as npt

    from blocks import NP_DTYPE, TF_DTYPE, NP_ITYPE, TF_ITYPE


    # Encoder-decoder feeding in encoder state at each decoder input

    tf.reset_default_graph()

    # Random seed
    np.random.seed(2)
    tf.set_random_seed(2)

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

    print tf_encoder_states
    print tf_decoder_output

    np_encoder_states, np_decoder_output = np_encdec_lazydynamic(
        test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
        W_output, b_output, n_maxlength
        )

    print np_encoder_states
    print np_decoder_output

    npt.assert_almost_equal(tf_encoder_states, np_encoder_states, decimal=5)
    npt.assert_almost_equal(tf_decoder_output, np_decoder_output, decimal=5)


    # # Encoder-decoder feeding back output as input in decoder

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
    # network_tuple = build_encdec_outback(x, x_lengths, n_hidden, rnn_type=rnn_type)
    # encoder_states = network_tuple[0]
    # decoder_output = network_tuple[1]
    # with tf.variable_scope("rnn_encoder/BasicRNNCell/Linear", reuse=True):
    #     W_encoder = tf.get_variable("Matrix")
    #     b_encoder = tf.get_variable("Bias")
    # with tf.variable_scope("rnn_decoder/BasicRNNCell/Linear", reuse=True):
    #     W_decoder = tf.get_variable("Matrix")
    #     b_decoder = tf.get_variable("Bias")
    # with tf.variable_scope("rnn_decoder/linear_output", reuse=True):
    #     W_output = tf.get_variable("W")
    #     b_output = tf.get_variable("b")

    # # TensorFlow graph
    # init = tf.initialize_all_variables()
    # with tf.Session() as session:
    #     session.run(init)

    #     # Output
    #     tf_encoder_states = encoder_states.eval({x: test_data, x_lengths: lengths})
    #     tf_decoder_output = decoder_output.eval({x: test_data, x_lengths: lengths})

    #     # Weights
    #     W_encoder = W_encoder.eval()
    #     b_encoder = b_encoder.eval()
    #     W_decoder = W_decoder.eval()
    #     b_decoder = b_decoder.eval()
    #     W_output = W_output.eval()
    #     b_output = b_output.eval()

    # # print tf_encoder_states
    # # print tf_decoder_output

    # np_encoder_states, np_decoder_output = np_encdec_outback(
    #     test_data, lengths, W_encoder, b_encoder, W_decoder, b_decoder,
    #     W_output, b_output, n_maxlength
    #     )

    # # print np_encoder_states
    # # print np_decoder_output

    # npt.assert_almost_equal(tf_encoder_states, np_encoder_states, decimal=5)
    # npt.assert_almost_equal(tf_decoder_output, np_decoder_output, decimal=5)


if __name__ == "__main__":
    main()
