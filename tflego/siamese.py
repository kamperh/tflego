"""
Functions for building Siamese models and related loss functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy as np
import tensorflow as tf


#-----------------------------------------------------------------------------#
#                                 SIAMESE LEGO                                #
#-----------------------------------------------------------------------------#

def build_siamese(x_l, x_r, model_build_func, model_param_dict,
        x_l_lengths=None, x_r_lengths=None):
    """
    Build Siamese model by using `model_build_func` for left and right sides.

    Parameters
    ----------
    model_param_dict : dict
        The parameters used to construct both sides of the model.
    x_l_lengths : Tensor [n_data]
        If provided, the lengths are also passed to `model_build_func`.

    Return
    ------
    (model_l, model_r) : (Tensor, Tensor)
    """

    assert (
        (x_l_lengths is None and x_r_lengths is None) or (x_l_lengths is not
        None and x_r_lengths is not None)
        )

    # Parameters for the left and right sides
    model_l_params_dict = model_param_dict.copy()
    model_r_params_dict = model_param_dict.copy()
    model_l_params_dict["x"] = x_l
    model_r_params_dict["x"] = x_r
    if x_l_lengths is not None:
        model_l_params_dict["x_lengths"] = x_l_lengths
        model_r_params_dict["x_lengths"] = x_r_lengths

    # Siamese model
    with tf.variable_scope("siamese") as scope:
        model_l = model_build_func(**model_l_params_dict)
        scope.reuse_variables()
        model_r = model_build_func(**model_r_params_dict)

    return (model_l, model_r)


def build_siamese_triplet(x_a, x_s, x_d, model_build_func, model_param_dict,
        x_a_lengths=None, x_s_lengths=None, x_d_lengths=None):
    """
    Build a Siamese triplets model.

    Parameters `x_a` and `x_s` are of the same type, while `x_D` is different.
    See `build_siamese` for more details.
    """

    assert (
        (x_a_lengths is None and x_s_lengths is None and x_d_lengths is None) or
        (x_a_lengths is not None and x_s_lengths is not None and x_d_lengths is not None)
        )

    # Parameters for the left and right sides
    model_a_params_dict = model_param_dict.copy()
    model_s_params_dict = model_param_dict.copy()
    model_d_params_dict = model_param_dict.copy()
    model_a_params_dict["x"] = x_a
    model_s_params_dict["x"] = x_s
    model_d_params_dict["x"] = x_d
    if x_a_lengths is not None:
        model_a_params_dict["x_lengths"] = x_a_lengths
        model_s_params_dict["x_lengths"] = x_s_lengths
        model_d_params_dict["x_lengths"] = x_d_lengths

    # Siamese model
    with tf.variable_scope("siamese") as scope:
        model_a = model_build_func(**model_a_params_dict)
        scope.reuse_variables()
        model_s = model_build_func(**model_s_params_dict)
        model_d = model_build_func(**model_d_params_dict)

    return (model_a, model_s, model_d)



#-----------------------------------------------------------------------------#
#                                LOSS FUNCTIONS                               #
#-----------------------------------------------------------------------------#

def norm(x):
    # return tf.sqrt(tf.reduce_sum(tf.square(x), 1, keep_dims=True))
    return tf.sqrt(tf.reduce_sum(tf.square(x), 1))


def cos_similarity(x_1, x_2):
    # return (
    #     tf.reduce_sum(tf.mul(x_1, x_2), -1, keep_dims=True) / (norm(x_1) * norm(x_2))
    #     )
    return tf.reduce_sum(tf.mul(x_1, x_2), -1) / (norm(x_1) * norm(x_2))


def cos_distance(x_1, x_2):
    return (1 - cos_similarity(x_1, x_2)) / 2.0


def loss_triplets_cos(x_a, x_s, x_d, margin):
    """
    Parameters `x_a` and `x_s` are of the same type, while `x_d` is different.
    """
    return tf.reduce_mean(tf.maximum(0., margin + cos_distance(x_a, x_s) - cos_distance(x_a, x_d)))


def np_loss_triplets_cos(x_a, x_s, x_d, margin):
    import scipy.spatial.distance as distance
    np_cos_distance = lambda x_1, x_2: distance.cosine(x_1, x_2) / 2.0
    losses = []
    for i_pair in xrange(x_a.shape[0]):
        print i_pair,
        cur_x_a = x_a[i_pair]
        cur_x_s = x_s[i_pair]
        cur_x_d = x_d[i_pair]
        losses.append(max([
            0., margin + np_cos_distance(cur_x_a, cur_x_s) - np_cos_distance(cur_x_a, cur_x_d)
            ]))
    return np.mean(losses)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():

    import numpy as np
    import numpy.testing as npt

    from blocks import NP_DTYPE, NP_ITYPE, TF_DTYPE
    import blocks

    # Siamese feedforward neural network

    import scipy.spatial.distance as distance

    def generate_matches_array(labels):
        """
        Return an array of bool in the same order as the distances from
        `scipy.spatial.distance.pdist` indicating whether a distance is for
        matching or non-matching labels.
        """
        N = len(labels)
        matches = np.zeros(N*(N - 1)/2, dtype=np.bool)

        # For every distance, mark whether it is a true match or not
        cur_matches_i = 0
        for n in range(N - 1):
            cur_label = labels[n]
            matches[cur_matches_i:cur_matches_i + (N - n) - 1] = np.asarray(labels[n + 1:]) == cur_label
            cur_matches_i += N - n - 1

        return matches

    tf.reset_default_graph()

    # Random seed
    np.random.seed(2)
    tf.set_random_seed(2)

    # Test data
    n_data = 7
    n_classes = 3
    n_input = 28
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)
    labels = np.asarray(np.random.randint(n_classes, size=n_data), dtype=NP_ITYPE)
    matches_vec = generate_matches_array(labels)
    same_matrix = distance.squareform(matches_vec)
    print "Labels:", labels
    print "Matches:\n", same_matrix

    # Process same and different pairs
    I, J = np.where(np.triu(same_matrix))  # indices of same pairs
    x_a_indices = []
    x_s_indices = []
    for i, j in zip(I, J):
        x_a_indices.append(i)
        x_s_indices.append(j)
        x_a_indices.append(j)
        x_s_indices.append(i)
    np.fill_diagonal(same_matrix, 1)
    x_d_indices = -1*np.ones(len(x_a_indices), dtype=NP_ITYPE)
    for i_token in range(same_matrix.shape[0]):
        cur_matches = np.where(np.array(x_a_indices) == i_token)[0]
        if cur_matches.shape[0] > 0:
            x_d_indices[cur_matches] = np.random.choice(
                np.where(same_matrix[i_token] == False)[0],
                size=len(cur_matches), replace=True
                )
    print "x_a_indices:", x_a_indices
    print "x_s_indices:", x_s_indices
    print "x_d_indices:", list(x_d_indices)

    # Model parameters
    margin = 0.1

    # Tensorflow model
    x_a = tf.placeholder(TF_DTYPE, [None, n_input])
    x_s = tf.placeholder(TF_DTYPE, [None, n_input])
    x_d = tf.placeholder(TF_DTYPE, [None, n_input])
    loss = loss_triplets_cos(x_a, x_s, x_d, margin)

    # Tensorflow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        tf_loss = loss.eval({
            x_a: test_data[x_a_indices],
            x_s: test_data[x_s_indices],
            x_d: test_data[x_d_indices]
            })

    np_loss = np_loss_triplets_cos(
        test_data[x_a_indices], test_data[x_s_indices], test_data[x_d_indices],
        margin
        )

    print "tf_loss:", tf_loss
    print "np_loss:", np_loss

    npt.assert_almost_equal(np_loss, tf_loss)

    return


    # Model parameters
    n_hiddens = [256, 10]

    # Tensorflow model
    x_a = tf.placeholder(TF_DTYPE, [None, n_input])
    x_s = tf.placeholder(TF_DTYPE, [None, n_input])
    x_d = tf.placeholder(TF_DTYPE, [None, n_input])
    keep_prob = 1.0
    model_param_dict = {"n_hiddens": n_hiddens, "keep_prob": keep_prob}
    model_a, model_s, model_d = build_siamese_triplet(
        x_a, x_s, x_d, blocks.build_feedforward, model_param_dict
        )
    with tf.variable_scope("siamese"):
        with tf.variable_scope("ff_layer_0", reuse=True):
            W_0 = tf.get_variable("W", dtype=TF_DTYPE)
            b_0 = tf.get_variable("b", dtype=TF_DTYPE)
        with tf.variable_scope("ff_layer_1", reuse=True):
            W_1 = tf.get_variable("W", dtype=TF_DTYPE)
            b_1 = tf.get_variable("b", dtype=TF_DTYPE)

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        model_a_output = model_a.eval({x_a: test_data})
        model_s_output = model_s.eval({x_s: test_data})
        model_d_output = model_d.eval({x_d: test_data})
        
        # Weights
        W_0 = W_0.eval()
        b_0 = b_0.eval()
        W_1 = W_1.eval()
        b_1 = b_1.eval()

    npt.assert_almost_equal(model_a_output, model_s_output)
    npt.assert_almost_equal(model_a_output, model_d_output)


if __name__ == "__main__":
    main()
