"""
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

import numpy as np
import numpy.testing as npt
import scipy.spatial.distance as distance

from blocks import NP_DTYPE, NP_ITYPE, TF_DTYPE
from siamese import *
import blocks


def test_siamese():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_data = 3
    n_input = 28*28
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Model parameters
    n_hiddens = [256, 10]

    # Tensorflow model
    x_l = tf.placeholder(TF_DTYPE, [None, n_input])
    x_r = tf.placeholder(TF_DTYPE, [None, n_input])
    keep_prob = 1.0
    model_param_dict = {"n_hiddens": n_hiddens, "keep_prob": keep_prob}
    model_l, model_r = build_siamese(x_l, x_r, blocks.build_feedforward, model_param_dict)

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        model_l_output = model_l.eval({x_l: test_data})
        model_r_output = model_r.eval({x_r: test_data})
        
    npt.assert_almost_equal(model_l_output, model_r_output)


def test_siamese_triplets():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_data = 3
    n_input = 28*28
    test_data = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

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

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)

        # Output
        model_a_output = model_a.eval({x_a: test_data})
        model_s_output = model_s.eval({x_s: test_data})
        model_d_output = model_d.eval({x_d: test_data})
        
    npt.assert_almost_equal(model_a_output, model_s_output)
    npt.assert_almost_equal(model_a_output, model_d_output)


def test_cos_distance():

    tf.reset_default_graph()

    # Random seed
    np.random.seed(1)
    tf.set_random_seed(1)

    # Test data
    n_data = 4
    n_input = 28*28
    test_data1 = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)
    test_data2 = np.asarray(np.random.randn(n_data, n_input), dtype=NP_DTYPE)

    # Tensorflow model
    x_1 = tf.placeholder(TF_DTYPE, [None, n_input])
    x_2 = tf.placeholder(TF_DTYPE, [None, n_input])
    d = cos_distance(x_1, x_2)

    # TensorFlow graph
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        tf_outpout = d.eval({x_1: test_data1, x_2: test_data2})

    # Numpy
    np_output = []
    for x_1, x_2 in zip(test_data1, test_data2):
        np_output.append(distance.cosine(x_1, x_2) / 2.0)
    np_output = np.array(np_output)

    npt.assert_almost_equal(np_output, tf_outpout)


def test_loss_triplets_cos():

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
