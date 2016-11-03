"""
Training functions.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2016
"""

from datetime import datetime
import numpy as np
import sys
import tensorflow as tf
import timeit


def train_fixed_epochs(n_epochs, optimizer, train_loss_tensor,
        train_feed_iterator, feed_placeholders, test_loss_tensor=None,
        test_feed_iterator=None, load_model_fn=None, save_model_fn=None,
        config=None, epoch_offset=0):
    """
    Train for a fixed number of epochs.
    
    TO-DO: Optional validation (and save best model separately).
    
    Parameters
    ----------
    train_loss : Tensor
        The function that is optimized; should match the feed specified through
        `train_feed_iterator` and `feed_placeholders`. This can also be a list
        of Tensors, in which case the training loss is output as an array.
    train_feed_batch_iterator : generator
        Generates the values for the `feed_placeholders` for each training
        batch.
    feed_placeholders : list of placeholder
        The placeholders that is required for the `train_loss` (and optionally
        `test_loss`) feeds.
    load_model_fn : str
        If provided, initialize session from this file.
    save_model_fn : str
        If provided, save session to this file.
    
    Return
    ------
    record_dict : dict
        Statistics tracked during training. Each key describe the statistic,
        while the value is a list of (epoch, value) tuples.
    """
    
    # Statistics
    record_dict = {}
    record_dict["epoch_time"] = []
    record_dict["train_loss"] = []
    if test_loss_tensor is not None:
        record_dict["test_loss"] = []
    
    print datetime.now()
    
    def feed_dict(vals):
        return {key: val for key, val in zip(feed_placeholders, vals)}

    # Launch the graph
    saver = tf.train.Saver()
    if load_model_fn is None:
        init = tf.initialize_all_variables()
    with tf.Session(config=config) as session:
        
        # Start or restore session
        if load_model_fn is None:
            session.run(init)
        else:
            saver.restore(session, load_model_fn)
    
        # Train
        for i_epoch in xrange(n_epochs):
            print("Epoch {}:".format(epoch_offset + i_epoch)),
            start_time = timeit.default_timer()
            
            # # Train model
            # train_losses = []
            # for cur_feed in train_feed_iterator:
            #     _, cur_loss = session.run(
            #         [optimizer, train_loss_tensor],
            #         feed_dict=feed_dict(cur_feed)
            #         )
            #     train_losses.append(cur_loss)
            
            # # Test model
            # if test_loss_tensor is not None:
            #     test_losses = []
            #     for cur_feed in test_feed_iterator:
            #         cur_loss = session.run(
            #             [test_loss_tensor],
            #             feed_dict=feed_dict(cur_feed)
            #             )
            #         test_losses.append(cur_loss)
            #     test_loss = np.mean(test_losses)
            #     record_dict["test_loss"].append((i_epoch, test_loss))

            # Train model
            train_losses = []
            if not isinstance(train_loss_tensor, (list, tuple)):
                for cur_feed in train_feed_iterator:
                    _, cur_loss = session.run(
                        [optimizer, train_loss_tensor],
                        feed_dict=feed_dict(cur_feed)
                        )
                    train_losses.append(cur_loss)
                train_loss = np.mean(train_losses)
            else:
                for cur_feed in train_feed_iterator:
                    cur_loss = session.run(
                        [optimizer] + train_loss_tensor,
                        feed_dict=feed_dict(cur_feed)
                        )
                    cur_loss.pop(0)  # remove the optimizer
                    cur_loss = np.array(cur_loss)
                    train_losses.append(cur_loss)
                train_loss = np.mean(train_losses, axis=0)
            record_dict["train_loss"].append((i_epoch, train_loss))

            # Test model
            if test_loss_tensor is not None:
                test_losses = []
                if not isinstance(test_loss_tensor, (list, tuple)):
                    for cur_feed in test_feed_iterator:
                        cur_loss = session.run(
                            [test_loss_tensor],
                            feed_dict=feed_dict(cur_feed)
                            )
                        test_losses.append(cur_loss)
                    test_loss = np.mean(test_losses)
                else:
                    for cur_feed in test_feed_iterator:
                        cur_loss = session.run(
                            test_loss_tensor,
                            feed_dict=feed_dict(cur_feed)
                            )
                        cur_loss = np.array(cur_loss)
                        test_losses.append(cur_loss)
                    test_loss = np.mean(test_losses, axis=0)
                record_dict["test_loss"].append((i_epoch, test_loss))

            # Statistics
            end_time = timeit.default_timer()
            epoch_time = end_time - start_time
            record_dict["epoch_time"].append((i_epoch, epoch_time))
            
            # log = "{:.3f} sec, train loss: {:.5f}".format(epoch_time, train_loss)
            log = "{:.3f} sec".format(epoch_time)
            log += ", train loss: " + str(train_loss)
            if test_loss is not None:
                # log += ", test loss: {:.5f}".format(test_loss)
                log += ", test loss: " + str(test_loss)
            print log
            sys.stdout.flush()

        if save_model_fn is not None:
            print "Writing: {}".format(save_model_fn)
            saver.save(session, save_model_fn)
            
    total_time = sum([i[1] for i in record_dict["epoch_time"]])
    print "Training time: {:.3f} min".format(total_time/60.)
    
    print datetime.now()
    return record_dict
