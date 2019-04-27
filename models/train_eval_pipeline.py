import os
import sys
sys.path.append(os.path.abspath('../utils'))

import time
import utils
import pickle
import datetime
import embeddings
import numpy as np
import tensorflow as tf
from BiLSTM import BiLSTM


flags = tf.flags
flags.DEFINE_string('data_dir', '../data', 'Data directory')
flags.DEFINE_string('checkpoints_dir', 'checkpoints',
                       'Checkpoints directory. Parameters will be saved there')
flags.DEFINE_string('summaries_dir', 'logs',
                       'Directory where TensorFlow summaries will be stored')
flags.DEFINE_integer('batch_size', 100, 'Batch size')
flags.DEFINE_integer('max_seq_length', 256, 'Max sequence length')
flags.DEFINE_integer('train_steps', 300, 'Number of training steps')
flags.DEFINE_integer('hidden_size', 75, 'Hidden size of LSTM layer')
flags.DEFINE_integer('embedding_size', 300, 'Size of embeddings layer')
flags.DEFINE_float('learning_rate', 0.01, 'RMSProp learning rate')
flags.DEFINE_float('dropout_keep_prob', 0.5, '0<dropout_keep_prob<=1. Dropout keep-probability')
FLAGS = flags.FLAGS


def main():
    """
    Entry point for training and evaluation.
    """
    # Summaries
    summaries_dir = '{0}/{1}'.format(FLAGS.summaries_dir,
                                     datetime.datetime.now().strftime('%d_%b_%Y-%H_%M_%S'))
    train_writer = tf.summary.FileWriter(summaries_dir + '/train')
    validation_writer = tf.summary.FileWriter(summaries_dir + '/validation')

    # Model directory
    model_name = str(int(time.time()))
    model_dir = '{0}/{1}'.format(FLAGS.checkpoints_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save configuration
    FLAGS(sys.argv)
    # config = FLAGS.__dict__['__flags']
    config = FLAGS
    with open('{}/config.pkl'.format(model_dir), 'wb') as f:
        pickle.dump(config, f)


    # Load word embeddings model
    # ft_model = embeddings.get_fastText_embedding()
    ft_model = {}
    ft_model['one'] = np.random.rand(300)
    ft_model['two'] = np.random.rand(300)

    with tf.Session() as sess:
        model = BiLSTM(hidden_size=[FLAGS.hidden_size], word_embeddings=ft_model,
            embedding_dim=300, max_seq_length=FLAGS.max_seq_length,
            learning_rate=FLAGS.learning_rate, doc_vocab_size=120000)

        # Saver object
        saver = tf.train.Saver()

        # Restore checkpoint
        if args.checkpoint:
            saver.restore(sess, args.checkpoint)

        # Train model
        global_step = 0
        sess.run(tf.initialize_global_variables())


        # TODO implement tf.Dataset.
        # sess.run(model.dataset_iterator.make_initializer(train_dataset))

        for epoch in xrange(FLAGS.epochs):
            # train_batch_generator = utils.generate_data_batch(FLAGS.batch_size)
            # for X_train, y_train in train_batch_generator:
            X_train, y_train = utils.generate_data_batch(batch_size=FLAGS.batch_size, max_seq_length=FLAGS.max_seq_length, word_embeddings=ft_model)
            feeds_train = [
                model.loss,
                model.train_step,
                model.merged
                # model.embedding_lookup
            ]
            feed_dict_train = {
            model.input: X_train,
            model.target: y_train,
            # model.seq_len: train_seq_len,   # Use fixed-length sequences.
            model.dropout_keep_prob: FLAGS.dropout_keep_prob
            # model.embedding_init
            }

            try:
                train_loss, _, summary = sess.run(feeds_train, feed_dict_train)
            except Exception as e:
                debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                raise e

            train_writer.add_summary(summary, global_step)
            print('{0}/{1} train loss: {2:.4f}'.format(global_step + 1, FLAGS.train_steps, train_loss))

            # Check validation performance
            if (global_step + 1) % 101 == 0:
                # TODO implement tf.Dataset.
                # validation_init_op = iterator.make_initializer(valid_dataset)

                X_val, y_val = utils.generate_data_batch(train=False)
                feed_val = [
                    model.loss,
                    model.accuracy,
                    model.merged
                    # model.embedding_lookup
                ]
                feed_dict_val = {
                    model.input: X_val,
                    model.target: y_val,
                    # model.seq_len: val_seq_len,   # Use fixed-length sequences.
                    model.dropout_keep_prob: 1
                    # model.embedding_init
                }
                try:
                    val_loss, accuracy, summary = sess.run(feed_val, feed_dict_val)
                except Exception as e:
                    debug_data(seq_source_ids, seq_ref_ids, seq_source_len, seq_ref_len, id_to_vocab)
                    raise e

                validation_writer.add_summary(summary, i)
                print('   validation loss: {0:.4f} (accuracy {1:.4f})'.format(val_loss, accuracy))

                global_step += 1
            # End train batch

            save_path = saver.save(sess, '{}/model.ckpt'.format(model_dir), global_step=global_step)
        # End epoch

        # evaluate()
    # End sess


def parse_arguments():
    """
    Argument parser configuration.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default="logs", help="Log directory to store tensorboard summary and model checkpoints")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Mini batch size")
    parser.add_argument('--max_seq_length', type=int, default=40, help="Maximum sequence length.  Sentence lengths beyond this are truncated.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden dimension size")
    parser.add_argument('--keep_prob', type=float, default=0.8, help="Keep probability for dropout")
    parser.add_argument('--decoder', type=str, choices=['greedy', 'sample'], help="Decoder type")
    # parser.add_argument('--mode', type=str, default=None, choices=['train', 'dev', 'test', 'infer'], help='train or dev or test or infer or minimize')
    parser.add_argument('--checkpoint', type=str, default=None, help="Model checkpoint file")
    parser.add_argument('--minimize_graph', type=bool, default=False, help="Save existing checkpoint to minimal graph")

    return parser.parse_args()


if __name__ == '__main__':
    main()
    # tf.app.run()
    # df = utils.load_split_amzn_reviews(150)
