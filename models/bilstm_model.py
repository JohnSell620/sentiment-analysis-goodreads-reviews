import datetime as dt
import tensorflow as tf


class BiLSTM(object):
    def __init__(self, hidden_size, word_embeddings, emmbedding_len, doc_vocab_size, max_seq_length, n_classes=2, learning_rate=0.01):
        """
        Builds the TensorFlow BiLSTM model.

        :param hidden_size: Number of units in the LSTM cell of each rnn layer.
        :param word_embeddings: Word embeddings.
        :param emmbedding_len: Dimension of 1-D word embeddings.
        :param max_seq_length: Maximum length of an input tensor.
        :param n_classes: (Optional) Number of classification classes.
        :param learning_rate: (Optional) Learning rate of RMSProp algorithm.
        """
        # Define placeholders.
        self.input = tf.placeholder(tf.int32, [None, max_seq_length], name='input')
        # self.input = tf.placeholder(tf.int32, [None, max_seq_length, emmbedding_len], name='input')
        self.seq_len = tf.placeholder(tf.int32, [None], name='lengths')
        self.target = tf.placeholder(tf.float32, [None, n_classes], name='target')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Define NN parameters.
        self.word_embeddings = self.__embedding_layer(
            self.input,
            word_embeddings,
            emmbedding_len,
            doc_vocab_size)
        self.scores = self.__scores(
            self.word_embeddings,
            self.seq_len,
            hidden_size,
            n_classes,
            self.dropout_keep_prob,
            random_state)
        self.predict = self._prediction(self.scores)
        self.losses = self.__losses(self.scores, self.target)
        self.loss = self.__loss(self.losses)
        self.train_step = self.__train_step(learning_rate, self.loss)
        self.accuracy = self.__accuracy(self.predict, self.target)
        self.merged = tf.merge_all_summaries()

    # TODO
    def __dataset_iterator():
        """
        Returns TensorFlow Dataset.
        """
        X_train, y_train, X_test, y_test = clean_reviews(df)
        dx_train = tf.data.Dataset.from_tensor_slices(X_train)
        dy_train = tf.data.Dataset.from_tensor_slices(y_train).map(lambda z: tf.one_hot(z, 2))
        train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
        return tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)


    def __bilstm_layer(input_data, num_layers, rnn_size, seq_len, keep_prob):
        """
        BiLSTM layer with rnn_size units per LSTM cell.

        :param input_data: Input data with shape [batch_size, max_seq_len, emmbedding_len].
        :param num_layers: Number of layers in RNN.
        :param rnn_size: Number of units in LSTM cells.
        :param seq_len: Length of input sequence.
        :param keep_prob: Dropout keep probability.
        :return output: BiLSTM layer.
        """
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer),reuse=tf.AUTO_REUSE):

                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
                cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1, seed=2))
                cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_pro=keep_prob)

                outputs, states = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw,
                    cell_bw,
                    input_data,
                    sequence_length=seq_len,
                    dtype=tf.float32)
                output = tf.concat(outputs,2)
        return output

    def __embedding_layer(self, X, word_embeddings, embedding_len, doc_vocab_size):
        """
        Embedding layer with shape [vocab_size, embedding_size].

        :param X: Input with shape [batch_size, max_seq_length]
        :param word_embeddings: Word embeddings (Glove, fastText, or Elmo)
        :return: Embedding lookup tensor with shape [batch_size, max_length, embedding_size]
        """
        with tf.name_scope('word_embeddings'):
            # embeddings = tf.Variable(tf.random_uniform([doc_vocab_size, embedding_size], -1, 1, seed=seed))
            embeddings = tf.Variable(
                name='word_embeddings',
                shape=[doc_vocab_size, embedding_len],
                initializer=tf.constant_initializer(word_embeddings),
                trainable=True)
            embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_len])
            embedding_init = embeddings.assign(embedding_placeholder)
            embedded_words = tf.nn.embedding_lookup(embeddings, X)
        return embedded_words


    def __scores(self, embedded_words, seq_len, hidden_size, n_classes, dropout_keep_prob, random_state=None):
        """
        Builds the Bidirectional LSTM layers and the final fully connected layer.

        :param embedded_words: Embedding lookup tensor with shape [batch_size, max_length, embedding_size].
        :param seq_len: Sequence length tensor with shape [batch_size].
        :param hidden_size: Array holding the number of units in the LSTM cell of each rnn layer.
        :param n_classes: Number of classification classes.
        :param dropout_keep_prob: Tensor holding the dropout keep probability.
        :param random_state: Optional. Random state for the dropout wrapper.
        :return: Linear activation of each class with shape [batch_size, n_classes].
        """
        # Build LSTM layers
        for h in hidden_size:
            X_unstack = tf.unstack(embedded_words, seq_length, 1)
            outputs = self.__bilstm_layer(X_unstack, h, len(hidden_size), seq_length, dropout_keep_prob)

        # Shape of outputs: [batch_size, max_seq_len, hidden_size].
        outputs = tf.reduce_mean(outputs, reduction_indices=[1])

        # Shape of outputs: [batch_size, hidden_size]. Build fully connected layer
        with tf.name_scope('final_layer/weights'):
            w = tf.Variable(tf.truncated_normal([hidden_size[-1], n_classes], seed=random_state))
            self.variable_summaries(w, 'final_layer/weights')
        with tf.name_scope('final_layer/biases'):
            b = tf.Variable(tf.constant(0.1, shape=[n_classes]))
            self.variable_summaries(b, 'final_layer/biases')
        with tf.name_scope('final_layer/wx_plus_b'):
            scores = tf.nn.xw_plus_b(outputs, w, b, name='scores')
            tf.histogram_summary('final_layer/xw_plus_b', scores)
        return scores


    def _prediction(self, scores):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes].
        :return: Softmax activations with shape [batch_size, n_classes].
        """
        with tf.name_scope('final_layer/softmax'):
            softmax = tf.nn.softmax(scores, name='predictions')
            tf.histogram_summary('final_layer/softmax', softmax)
        return softmax


    def __losses(self, scores, target):
        """
        :param scores: Linear activation of each class with shape [batch_size, n_classes].
        :param target: Target tensor with shape [batch_size, n_classes].
        :return: Cross entropy losses with shape [batch_size].
        """
        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(scores, target, name='cross_entropy')
        return cross_entropy


    def __loss(self, losses):
        """
        :param losses: Cross entropy losses with shape [batch_size].
        :return: Mean of cross entropy loss.
        """
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(losses, name='loss')
            tf.scalar_summary('loss', loss)
        return loss


    def __train_step(self, learning_rate, loss):
        """
        :param learning_rate: Adam Learning rate.
        :param loss: Mean of cross entropy loss.
        :return: Adam optimizer training step.
        """
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)


    def __accuracy(self, predict, target):
        """
        :param predict: Softmax activations with shape [batch_size, n_classes].
        :param target: Target tensor with shape [batch_size, n_classes].
        :return: Mean accuracy obtained per batch.
        """
        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
            tf.scalar_summary('accuracy', accuracy)
        return accuracy
