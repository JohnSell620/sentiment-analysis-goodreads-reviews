import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from random import randint
from utils.utils import utils


# RNN and training parameters
numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

# Load word embeddings
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] # Encode as UTF-8
wordVectors = np.load('wordVectors.npy')

# Reset the computation graph
tf.reset_default_graph()

# Define placeholders
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

# Load word vectors
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

# Construct the RNN
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

# Define weight and bias variables and matrix multiplication
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

# Compute prediction accuracy
correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Load pretrained network
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('converted_checkpoint'))

# Retrive raw input data
df = utils.load_transform_data()

# Classify reviews
result = np.zeros((df.shape[0],2))
for i, row in enumerate(df.itertuples(),0):
    result[i][0] = getattr(row,'id')
    inputText = getattr(row,'review')
    inputMatrix = utils.get_sentence_matrix(inputText, batchSize, maxSeqLength, wordsList)
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    if (predictedSentiment[0] > predictedSentiment[1]):
        result[i][1] = 1
    else:
        result[i][1] = -1

sentiment = pd.DataFrame(result).astype('int32')
sentiment.columns = ['id', 'class']

# Store predictions in MySQL database or HDFS
utils.store_prediction_hdfs(sentiment)
