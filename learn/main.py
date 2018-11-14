import pandas as pd
import numpy as np
from random import randint
import tensorflow as tf
import re
import sys
import utils


# variables used in RNN training
numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

# data structures for words
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] # Encode as UTF-8
wordVectors = np.load('wordVectors.npy')

# restore the graph
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# load pretrained network
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('converted_checkpoint'))

# store clean review data to data frame
df = utils.retrieve_data()

# classify reviews
result = np.zeros((df.shape[0],2))
for i, row in enumerate(df.itertuples(),0):
    result[i][0] = getattr(row,'id')
    inputText = getattr(row,'review')
    inputMatrix = utils.getSentenceMatrix(inputText, batchSize, maxSeqLength, wordsList)
    predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]
    if (predictedSentiment[0] > predictedSentiment[1]):
        result[i][1] = 1
    else:
        result[i][1] = -1

sentiment = pd.DataFrame(result).astype('int32')
sentiment.columns = ['id', 'class']

# store predictions in MySQL database
utils.store_prediction_mysql(sentiment)
