import os
import re
import json
import string
import operator
import embeddings
import collections
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

from ast import literal_eval
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, MetaData, ForeignKey

# One-time calls
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

import hdfs


def load_transform_data(hdfs=False):
    df = pd.DataFrame()
    if hdfs:
        with hdfs.open_file('/usr/local/hadoop/hadoopdata/hdfs/datanode/goodreads.csv', 'r') as f:
            df = pd.read_csv(f)
    else:
        engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
        with engine.connect() as db_conn:
            df = pd.read_sql('SELECT * FROM reviews', con=db_conn)
            print('loaded dataframe from MySQL. records: ', len(df))
            db_conn.close()

    # convert rating string to float
    df.rating = df.rating.astype(float).fillna(0.0)

    # remove NaN ratings and null reviews
    df = df[np.isfinite(df['rating'])]
    df = df[pd.notnull(df['review'])]

    # drop reviews without letters (words)
    alphabet = list(string.ascii_letters)
    df = df[(pd.concat([df.review.str.contains(word,regex=False) \
                                    for word in alphabet],axis=1)).sum(1) > 0]
    return df


def load_split_amzn_reviews(record_count=None, num_chunks=1):
    json_file = os.path.join(os.getcwd(), '../data/', 'reviews_Books_5.json')
    counter = 0
    for chunk in pd.read_json(json_file, chunksize=30000, lines=True):
        counter += 1
        df = chunk[['overall', 'reviewText']]
        df['sentiment'] = np.where(df['overall'] < 3, 0, np.where(df['overall'] > 4, 1, -1))
        df = df[df.sentiment != -1]
        df = df[['reviewText', 'sentiment']]
        if counter == num_chunks:
            break
    return df[:record_count]


def normalize_text(string, remove_stopwords=False, stem_words=False):
    """
    Remove punctuation, parentheses, question marks, etc.
    """
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower()
    string = string.replace("<br />", " ")
    string = string.replace(r"(\().*(\))|([^a-zA-Z'])",' ')
    string = string.replace('&', 'and')
    string = string.replace('@', 'at')
    string = string.replace('0', 'zero')
    string = string.replace('1', 'one')
    string = string.replace('2', 'two')
    string = string.replace('3', 'three')
    string = string.replace('4', 'four')
    string = string.replace('5', 'five')
    string = string.replace('6', 'six')
    string = string.replace('7', 'seven')
    string = string.replace('8', 'eight')
    string = string.replace('9', 'nine')
    string = string.split()
    if remove_stopwords:
        stop_words = stopwords.words('english')
        string = [w for w in string if w not in stop_words]
    if stem_words:
        ps = PorterStemmer()
        string = [ps.stem(w) for w in string]
    string = ' '.join(string)
    return re.sub(strip_special_chars, "", string)


def build_dicts(reviewText):
    """
    Build dictionaries mapping words to unique integer values.
    """
    counts = collections.Counter(reviewText).most_common()
    dictionary = {}
    for word, _ in counts:
        dictionary[word] = len(dictionary)
    return dictionary


def build_vocabulary():
    df = load_split_amzn_reviews()
    vocabulary = collections.defaultdict(lambda : 0)
    iter = 1
    for _, row in df.iterrows():
        review = row['reviewText']
        review = normalize_text(review, remove_stopwords=True, stem_words=True)
        words = review.split()
        for word in list(set(words)):
            w_key = vocabulary.get(word)
            if w_key is not None:
                continue
            else:
                vocabulary[word] = iter
            iter += 1
    vocabulary[''] = 0
    return vocabulary


def compress_word_embedding(vocabulary, embedding):
    """
    Build word embedding for Amazon reviews.
    :return: Numpy array of shape [data_vocab_size, embedding_size(300)]
    """
    ft_model = embeddings.get_fastText_embedding()
    vocabulary = build_vocabulary()
    compressed_embedding = np.zeros((len(vocabulary), 300), dtype='float32')
    for key, value in vocabulary.items():
        compressed_embedding[value] = ft_model[key].astype('float32')
    return compressed_embedding


def generate_data_batch(batch_size=100, max_seq_length=150, vocabulary=None, embeddings=None, training_split=0.8, train=True):
    """
    Generate a random training batch of size batch_size.
    """
    df = load_split_amzn_reviews()
    train_test_split_idx = int(training_split * len(df))
    if train:
        df = df[:train_test_split_idx]
    else:
        df = df[train_test_split_idx:]
        batch_size = 50

    batch_j = 0
    batch_x = None
    batch_y = None
    seq_lengths = np.zeros(batch_size, dtype='int32')

    if vocabulary is None:
        vocabulary = build_vocabulary()
    if embeddings is None:
        embeddings = np.random.rand(len(vocabulary), 300)

    while True:
        if train:
            df = df.sample(frac=0.25)
        for _, row in df.iterrows():
            review = row['reviewText']
            review = review.split()
            review = review[:max_seq_length]

            if batch_x is None:
                # batch_x = np.zeros((batch_size, max_seq_length), dtype='int32')
                batch_x = np.zeros((batch_size, max_seq_length, 300), dtype='float32')
                batch_y = np.zeros((batch_size, 2), dtype='float32')

            for k, word in enumerate(review):
                # batch_x[batch_j][k] = vocabulary[word]
                batch_x[batch_j][k] = embeddings[vocabulary[word]]
            batch_y[batch_j] = np.eye(2)[row['sentiment']]
            seq_lengths[batch_j] = len(review)
            batch_j += 1

            if batch_j == batch_size:
                return batch_x, batch_y, seq_lengths


# TODO
def get_tf_dataset(df):
    """
    Returns TensorFlow Dataset.

    :param df: Pandas dataframe containing review text and sentiment labels.
    :return: TensorFlow Dataset.
    """
    X_train, y_train, X_test, y_test = clean_reviews(df)

    dx_train = tf.data.Dataset.from_tensor_slices(X_train)
    dy_train = tf.data.Dataset.from_tensor_slices(y_train).map(lambda z: tf.one_hot(z, 2))
    train_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)
    dx_test = tf.data.Dataset.from_tensor_slices(X_train)
    dy_test = tf.data.Dataset.from_tensor_slices(y_train).map(lambda z: tf.one_hot(z, 2))
    test_dataset = tf.data.Dataset.zip((dx_train, dy_train)).shuffle(500).repeat().batch(30)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    next_element = iterator.get_next()


def get_sentence_matrix(sentence, batchSize, maxSeqLength, wordsList):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = normalize_text(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
        except IndexError:
            break
    return sentenceMatrix


def store_prediction_hdfs(sentiment):
    client_hdfs = InsecureClient('http://' + os.environ['IP_HDFS'] + ':50070')
    with client_hdfs.write('/usr/local/hadoop/hadoopdata/hdfs/datanode/sentiments.csv', encoding='utf-8') as writer:
        sentiment.to_csv(writer)


def store_prediction_mysql(sentiment):
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
    meta = MetaData()

    # reflect the existing reviews table
    reviews = Table('reviews', meta, autoload=True, autoload_with=engine)

    # create the sentiments table MetaData object
    sentiments = Table('sentiments', meta,
        Column('index', Integer, primary_key=True),
        Column('id', Integer, ForeignKey('reviews.id')),
        Column('class', Integer)
    )

    # create the sentiments table
    meta.create_all(engine)

    # connect to database and insert classification data
    with engine.connect() as db_conn:
        sentiment.to_sql('sentiments',
                        con=db_conn,
                        if_exists='append',
                        index=False)
        print('inserted dataframe to MySQL. records: ', len(sentiment))
        db_conn.close()


def load_sentiments():
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
    with engine.connect() as db_conn:
        df = pd.read_sql('SELECT r.id, r.title, r.genre, r.user, r.reviewDate, r.review, r.rating, s.class FROM reviews AS r INNER JOIN sentiments AS s ON r.id = s.id', con=db_conn)
        print('loaded dataframe from MySQL. records: ', len(df))
        db_conn.close()
    return df
