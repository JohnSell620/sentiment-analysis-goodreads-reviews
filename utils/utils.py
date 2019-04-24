import os
import re
import json
import string
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, MetaData, ForeignKey

# One-time calls
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

# import pydoop.hdfs as hdfs
import hdfs


def load_split_amzn_reviews():
    json_file = os.path.join(os.getcwd(), '../data/', 'reviews_Books_5.json')
    counter = 0
    for chunk in pd.read_json(json_file, chunksize=30000, lines=True):
        counter += 1
        df = chunk[['overall', 'reviewText']]
        df['sentiment'] = np.where(df['overall'] < 3, -1, np.where(df['overall'] > 4, 1, 0))
        df = df[df.sentiment != 0]
        df = df[['reviewText', 'sentiment']]
        if counter == num_chunks:
            break
    return df


# TODO
def generate_train_batch(df, batch_size):
    batch_j = 0
    batch_x = None
    batch_y = None

    while True:
        df = df.sample(frac=1)

        for j, row in df.iterrows():
            comment = row['reviewText']

            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

            batch_x[batch_j] = text_to_vector(comment)
            batch_y[batch_j] = row[classes].values
            batch_j += 1

            if batch_j == batch_size:
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_j = 0


    # for j in range(len(df)):
    #     text = df.iloc[[j]]
    #     words = text.split()
    #     words = np.asarray(words)
    #     words = np.reshape(words, [-1, ])
    #
    #     count = collections.Counter(words).most_common()
    #     for word, _ in count:
    #         dictionary[word] = len(dictionary)
    #     reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


# TODO
def get_validation_data():
    pass


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


def cleanSentences(string, remove_stopwords=False, stem_words=False):
    # remove punctuation, parentheses, question marks, etc.
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower()
    string = string.replace("<br />", " ")
    string = string.replace(r"(\().*(\))|([^a-zA-Z'])",' ')
    if remove_stopwords:
        stop_words = stopwords.words('english')
        if stem_words:
            ps = PorterStemmer()
            string = [' '.join([ps.stem(j.lower()) for j in w.split()\
                if j not in stop_words]) for w in string]
        else:
            string = [' '.join([j.lower() for j in w.split()\
                if j not in stop_words]) for w in string]
    # return re.sub(strip_special_chars, "", string.lower())
    return re.sub(strip_special_chars, "", string)


def getSentenceMatrix(sentence, batchSize, maxSeqLength, wordsList):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
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


def retrieve_sentiments():
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
    with engine.connect() as db_conn:
        df = pd.read_sql('SELECT r.id, r.title, r.genre, r.user, r.reviewDate, r.review, r.rating, s.class FROM reviews AS r INNER JOIN sentiments AS s ON r.id = s.id', con=db_conn)
        print('loaded dataframe from MySQL. records: ', len(df))
        db_conn.close()
    return df


if __name__ == '__main__':
    df = load_split_amzn_reviews()
