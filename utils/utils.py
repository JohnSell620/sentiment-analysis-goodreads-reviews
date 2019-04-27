import os
import re
import json
import string
import embeddings
import numpy as np
import pandas as pd
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

# import pydoop.hdfs as hdfs
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
        # df['sentiment'] = np.where(df['overall'] < 3, -1, np.where(df['overall'] > 4, 1, 0))
        df['sentiment'] = np.where(df['overall'] < 3, -1, np.where(df['overall'] > 4, 1, 0))
        df = df[df.sentiment != 0]
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


def generate_data_batch(batch_size=100, max_seq_length=150, embedding_size=300, train=True, training_split=0.8, word_embeddings=None):
    """
    Generate a random training batch of size batch_size.
    """
    df = load_split_amzn_reviews()
    train_test_split_idx = int(training_split * len(df))
    if train:
        df = df[:train_test_split_idx]
    else:
        df = df[train_test_split_idx:]
        batch_size = len(df)

    if word_embeddings is None:
        word_embeddings = embeddings.get_fastText_embedding()

    def batch_embedding_matrix(words, max_seq_length=max_seq_len):
        X = np.zeros((max_seq_length, word_embeddings.get_dimension()), dtype='float32')
        for j, word in enumerate(words[:max_seq_len]):
            X[j,:] = word_embeddings.get_word_vector(word).astype('float32')
        return X


    batch_j = 0
    batch_x = None
    batch_y = None

    while True:
        df = df.sample(frac=0.25)
        for j, row in df.itterows():
            review = row['reviewText']

            if batch_x is None:
                batch_x = np.zeros((batch_size, max_seq_length, embedding_size), dtype='float32')
                batch_y = np.zeros((batch_size, 2), dtype='float32')

            batch_x[batch_j] = batch_embedding_matrix(review.split())
            batch_y[batch_j] = literal_eval(row['sentiment'])
            batch_j += 1

            if batch_j == batch_size:
                return batch_x, batch_y
                # yield batch_x, batch_y
                # batch_x = None
                # batch_y = None
                # batch_j = 0


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


if __name__ == '__main__':
    # df = load_split_amzn_reviews()
    s = 'A test sentence to be normalized @ 1 time TODAY# <a> </ul><img>'
    s2 = """
    long ago , the mice had a general council to consider what measures
    they could take to outwit their common enemy , the cat . some said
    this , and some said that but at last a young mouse got up and said
    he had a proposal to make , which he thought would meet the case .
    you will all agree , said he , that our chief danger consists in the
    sly and treacherous manner in which the enemy approaches us . now ,
    if we could receive some signal of her approach , we could easily
    escape from her . i venture , therefore , to propose that a small
    bell be procured , and attached by a ribbon round the neck of the cat
    . by this means we should always know when she was about , and could
    easily retire while she was in the neighbourhood . this proposal met
    with general applause , until an old mouse got up and said that is
    all very well , but who is to bell the cat ? the mice looked at one
    another and nobody spoke . then the old mouse said it is easy to
    propose impossible remedies .
    """
    print(s2)
    # s2 = s2.replace('\n','')
    print(s2.split())
    sc = normalize_text(s2)
    print(sc)
