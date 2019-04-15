import os
import re
import json
import string
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, MetaData, ForeignKey

# import pydoop.hdfs as hdfs
import hdfs


def load_amzn_reviews_json():
    json_file = os.path.join(os.getcwd(), '../data/', 'reviews_Books_5.json')
    with open(json_file) as data_file:
        data_dict = json.load(data_file)
    data = pd.DataFrame.from_dict(data_dict, orient='index')
    data.reset_index(level=0, inplace=True)
    return data


def retrieve_data_hdfs():
    with hdfs.open_file('/usr/local/hadoop/hadoopdata/hdfs/datanode/goodreads.csv', 'r') as f:
        df = pd.read_csv(f)
    return df


def retrieve_data_mysql():
    # connect to MySQL database and query data
    engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
    with engine.connect() as db_conn:
        df = pd.read_sql('SELECT * FROM reviews', con=db_conn)
        print('loaded dataframe from MySQL. records: ', len(df))
        db_conn.close()
    return df


def retrieve_data():
    # df = retrieve_data_hdfs()
    df = retrieve_data_mysql()

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


def cleanSentences(string):
    # remove punctuation, parentheses, question marks, etc.
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


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
    # df = load_amzn_reviews_json()
    json_file = os.path.join(os.getcwd(), '../data/', 'reviews_Books_5.json')
    df = pd.read_json(json_file)
