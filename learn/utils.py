import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, MetaData, ForeignKey
import string


def retrieve_data():
    # connect to MySQL database and query data
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


def store_prediction(sentiment):
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
