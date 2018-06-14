import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sqlalchemy import create_engine
import string


# connect to MySQL database and query data
engine = create_engine('mysql+pymysql://root:root@localhost:3306/goodreads')
db_conn = engine.connect()
df = pd.read_sql('SELECT * FROM reviews', con=db_conn)
print('loaded dataframe from MySQL. records: ', len(df))
db_conn.close()

# convert rating string to float
df.rating = df.rating.astype(float).fillna(0.0)

# remove NaN ratings and null reviews
df = df[np.isfinite(df['rating'])]
df = df[pd.notnull(df['review'])]

# create discrete values
bins = 10
min_rating = df['rating'].min()
span = df['rating'].max() - min_rating
step = span/bins
bin = np.empty(bins, dtype=float)
for i in range(bins):
    bin[i] = min_rating + (i+1)*step

bin = np.insert(bin, 0, min_rating-0.01)
strings = ["%.3f" % number for number in bin[1:]]

category = pd.cut(df.rating,bin,labels=strings)
category = category.to_frame()
category.columns = ['range']
df_new = pd.concat([df,category],axis=1)

# drop reviews without letters (words)
alphabet = list(string.ascii_letters)
df_new = df_new[(pd.concat([df_new.review.str.contains(word,regex=False) for word in alphabet],axis=1)).sum(1) > 0]

# features matrix and response vector
X = df_new.review
y = df_new.range

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# to convert text to token counts
vect = CountVectorizer()

# transform training and testing data into document-term matrices
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# examine the vocabulary and document-term matrix together
pd.DataFrame(X_train_dtm.toarray(), columns=vect.get_feature_names())
pd.DataFrame(X_test_dtm.toarray(), columns=vect.get_feature_names())

# Multinomial Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

print(metrics.accuracy_score(y_test, y_pred_class))
