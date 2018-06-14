## Overview
&nbsp;This project goes through the entire data mining process in an attempt to better understand book reviews on the Goodreads website. The typical book listed on the Goodreads website has one rating (an average) along with many reviews, and most of the featured books' ratings fall in a limited range, so it is inherently difficult to understand the meaning of the rating. The goal is to examine the sentiments of user reviews and book ratings across numerous genres.

&nbsp;This work examines these relationships with machine learning techniques, namely, NLP - document level sentiment classification - and visualization of the results. The classification is done using a pretrained RNN with long short term memory units (LSTMs) and with a pretrained Word2Vec model; both were pretrained by Adit Deshpande and may be found [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis).

&nbsp;The Word2Vec model was trained using the word vector generation model [GloVe](https://nlp.stanford.edu/projects/glove/). The word embedding matrix contains 400,000 word vectors with words having dimensionality of 50. The RNN was trained on the IMDb movie review dataset containing 12,500 positive and 12,500 negative reviews.

## Dependencies
- web scraping: [Scrapy](https://scrapy.org) 1.4.0, [Selenium](https://www.seleniumhq.org/) (3.8.0), [PyMySQL](https://pymysql.readthedocs.io/en/latest/) 0.8.0.
- ML and computation: [Pandas](http://pandas.pydata.org) (0.22.0), [NumPy](http://www.numpy.org) (1.14.2), [SQLAlchemy](https://www.sqlalchemy.org/) (1.2.7).
-Dataviz: [D3.js](https://d3js.org/).

## Usage
I plan to add some scripts for installing dependencies, creating the database, running the web crawler, running the classifier, and generating the graphs.

In pipelines.py, you may add certain words to the words_to_filter array in the RequiredFieldsPipeline class to filter the reviews.

## Acknowledgements
Adit Deshpande's [article](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow) on [oreilly.com](www.oreilly.com).
