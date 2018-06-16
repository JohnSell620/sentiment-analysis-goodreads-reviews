## Overview
&nbsp;&nbsp;&nbsp; This project goes through the entire data mining process in an attempt to better understand book reviews on the Goodreads website. The typical book listed on the Goodreads website has one rating (an average) along with many reviews, and most of the featured books' ratings fall into a limited range, thus it is inherently difficult to understand the meaning of the rating. The goal here is to examine the sentiments of user reviews and book ratings across numerous genres.

&nbsp;&nbsp;&nbsp; This work examines these relationships as a NLP problem, namely, a document level sentiment classification problem. Sentiment predictions are made and then data visualization techniques are used to gain insight about the review-rating-genre relationship.

&nbsp;&nbsp;&nbsp; Three machine learning techniques are used in this project to obtain classifications. One classification is done using a pretrained RNN with long short term memory units (LSTMs) and with a pretrained Word2Vec model; both were pretrained by Adit Deshpande and may be found [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis). (TODO) In addition, the scikit-learn Multinomial Naive Bayes Classifier is used with CountVectorizer, and a C++ implementation of a Naive Bayes classifier by the [Text Mining Group, Nanjing University of Science & Technology,](https://github.com/NUSTM) is used for classifying.

&nbsp;&nbsp;&nbsp; The Word2Vec model was trained using the word vector generation model [GloVe](https://nlp.stanford.edu/projects/glove/). The word embedding matrix contains 400,000 word vectors with words having dimensionality of 50. The RNN was trained on the IMDb movie review dataset containing 12,500 positive and 12,500 negative reviews.

## Latest Results
&nbsp;&nbsp;&nbsp; The nodes in the graph are colored by genre. Their radii vary by the average rating of the title. Position in the y-direction are given by the rating multiplied by the sentiment (+1 or -1).

![D3.js](./results/class_by_id_1.png?raw=true "D3 Class * Rating vs ID")

## Dependencies
- web scraping: [Scrapy](https://scrapy.org) 1.4.0, [Selenium](https://www.seleniumhq.org/) (3.8.0), [PyMySQL](https://pymysql.readthedocs.io/en/latest/) 0.8.0.
- ML and computation: [Pandas](http://pandas.pydata.org) (0.22.0), [NumPy](http://www.numpy.org) (1.14.2), [SQLAlchemy](https://www.sqlalchemy.org/) (1.2.7).
- Dataviz: [D3.js](https://d3js.org/).

## Usage
1. Install dependencies:
```
$ python -m virtualenv goodreads
$ source goodreads/bin/activate
$ pip install -r requirements.txt
```

2. Create SQL table to store Goodreads review data:
```SQL
CREATE TABLE `reviews` (
 `id` int(11) NOT NULL AUTO_INCREMENT,
 `title` varchar(128) NOT NULL,
 `genre` varchar(255) NOT NULL,
 `link_url` varchar(255) NOT NULL,
 `book_url` varchar(255) NOT NULL,
 `user` varchar(32) NOT NULL,
 `reviewDate` varchar(32) NOT NULL,
 `review` text NOT NULL,
 `rating` varchar(24) NOT NULL,
 PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=502 DEFAULT CHARSET=latin1;
```

3. Run Scrapy web crawler:
```
$ scrapy crawl goodreads
```
In pipelines.py, you may add certain words to the words_to_filter array in the RequiredFieldsPipeline class to filter the reviews.

4. Choose classification algorithm to run:
..*scikit-learn Multinomial Naive Bayes classifier: `python multnb_sklearn.py`
..*XIA-NB C++ Naive Bayes Classifier: `python nb_xia.py`
..*RNN with LSTMs: `python rnn.py`

5. Visualize data:
..1. Start php server in goodreads/visualization directory: `php -S localhost:8000`. If you use `python -m http.server`, you will get the error "Failed to load http://localhost:8000/data.php: No 'Access-Control-Allow-Origin' header is present on the requested resource..."
..2. Open index.html in browser.

## Acknowledgements
1. Adit Deshpande's [article](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow) on [oreilly.com](www.oreilly.com).

2. The [Naive Bayes Classifier](https://github.com/NUSTM/XIA-NB) by the [Text Mining Group, Nanjing University of Science & Technology,](https://github.com/NUSTM).
