import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys, os.path
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(\
    os.path.dirname(__file__), '..')) + '/learn/')
import utils


# import and format data
data = utils.retrieve_sentiments()
data = data[data['rating'].notnull()].copy()
data['rating'] = data['rating'].astype(float)
data = data[data['class'].notnull()].copy()
data['class'] = data['class'].astype(int)
data['word_count'] = data['review'].str.count(' ') + 1


def plot_bars():
    ratings = data.drop_duplicates(subset='title')
    ratings = ratings[['rating']]
    ratings = ratings.sort_values('rating')
    ratings['count'] = range(1, len(ratings) + 1)
    sns.barplot(x='count', y='rating', data=ratings)\
        .set_title('Rating Distribution')
    plt.xticks([])
    plt.xlabel('reviews')
    plt.show()

    # plot number of reviews by genre
    genres = data[['genre', 'rating']]
    sns.countplot(y='genre', data=genres)\
        .set_title('Number of Reviews by Genre')
    plt.show()

    # plot average rating grouped by genre
    genres.groupby('genre')['rating'].mean()
    sns.barplot(x='genre', y='rating', data=genres)\
        .set_title('Average Rating by Genre')
    plt.xticks(rotation=90)
    plt.show()

    # plot average ratings by genres and class
    genre_sentiments = data[['genre', 'rating', 'class']]
    g4 = sns.catplot(x='genre', col='class',
        data=genre_sentiments, kind='count')
    for ax in g4.axes.flatten():
        for tick in ax.get_xticklabels():
            tick.set(rotation=90)
    axes = g4.axes.flatten()
    axes[0].set_ylabel('Review Count by Genre and Sentiment')
    plt.show()

    # plot distribution of review length
    review_len = data[['word_count']]
    review_len = review_len.sort_values('word_count')
    review_len['count'] = range(1, len(review_len) + 1)
    sns.barplot(x='count', y='word_count', data=review_len)\
        .set_title('Distribution of Review Length')
    plt.xticks([])
    plt.xlabel('reviews')
    plt.ylabel('word length')
    plt.show()


def plot_boxes():
    word_len_rating = data[['word_count', 'rating']]
    sns.catplot(x='rating', y='word_count', data=word_len_rating, kind='box')
    plt.xticks(rotation=90)
    plt.title('Distribution of Review Length by Rating')
    plt.show()

def plot_hists():
    pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        plot_bars()
        plot_boxes()
        plot_hists()
    else:
        print(data.head())
