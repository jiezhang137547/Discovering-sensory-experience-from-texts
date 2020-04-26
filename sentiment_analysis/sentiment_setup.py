import pandas as pd
import json
import numpy as np
import nltk

nltk.download('vader_lexicon')

# load SentimentIntenseAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# instantiate new SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()



word_embedding_smell_place = pd.read_csv("tweets_smell_.csv")
# print(word_embedding_smell_place.head())



# word_embedding_smell_place= pd.read_csv("all_tweets_with_place.csv")   #来自extract_test_data.py
# id_LOC_368= pd.read_csv("id_LOC_368.csv")  #手动修改好了的 但是没有smell triggers
# word_embedding_smell_place = pd.merge(id_LOC_368,all_tweets_with_place,on="id",how="left")
# word_embedding_smell_place =word_embedding_smell_place[['id','LOC_x', 'index','processed_tweets','place','loc_pretrained']]
# print(len(word_embedding_smell_place))
# print(word_embedding_smell_place.columns)
# word_embedding_smell_place = pd.read_csv("word_embedding_df.csv")
# word_embedding_smell_place = word_embedding_smell_place.drop_duplicates(subset='id', keep="first")

# print(word_embedding_smell_place.dtypes)
# word_embedding_smell_place['processed_tweets'] = word_embedding_smell_place['processed_tweets'].astype(str)
# word_embedding_smell_place['id'] = word_embedding_smell_place['id'].astype(str)

# print(word_embedding_smell_place.dtypes)
# generate sentiment scores
sentiment_scores = word_embedding_smell_place.processed_tweets.apply(sid.polarity_scores)
sentiment_scores = [t['compound'] for t in sentiment_scores]

word_embedding_smell_place['sentiment_scores'] = sentiment_scores

word_embedding_smell_place.to_csv("word_embedding_smell_place_sentiment.csv", index=False)

word_embedding_smell_place_positive = word_embedding_smell_place[word_embedding_smell_place['sentiment_scores'] > 0]

word_embedding_smell_place_negative = word_embedding_smell_place[word_embedding_smell_place['sentiment_scores'] < 0]

print(word_embedding_smell_place_negative.head(10))

