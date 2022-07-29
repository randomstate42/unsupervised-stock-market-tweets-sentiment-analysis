from typing import List
import pandas as pd
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser


class Preprocess:
    def __init__(self, *args, data_path) -> None:
        self.df = pd.read_csv(data_path)
        self.df = self.df[self.df["language"].isin(list(args))]
        if self.df["tweet"].duplicated(keep="first").sum() > 0:
            self.df = self.df.drop_duplicates(subset="tweet", keep="first")
        nltk.download("stopwords")


    def clean_tweet(self):
        def clean_callback(tweet):
            lemma = WordNetLemmatizer()
            stop_words = stopwords.words("english")
            tweet = tweet.lower()
            tweet = re.sub("https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*", " ", tweet)
            tweet = re.sub("\$[a-zA-Z0-9]*", " ", tweet)
            tweet = re.sub("\@[a-zA-Z0-9]*", " ", tweet)
            tweet = re.sub("[^a-zA-Z']", " ", tweet)
            tweet = " ".join([w for w in tweet.split() if len(w) > 1])

            tweet = " ".join(
                [
                    lemma.lemmatize(x)
                    for x in nltk.wordpunct_tokenize(tweet)
                    if x not in stop_words
                ]
            )
            tweet = [
                lemma.lemmatize(x, nltk.corpus.reader.wordnet.VERB)
                for x in nltk.wordpunct_tokenize(tweet)
                if x not in stop_words
            ]
            return tweet

        self.df["clean_tweet"] = self.df["tweet"].apply(lambda x: clean_callback(x))
        self.df["clean_tweet"] = self.df["clean_tweet"].apply(lambda x: " ".join(x))


    def clean_hashtags(self):
        def hashtags_callback(hashtags):
            if hashtags:
                hashtags = hashtags.lower()
                hashtags = re.sub("\$[a-zA-Z0-9]*", " ", hashtags)
                hashtags = re.sub("[^a-zA-Z]", " ", hashtags)
                hashtags = hashtags.strip()
            return hashtags

        self.df["hashtags"] = self.df["hashtags"].astype(str)
        self.df["hashtags"] = self.df["hashtags"].apply(lambda x: hashtags_callback(x))


    def convert_datetime(self):
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["month"] = self.df["date"].dt.month
        self.df["year"] = self.df["date"].dt.year


    def get_n_bigrams(self):
        sent = [row for row in self.df["clean_tweet"]]
        phrases = Phrases(sent, min_count=1, progress_per=50000)
        bigram = Phraser(phrases)
        sentences = bigram[sent]

        return sentences
