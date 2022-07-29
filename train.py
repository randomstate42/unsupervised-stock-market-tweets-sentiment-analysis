import multiprocessing
import pandas as pd
import numpy as np
import pickle

from gensim.models import Word2Vec
from sklearn.cluster._kmeans import KMeans



class UnsupervisedLabeler:
    def __init__(self):
        pass

    def train_word2vec(self, sentences, *args):
        w2v_model = Word2Vec(
            args,
            min_count=4,
            window=5,
            vector_size=300,
            sample=1e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            seed=42,
            workers=multiprocessing.cpu_count() - 1,
        )
        w2v_model.build_vocab(sentences, progress_per=50000)
        w2v_model.train(
            sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1
        )
        print("saving word2vec model")
        w2v_model.save("word2vec.model")

        return w2v_model


    def train_K_means(self, word_vectors, *args):
        # self._word_vectors = Word2Vec.load("word2vec.model").wv
        self.kmeans = KMeans(
            args, n_clusters=3, max_iter=1000, random_state=42, n_init=50
        ).fit(X=word_vectors.vectors.astype("double"))
        print("saving kmeans model")
        with open("kmeans_pkl", "wb") as f:
            pickle.dump(self.kmeans, f)


    def check_clusters(self):
        print(">>>>> 1st cluster word vectors")
        self._word_vectors.similar_by_vector(
            self.kmeans.cluster_centers_[0], topn=100, restrict_vocab=None
        )
        print(">>>>> 2nd cluster word vectors")
        self._word_vectors.similar_by_vector(
            self.kmeans.cluster_centers_[1], topn=100, restrict_vocab=None
        )
        print(">>>>> 3d cluster word vectors")
        self._word_vectors.similar_by_vector(
            self.kmeans.cluster_centers_[2], topn=100, restrict_vocab=None
        )


    def assign_labels(self, word_vectors, pos, neg):
        self._words = pd.DataFrame(word_vectors.index_to_key)
        self._words.columns = ["words"]
        self._words["vectors"] = self._words.apply(lambda x: self._word_vectors[f"{x}"])
        self._words["cluster"] = self._words.vectors.apply(
            lambda x: self.kmeans.predict([np.array(x)])
        )
        self._words.cluster = self._words.cluster.apply(lambda x: x[0])
        self._words["cluster_value"] = [
            1 if i == pos else -1 if i == neg else 0 for i in self._words.cluster
        ]
        self._words["closeness_score"] = self._words.apply(
            lambda x: 1 / (self.kmeans.transform([x.vectors]).min()), axis=1
        )
        positive = [
            "good",
            "great",
            "clean",
            "walk",
            "prosperity",
            "light",
            "superb",
            "amaze",
            "brilliant",
            "awesome",
            "win",
            "better",
        ]
        neutral = [
            "can",
            "go",
            "going",
            "got",
            "air",
            "climate",
            "in",
            "shall",
            "he",
            "happens",
            "grocery",
            "person",
            "storage",
            "space",
            "really",
            "time",
            "apartment",
        ]
        negative = [
            "pathetic",
            "stupid",
            "mad",
            "idiot",
            "insane",
            "sad",
            "tough",
            "annoy",
            "boo",
            "bad",
            "unhealthy",
            "weird",
            "brutal",
            "fail",
        ]
        for i in positive:
            self._words.loc[self._words["words"] == i, "cluster_value"] = 1
        for i in neutral:
            self._words.loc[self._words["words"] == i, "cluster_value"] = 0
        for i in negative:
            self._words.loc[self._words["words"] == i, "cluster_value"] = -1
        emotion = {0: "neutral", 1: "positive", -1: "negative"}
        self._words["sentiments"] = self._words["cluster_value"].map(emotion)

        return self._words


    def create_sentiments(self, words):
        def sentiment_callback(input_data, sent_dict):
            total = 0
            count = 0
            test = input_data["clean_tweet"]
            for t in test:
                if sent_dict.get(t):
                    total += int(sent_dict.get(t))
                count += 1
            avg = total / count
            sentiment = -1 if avg < -0.2 else 1 if avg > 0.2 else 0

            return sentiment

        words_dict = dict(zip(words.words, words.cluster_value))
        self.df["sentiment"] = self.df.apply(
            sentiment_callback, args=(words_dict,), axis=1
        )


def main():
    from preprocess import Preprocess

    processor = Preprocess("en", data_path="stock_tweets")
    processor.clean_tweet()
    processor.clean_hashtags()
    sentences = processor.get_n_bigrams()
    labeler = UnsupervisedLabeler()
    w2v_model = labeler.train_word2vec(sentences)
    word_vectors = w2v_model.wv
    labeler.train_K_means(word_vectors)
    labeler.check_clusters()
    print(">>> assigning clusters")
    pos = int(input("enter posistive cluster "))
    neg = int(input("enter negative cluster"))
    words_df = labeler.assign_labels(word_vectors, pos, neg)
    labeler.create_sentiments(words_df)


if __name__ == "__main__":
    main()
