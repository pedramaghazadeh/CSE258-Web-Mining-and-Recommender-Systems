import os
import math
import scipy.optimize
import numpy as np
import string
import random
import gzip

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from collections import defaultdict
from sklearn import linear_model

# -----------------------------
# NEW: NLP preprocessing tools
# -----------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re

# Ensure NLTK resources exist
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

# --------------------------------------
# CONFIGURABLE PREPROCESSING PIPELINE
# --------------------------------------
PREPROCESSING_OPTIONS = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
    "apply_stemming": False,
    "apply_lemmatization": False,

    # TF-IDF parameters
    "ngram_range": (1, 3),
    "max_features": None,      # None = unlimited
    "sublinear_tf": True,
    "min_df": 3,               # remove rare words
    "max_df": 0.85             # remove overly common words
}

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Apply preprocessing based on enabled options."""
    opts = PREPROCESSING_OPTIONS

    if opts["lowercase"]:
        text = text.lower()

    if opts["remove_punctuation"]:
        text = re.sub(r"[^\w\s]", " ", text)

    tokens = text.split()

    if opts["remove_stopwords"]:
        tokens = [t for t in tokens if t not in stop_words]

    if opts["apply_stemming"]:
        tokens = [stemmer.stem(t) for t in tokens]

    if opts["apply_lemmatization"]:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# =================================================================
# CATEGORY PREDICTION
# =================================================================
def predictCat(prefix = "../datasets/assignment1/"):

    # -------------------------
    # Load training data
    # -------------------------
    data = [d for d in readGz(f"{prefix}train_Category.json.gz")]

    # -------------------------
    # Preprocess corpus
    # -------------------------
    corpus_train = [preprocess_text(d['review_text']) for d in data]
    y_all = np.array([d['genreID'] for d in data])

    # -------------------------
    # TF-IDF with extended options
    # -------------------------
    opts = PREPROCESSING_OPTIONS

    tfidf_word = TfidfVectorizer(
        lowercase=False,                    # Already handled manually
        stop_words=None,                    # Already handled manually
        ngram_range=opts["ngram_range"],
        max_features=opts["max_features"],
        sublinear_tf=opts["sublinear_tf"],
        min_df=opts["min_df"],
        max_df=opts["max_df"]
    )

    tfidf_char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=opts["max_features"]
    )

    tfidf = FeatureUnion([
        ('word', tfidf_word),
        ('char', tfidf_char)
    ])

    X_all = tfidf.fit_transform(corpus_train)
    X_all = Normalizer(norm="l2").fit_transform(X_all)

    # -------------------------
    # Train/Validation split
    # -------------------------
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, shuffle=True
    )

    # -------------------------
    # Preprocess test data
    # -------------------------
    data_test = list(readGz(f"{prefix}test_Category.json.gz"))
    corpus_test = [preprocess_text(d['review_text']) for d in data_test]
    Xtest = tfidf.transform(corpus_test)

    # -------------------------
    # Logistic Regression
    # -------------------------
    mod = linear_model.LogisticRegression(
        C=1.2,
        penalty='l2',
        solver='liblinear',
        max_iter=30000,
        verbose=1,
        class_weight='balanced',
        n_jobs=-1
    )

    mod.fit(Xtrain, ytrain)

    pred_valid = mod.predict(Xvalid)
    valid_acc = np.mean(pred_valid == yvalid)
    print(f"Validation Accuracy: {valid_acc:.4f}")

    pred_test = mod.predict(Xtest)

    # -------------------------
    # Output Predictions
    # -------------------------
    # if os.path.exists("predictions_Category.csv"):
    #     os.remove("predictions_Category.csv")

    # predictions = open("predictions_Category.csv", 'w')
    # predictions.write("userID,reviewID,prediction\n")

    # for ind, l in enumerate(data_test):
    #     predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(pred_test[ind]) + "\n")
    # predictions.close()

    # print("Done. Output saved to predictions_Category.csv")


if __name__ == "__main__":
    predictCat()