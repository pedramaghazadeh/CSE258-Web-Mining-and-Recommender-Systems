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
from collections import defaultdict
from sklearn import linear_model

prefix = "../datasets/assignment1/" # Change this to wherever you put the dataset

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

# ****************
# Predict Category
# ****************

def featureCat(datum, words, wordId, wordSet):
    feat = [0] * len(words)
    punctuation = set(string.punctuation)
    # Compute features counting instance of each word in "words"
    # after converting to lower case and removing punctuation
    for w in datum['review_text'].lower().split():
        w = ''.join([c for c in w if not c in punctuation])
        if w in wordSet:
            feat[wordId[w]] += 1
    feat.append(1) # offset (put at the end)
    return feat

def buildVocab(data, NW=2000):
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in data:
        r = ''.join([c for c in d['review_text'].lower() if c not in punctuation])
        for w in r.split():
            wordCount[w] += 1

    counts = sorted(wordCount.items(), key=lambda x: -x[1])
    words = [w for w, _ in counts[:NW]]
    wordId = {w: i for i, w in enumerate(words)}
    wordSet = set(words)
    return words, wordId, wordSet

def betterFeatures(data, words, wordId, wordSet):
    X = [featureCat(d, words, wordId, wordSet) for d in data]
    return X

def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)

### Category prediction: look for keywords in the review text
def predictCat():
  data = []

  for d in readGz(f"{prefix}train_Category.json.gz"):
      data.append(d)
      
  # # Build vocab only once from training data
  # words, wordId, wordSet = buildVocab(data, NW=2000)

  # # Use same mapping for both train and test
  # X = betterFeatures(data, words, wordId, wordSet)
  # Xtrain = X[:9*len(X)//10]
  # Xvalid = X[9*len(X)//10:]

  # y = [d['genreID'] for d in data]
  # ytrain = y[:9*len(y)//10]
  # yvalid = y[9*len(y)//10:]

  
  # tfidf = TfidfVectorizer(
  #       lowercase=True,
  #       stop_words='english',
  #       max_features=20000,
  #       ngram_range=(1,2)
  #   )
  tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=30000,
    analyzer='char_wb',      # word-boundary char-grams
    ngram_range=(2, 5)       # 3–5 character windows
  )

  tfidf_word = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
  tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=10000)
  tfidf = FeatureUnion([('word', tfidf_word), ('char', tfidf_char)])

  # Fit only on training data to avoid leakage
  corpus_train = [d['review_text'] for d in data]
  X_all = tfidf.fit_transform(corpus_train)
  y_all = np.array([d['genreID'] for d in data])

  # Split into train/valid
  Xtrain, Xvalid, ytrain, yvalid = train_test_split(
      X_all, y_all, test_size=0.1, random_state=42, shuffle=True
  )

  data_test = list(readGz(f"{prefix}test_Category.json.gz"))
  # Xtest = betterFeatures(data_test, words, wordId, wordSet)
  corpus_test = [d['review_text'] for d in data_test]
  Xtest = tfidf.transform(corpus_test)

  # print(f"TF-IDF built with {len(tfidf.vocabulary_)} features.")

  mod = linear_model.LogisticRegression(
        C=20, penalty='l2', solver='liblinear', max_iter=2000, verbose=1
    )
  mod.fit(Xtrain, ytrain)

  pred_valid = mod.predict(Xvalid)
  correctB = yvalid == pred_valid
  correctB = sum(correctB) / len(correctB)
  print(f"Validation accuracy: {correctB}")

  pred_test = mod.predict(Xtest)

  # Removing previous file
  if os.path.exists("predictions_Category.csv"):
      os.remove("predictions_Category.csv")
  predictions = open("predictions_Category.csv", 'w')
  predictions.write("userID,reviewID,prediction\n")

  for ind, l in enumerate(readGz(f"{prefix}test_Category.json.gz")):
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(pred_test[ind]) + "\n")
  predictions.close()

if __name__ == "__main__":
    predictCat()