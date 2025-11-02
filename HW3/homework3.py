import math
import scipy.optimize
import numpy as np
import string
import random
import gzip

from collections import defaultdict
from sklearn import linear_model

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

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

##################################################
# Rating prediction                              #
##################################################

def getGlobalAverage(trainRatings):
    # Return the average rating in the training set
    return np.mean(trainRatings)

def trivialValidMSE(ratingsValid, globalAverage):
    # Compute and return the MSE of a trivial model that always returns the global mean computed above
    ratingsValid = [r[2] for r in ratingsValid]
    return np.mean((np.array(ratingsValid) - globalAverage) ** 2)

# Our model is formalized as
# rating(u,i) = alpha + betaU(u) + betaI(i)

def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    newAlpha = np.mean([r - betaU[u] - betaI[i] for u, i, r in ratingsTrain])
    return newAlpha

def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    newBetaU = {}
    for u in ratingsPerUser:
        if u in betaU:
            newBetaU[u] = np.sum([r[1] - alpha - betaI[r[0]] for r in ratingsPerUser[u]]) / (len(ratingsPerUser[u]) + lamb)
    return newBetaU

def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    newBetaI = {}
    for i in ratingsPerItem:
        if i in betaI:
            newBetaI[i] = np.sum([r[1] - alpha - betaU[r[0]] for r in ratingsPerItem[i]]) / (len(ratingsPerItem[i]) + lamb)
    return newBetaI

def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    # Compute the MSE and the mse+regularization term
    mse = np.mean([(r[2] - alpha - betaU[r[0]] - betaI[r[1]]) ** 2 for r in ratingsTrain])
    regularizer = sum([b ** 2 for b in betaU.values()]) + sum([b ** 2 for b in betaI.values()])
    return mse, mse + lamb*regularizer

def validMSE(ratingsValid, alpha, betaU, betaI):
    # Compute the MSE on the validation set
    validMSE = np.mean([(r[2] - alpha - betaU.get(r[0],0) - betaI.get(r[1],0)) ** 2 for r in ratingsValid])
    return validMSE

def Iteratetrain(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, lamb, N):
    for i in range(N):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
        mse, mseReg = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)
    return alpha, betaU, betaI, mse, mseReg

def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    # Improve upon your model from the previous question (e.g. by running multiple iterations)
    alpha, betaU, betaI, mse, mseReg = Iteratetrain(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, 1.0, 10)
    return alpha, betaU, betaI

def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bu = 0
        bi = 0
        if u in betaU:
            bu = betaU[u]
        if b in betaI:
            bi = betaI[b]
        _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

    predictions.close()

##################################################
# Read prediction                                #
##################################################

def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    readValid = set()
    notRead = set()

    allItems = set([r[1] for r in allRatings])
    
    ratingsPerUser = defaultdict(set)
    ratingsPerItem = defaultdict(set)
    
    for r in allRatings:
        ratingsPerUser[r[0]].add(r[1])
        ratingsPerItem[r[1]].add(r[0])
    for r in ratingsValid:
        u, i = r[0], r[1]
        readValid.add((u, i))
        # Randomly add one item that user u has not read
        while(len(notRead) < len(readValid)):
            notRead.add((u, list(allItems - ratingsPerUser[u])[random.randint(0, len(allItems - ratingsPerUser[u]) - 1)]))
    return readValid, notRead

def baseLineStrategy(mostPopular, totalRead):
    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break
    return return1

def improvedStrategy(mostPopular, totalRead):
    # Same as above function, just find an item set that'll have higher accuracy
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead * 0.6: break
    return return1

def evaluateStrategy(return1, readValid, notRead):

    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label
    correct = 0
    total = 0
    for (u,i) in readValid:
        total += 1
        if i in return1:
            correct += 1

    for (u,i) in notRead:
        total += 1
        if i not in return1:
            correct += 1

    return correct/total if total > 0 else 0

def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular

    maxSim = 0
    users_b = [x for x, _ in ratingsPerItem[b]]
    for i, r in ratingsPerUser[u]:
        if i == b:
            continue
        users_i = [x for x, _ in ratingsPerItem[i]]
        sim = len(set(users_i) & set(users_b)) / len(set(users_i) | set(users_b))
        maxSim = max(maxSim, sim)

    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0

def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()

##################################################
# Category prediction                            #
##################################################

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

def betterFeatures(data):
    # Produce better features than those from the above question
    # Return matrix (each row is the feature vector for one entry in the dataset)
    wordCount = defaultdict(int)
    punctuation = set(string.punctuation)
    for d in data:
        r = ''.join([c for c in d['review_text'].lower() if not c in punctuation])
        for w in r.split():
            wordCount[w] += 1

    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    NW = 1000 # dictionary size
    
    words = [x[1] for x in counts[:NW]]
    wordId = {w: i for i, w in enumerate(words)}
    wordSet = set(words)

    X = [featureCat(d, words, wordId, wordSet) for d in data]
    return X

def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)

def writePredictionsCategory(pred_test):
    predictions = open("../datasets/assignment1/predictions_Category.csv", 'w')
    pos = 0

    for l in open("../datasets/assignment1/pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()