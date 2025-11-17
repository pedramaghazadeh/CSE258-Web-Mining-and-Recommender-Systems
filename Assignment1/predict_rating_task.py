import os
import math
import scipy.optimize
import numpy as np
import string
import random
import gzip

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
    yield l.strip().split(',')

# *************
# Predict Rating
# *************
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

def Iteratetrain(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, lamb, N, ratingsValid):
  best_vali_mse = float('inf')
  for i in range(N):
    alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
    betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
    betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
    mse, mseReg = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)

    vali_mse = validMSE(ratingsValid, alpha, betaU, betaI)
    print(f"Iteration {i+1}: Train MSE: {mse}, Train MSE+Reg: {mseReg}, Valid MSE: {vali_mse}")
    if vali_mse < best_vali_mse:
        best_vali_mse = vali_mse
        print(f"  New best model found at iteration {i+1} with Valid MSE: {vali_mse}")
  return alpha, betaU, betaI, mse, mseReg

def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, ratingsValid):
  # Improve upon your model from the previous question (e.g. by running multiple iterations)
  alpha, betaU, betaI, mse, mseReg = Iteratetrain(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, 5.0, 220, ratingsValid)
  return alpha, betaU, betaI

def predictRating():
  allRatings = []
  userRatings = defaultdict(list)

  for user,book,r in readCSV(f"{prefix}train_Interactions.csv.gz"):
    allRatings.append((user, book, float(r)))
    userRatings[user].append(float(r))

  ratingsTrain = allRatings[:190000]
  ratingsValid = allRatings[190000:]
  ratingsPerUser = defaultdict(list)
  ratingsPerItem = defaultdict(list)
  for u,b,r in ratingsTrain:
      ratingsPerUser[u].append((b,r))
      ratingsPerItem[b].append((u,r))

  trainRatings = [r[2] for r in ratingsTrain]
  globalAverage = getGlobalAverage(trainRatings)

  userAverage = {}
  for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

  betaU = {}
  betaI = {}
  for u in ratingsPerUser:
      betaU[u] = 0

  for b in ratingsPerItem:
      betaI[b] = 0

  alpha = globalAverage # Could initialize anywhere, this is a guess
  alpha, betaU, betaI = goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI, ratingsValid)

  # Removing previous file
  if os.path.exists("predictions_Rating.csv"):
     os.remove("predictions_Rating.csv")
  predictions = open("predictions_Rating.csv", 'w')
  for l in open(f"{prefix}pairs_Rating.csv"):
    if l.startswith("userID"):
      #header
      predictions.write(l)
      continue
    u,b = l.strip().split(',')
    if u in betaU and b in betaI:
      pred = alpha + betaU[u] + betaI[b]
      if pred < 0: pred = 0
      if pred > 5: pred = 5
      predictions.write(u + ',' + b + ',' + str(pred) + '\n')
    else:
      if u in userAverage:
        predictions.write(u + ',' + b + ',' + str(userAverage[u]) + '\n')
      else:
        predictions.write(u + ',' + b + ',' + str(globalAverage) + '\n')

  predictions.close()

if __name__ == "__main__":
    predictRating()