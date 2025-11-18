import os
import random
import gzip

from collections import defaultdict

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

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

def improvedStrategy(mostPopular, totalRead):
    # Same as above function, just find an item set that'll have higher accuracy
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead * 0.726: break
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

    return correct / total if total > 0 else 0

def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular

    maxSim = 0
    users_b = [x for x, _ in ratingsPerItem[b]]
    for i, r in ratingsPerUser[u]:
        if i == b:
            continue
        users_i = [x for x, _ in ratingsPerItem[i]]
        sim = Jaccard(set(users_i), set(users_b))
        maxSim = max(maxSim, sim)

    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0

def predictRead(prefix = "../datasets/assignment1/"):
    allRatings = []
    userRatings = defaultdict(list)

    for user,book,r in readCSV(f"{prefix}train_Interactions.csv.gz"):
        allRatings.append((user,book,r))
        userRatings[user].append(r)

    train_split = 0.9
    ratingsTrain = allRatings[:int(train_split * len(allRatings))]
    ratingsValid = allRatings[int(train_split * len(allRatings)):]
    
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u,b,r in ratingsTrain:
        ratingsPerUser[u].append((b,r))
        ratingsPerItem[b].append((u,r))

    bookCount = defaultdict(int)
    totalRead = 0

    for user,book,_ in readCSV(f"{prefix}train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    # 50-50 split of validation set into read and not-read
    readValid, notReadValid = generateValidation(allRatings, ratingsValid)
    assert len(readValid) == len(notReadValid)
    print(f"Validation set: {len(readValid)} reads and {len(notReadValid)} not-reads")

    # acc = 0
    # total = len(readValid) + len(notReadValid)
    # print(f"Total validation instances: {total}")
    # for u, b in readValid:
    #     if jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    #         acc += 1
    # for u, b in notReadValid:
    #     if not jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    #         acc += 1
    # print(f"Validation accuracy: {acc / total}")

    return1 = improvedStrategy(mostPopular, totalRead)
    valAcc = evaluateStrategy(return1, readValid, notReadValid)
    print(f"Validation accuracy of improved strategy: {valAcc}")
  
    # Removing previous file
    if os.path.exists("predictions_Read.csv"):
        os.remove("predictions_Read.csv")
    
    predictions = open("predictions_Read.csv", 'w')
    
    for l in open(f"{prefix}pairs_Read.csv"):
        if l.startswith("userID"):
            #header
            predictions.write(l)
            continue
        u,b = l.strip().split(',')

        # pred = jaccardThresh(u, b, ratingsPerItem, ratingsPerUser)
        # predictions.write(u + ',' + b + f",{pred}\n")
        # Predict '1' if book is in the return1 set
        if b in return1:
            predictions.write(u + ',' + b + f",1\n")
        else:
            predictions.write(u + ',' + b + f",0\n")

    predictions.close()

if __name__ == "__main__":
    predictRead()