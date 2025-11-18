import os
import random
import gzip
import matplotlib.pyplot as plt  # <-- ADDED

from collections import defaultdict
from sklearn.neural_network import MLPClassifier  # <-- ADDED

prefix = "../datasets/assignment1/" # Change this to wherever you put the dataset
AVG_SIM_CUTOFF = 0.002
MAX_SIM_CUTOFF = 0.004
LEN_PROP_CUTOFF = 43

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
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break
    return return1

def improvedStrategy(mostPopular, totalRead):
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead * 0.725: break
    return return1

def evaluateStrategy(return1, readValid, notRead):
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

def computeFeatures(u, b, ratingsPerItem, ratingsPerUser):
    # Compute avgSim, maxSim, and popularity (len) features
    maxSim = 0
    avgSim = 0
    users_b = [x for x, _ in ratingsPerItem[b]]
    sims = []
    for i, r in ratingsPerUser[u]:
        if i == b:
            continue
        users_i = [x for x, _ in ratingsPerItem[i]]
        sim = Jaccard(set(users_i), set(users_b))
        sims.append(sim)
    if sims:
        maxSim = max(sims)
        avgSim = sum(sims) / len(sims)
    len_prop = len(ratingsPerItem[b])
    return avgSim, maxSim, len_prop

def jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
    avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
    if maxSim > MAX_SIM_CUTOFF or len_prop > LEN_PROP_CUTOFF or avgSim > AVG_SIM_CUTOFF:
        return 1
    return 0

def predictRead():
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

    readValid, notReadValid = generateValidation(allRatings, ratingsValid)
    assert len(readValid) == len(notReadValid)

    acc = 0
    total = len(readValid) + len(notReadValid)

    # Baseline threshold strategy accuracy
    for u, b in readValid:
        if jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
            acc += 1
    for u, b in notReadValid:
        if not jaccardThresh(u, b, ratingsPerItem, ratingsPerUser):
            acc += 1
    print(f"Validation accuracy (threshold heuristic): {acc / total}")


    evaluateStrategy_result = evaluateStrategy(
        improvedStrategy(mostPopular, totalRead), readValid, notReadValid)
    print(f"Validation accuracy of improved strategy: {evaluateStrategy_result}")

    X = []
    y = []

    pos_maxSim, neg_maxSim = [], []
    pos_avgSim, neg_avgSim = [], []
    pos_lenProp, neg_lenProp = [], []

    # Positive samples
    for (u, b) in readValid:
        avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
        X.append([avgSim, maxSim, len_prop])
        y.append(1)
        pos_maxSim.append(maxSim)
        pos_avgSim.append(avgSim)
        pos_lenProp.append(len_prop)

    # Negative samples
    for (u, b) in notReadValid:
        avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
        X.append([avgSim, maxSim, len_prop])
        y.append(0)
        neg_maxSim.append(maxSim)
        neg_avgSim.append(avgSim)
        neg_lenProp.append(len_prop)

    # Shuffle data
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)

    clf = MLPClassifier(hidden_layer_sizes=(16, 16, 16),
                        activation='relu',
                        max_iter=3000,)
    clf.fit(X, y)

    # Evaluate NN on the same validation pairs
    nn_correct = 0
    for (u, b) in readValid:
        avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
        pred = clf.predict([[avgSim, maxSim, len_prop]])[0]
        if pred == 1:
            nn_correct += 1
    for (u, b) in notReadValid:
        avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
        pred = clf.predict([[avgSim, maxSim, len_prop]])[0]
        if pred == 0:
            nn_correct += 1

    print(f"Validation accuracy (NN on features): {nn_correct / total}")

    # print("Computing feature distributions for visualization...")

    # # Plot all OVERLAID histograms in one figure
    # fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    # fig.suptitle("Overlaid Feature Distributions (Positive vs Negative)", fontsize=18)

    # # maxSim
    # axs[0].hist(pos_maxSim, bins=40, alpha=0.5, label="Positive")
    # axs[0].hist(neg_maxSim, bins=40, alpha=0.5, label="Negative")
    # axs[0].axvline(MAX_SIM_CUTOFF, color="red", linestyle="--", label=f"cutoff={MAX_SIM_CUTOFF}")
    # axs[0].set_title("maxSim Distribution")
    # axs[0].legend()

    # # avgSim
    # axs[1].hist(pos_avgSim, bins=40, alpha=0.5, label="Positive")
    # axs[1].hist(neg_avgSim, bins=40, alpha=0.5, label="Negative")
    # axs[1].axvline(AVG_SIM_CUTOFF, color="red", linestyle="--", label=f"cutoff={AVG_SIM_CUTOFF}")
    # axs[1].set_title("avgSim Distribution")
    # axs[1].legend()

    # # len_prop
    # axs[2].hist(pos_lenProp, bins=40, alpha=0.5, label="Positive")
    # axs[2].hist(neg_lenProp, bins=40, alpha=0.5, label="Negative")
    # axs[2].axvline(LEN_PROP_CUTOFF, color="red", linestyle="--", label=f"cutoff={LEN_PROP_CUTOFF}")
    # axs[2].set_title("len_prop Distribution")
    # axs[2].legend()

    # plt.tight_layout()
    # plt.show()

    ##################################################################
    # -------------------- END VISUALIZATION -------------------------
    ##################################################################

    # Removing previous file
    # if os.path.exists("predictions_Read.csv"):
    #     os.remove("predictions_Read.csv")
    
    # predictions = open("predictions_Read.csv", 'w')
    
    # for l in open(f"{prefix}pairs_Read.csv"):
    #     if l.startswith("userID"):
    #         predictions.write(l)
    #         continue
    #     u,b = l.strip().split(',')

    #     # Use NN to predict read/unread from features
    #     avgSim, maxSim, len_prop = computeFeatures(u, b, ratingsPerItem, ratingsPerUser)
    #     pred = clf.predict([[avgSim, maxSim, len_prop]])[0]

    #     predictions.write(u + ',' + b + f",{int(pred)}\n")

    # predictions.close()

if __name__ == "__main__":
    predictRead()