import numpy as np

from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder

def feat(d, catID, maxLength, one_hot_encoder, includeCat = True, includeReview = True, includeLength = True):
    feat = []
    if includeCat:
        # My implementation is modular such that this one function concatenates all three features together,
        # depending on which are selected
        # One-hot encodeing using the catID
        X = one_hot_encoder.fit_transform(np.array([d["beer/style"]]).reshape(-1, 1)).toarray()
        feat += list(X[0])
    if includeReview:
        review = np.array([d['review/aroma'], d['review/appearance'], d['review/palate'], d['review/taste'], d['review/overall']])
        feat += list(review)
    if includeLength:
        review_text = np.array(len(d['review/text']))
        # Normalize review_text between 0 and 1
        review_text = review_text / maxLength
        feat += [review_text]
    return feat + [1]

def Q1(catID, dataTrain, dataValid, dataTest):
    # Only using category IDs in catID
    includeCat = True
    includeReview = False
    includeLength = False
    # Inputs
    # Using catID only and reserving 0 vector for unknown categories
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest  = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest  = [d['beer/ABV'] > 7 for d in dataTest]
    # Training the model
    model = linear_model.LogisticRegression(C=10, class_weight='balanced', max_iter=1000)
    model.fit(XTrain, yTrain)
    # Validation
    validBER = np.mean(model.predict(XValid) != yValid)
    # Testing
    testBER = np.mean(model.predict(XTest) != yTest)
    return model, validBER, testBER

def Q2(catID, dataTrain, dataValid, dataTest):
    # Only using category IDs in catID
    includeCat = True
    includeReview = True
    includeLength = True
    # Inputs
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]
    # Training the model
    model = linear_model.LogisticRegression(C=10, class_weight='balanced')
    model.fit(XTrain, yTrain)
    # Validation BER
    validBER = np.mean(model.predict(XValid) != yValid)
    # Testing BER
    testBER = np.mean(model.predict(XTest) != yTest)
    return model, validBER, testBER

def Q3(catID, dataTrain, dataValid, dataTest):
    # Only using category IDs in catID
    includeCat = True
    includeReview = True
    includeLength = True
    # Using complete regularization

    # Inputs
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    bestBER = [1, 1]
    # Training the model
    for c_val in [0.001, 0.01, 0.1, 1, 10]:
        model = linear_model.LogisticRegression(C=c_val, class_weight='balanced')
        model.fit(XTrain, yTrain)
        # Validation
        validBER = np.mean(model.predict(XValid) != yValid)
        # Testing
        testBER = np.mean(model.predict(XTest) != yTest)

        if validBER < bestBER[0]:
            bestBER = [validBER, testBER]
            bestModel = model
    return bestModel, bestBER[0], bestBER[1]

def Q4(c_val, catID, dataTrain, dataValid, dataTest):
    # Ablation study!
    # Only using category IDs in catID
    includeCat = True
    includeReview = True
    includeLength = False
    # Using complete regularization

    # Inputs
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    # Training the model
    model = linear_model.LogisticRegression(C=c_val, class_weight='balanced')
    model.fit(XTrain, yTrain)
    # Validation
    validBER_noLen = np.mean(model.predict(XValid) != yValid)
    # Testing
    testBER_noLen = np.mean(model.predict(XTest) != yTest)
    
    # Only using category IDs in catID
    includeCat = True
    includeReview = False
    includeLength = True
    # Using complete regularization

    # Inputs
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    # Training the model
    model = linear_model.LogisticRegression(C=c_val, class_weight='balanced')
    model.fit(XTrain, yTrain)
    # Validation
    validBER_noReview = np.mean(model.predict(XValid) != yValid)
    # Testing
    testBER_noReview = np.mean(model.predict(XTest) != yTest)

    # Only using category IDs in catID
    includeCat = False
    includeReview = True
    includeLength = True
    # Using complete regularization

    # Inputs
    one_hot_encoder = OneHotEncoder(categories=[sorted(catID.keys())], handle_unknown='ignore')
    maxLength = max([len(d['review/text']) for d in dataTrain])
    
    XTrain = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTrain]
    XValid = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataValid]
    XTest = [feat(d, catID, maxLength, one_hot_encoder, includeCat, includeReview, includeLength) for d in dataTest]
    
    yTrain = [d['beer/ABV'] > 7 for d in dataTrain]
    yValid = [d['beer/ABV'] > 7 for d in dataValid]
    yTest = [d['beer/ABV'] > 7 for d in dataTest]

    # Training the model
    model = linear_model.LogisticRegression(C=c_val, class_weight='balanced')
    model.fit(XTrain, yTrain)
    # Validation
    validBER_noCat = np.mean(model.predict(XValid) != yValid)
    # Testing
    testBER_noCat = np.mean(model.predict(XTest) != yTest)

    return testBER_noCat, testBER_noReview, testBER_noLen

def mostSimilar(i, N, usersPerItem):
    # Implement...
    similarities = []
    for j in usersPerItem:
        if j != i:
            # Compute Jaccard similarity between items i and j
            sim = Jaccard(set(usersPerItem[i]), set(usersPerItem[j]))
            similarities.append((sim, j))
    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[0], reverse=True)
    # Should be a list of (similarity, itemID) pairs
    return similarities[:N]

def Jaccard(s1, s2):
    # Implement
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return intersection / union if union > 0 else 0

def MSE(y, ypred):
    # Implement...
    y = np.array(y)
    ypred = np.array(ypred)
    return np.mean((y - ypred) ** 2)

def getMeanRating(dataTrain):
    # Implement...
    ratingMean = np.mean([d['star_rating'] for d in dataTrain])
    return ratingMean

def getUserAverages(itemsPerUser, ratingDict):
    # Implement (should return a dictionary mapping users to their averages)
    userAverages = {}
    for user, items in itemsPerUser.items():
        ratings = [ratingDict[(user, item)] for item in items if (user, item) in ratingDict]
        if ratings:
            userAverages[user] = np.mean(ratings)
        else:
            userAverages[user] = 0  # or some default value
    return userAverages

def getItemAverages(usersPerItem, ratingDict):
    # Implement (should return a dictionary mapping items to their averages)
    itemAverages = {}
    for item, users in usersPerItem.items():
        ratings = [ratingDict[(user, item)] for user in users if (user, item) in ratingDict]
        if ratings:
            itemAverages[item] = np.mean(ratings)
        else:
            itemAverages[item] = 0  # or some default value
    return itemAverages

def predictRating(user, item, ratingMean, reviewsPerUser, usersPerItem, itemsPerUser, userAverages, itemAverages):
    # Solution for Q6, should return a rating
    rating = itemAverages.get(item, ratingMean)
    numerator = 0
    denominator = 0
    for other_item in itemsPerUser.get(user, []):
        if other_item in usersPerItem:
            sim = Jaccard(set(usersPerItem[item]), set(usersPerItem[other_item]))
            user_rating_for_other_item = 0
            for review in reviewsPerUser[user]:
                if review["product_id"] == other_item:
                    user_rating_for_other_item = review["star_rating"]
                    break
            numerator += sim * (user_rating_for_other_item - itemAverages[other_item])
            denominator += sim
    if denominator != 0:
        rating += numerator / denominator
    return rating

def predictRatingQ7(user, item, ratingMean, reviewsPerUser, usersPerItem, itemsPerUser, userAverages, itemAverages):
    # Get the number of ratings for the current item
    n_i = len(usersPerItem.get(item, []))
    C_reg = 20
    if n_i > 0:
        # Item has been rated in the training set
        avg_i = itemAverages[item]
        # Regularized Item Average: Blends item average (avg_i) with global mean (ratingMean)
        baseline_rating = (n_i * avg_i + C_reg * ratingMean) / (n_i + C_reg)
    else:
        # Item has NOT been rated in the training set (Use global mean)
        baseline_rating = ratingMean
        
    rating = baseline_rating

    users_of_item = set(usersPerItem.get(item, []))
    numerator = 0
    denominator = 0

    for other_item in itemsPerUser.get(user, []):
        if other_item == item:
            continue
        # Ensure the other_item is in the training set
        if other_item in usersPerItem:
            # Get the set of users who rated the other_item
            users_of_other_item = set(usersPerItem.get(other_item, []))
            # Calculate common users and regularized similarity
            common_users = len(users_of_item.intersection(users_of_other_item))
            raw_sim = Jaccard(users_of_item, users_of_other_item)
            # Regularize Jaccard similarity
            weight = common_users / (common_users + 20)
            sim_reg = raw_sim * weight
            # If the regularized similarity is near zero, skip the item
            if sim_reg < 1e-6:
                continue
            # Get the user's rating for the 'other_item'
            user_rating_for_other_item = 0
            for review in reviewsPerUser[user]:
                if review["product_id"] == other_item:
                    user_rating_for_other_item = review["star_rating"]
                    break
            adjusted_rating = user_rating_for_other_item - itemAverages[other_item]
            # Update numerator and denominator with the regularized similarity
            numerator += sim_reg * adjusted_rating
            denominator += sim_reg
    if denominator != 0:
        rating += numerator / denominator
        
    return rating