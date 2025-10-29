import json
import gzip

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import dateutil.parser

from sklearn import linear_model

def load_data(file_path):
    # Load data from a JSON file into a pandas DataFrame
    f = gzip.open(file_path)
    dataset = []
    for l in f:
        dataset.append(json.loads(l))
    return dataset

def Q1(dataset):
    # Star ratings = y
    data = {k : [] for k in dataset[0].keys()}
    for _ in range(len(dataset)):
        for k in dataset[_].keys():
            data[k].append(dataset[_][k])
    
    y = data["rating"]
    # Review's length = x

    x = np.array([len(desc) for desc in data['review_text']])

    # Normalize x between 0 and 1
    x = (x - np.min(x)) / (np.max(x) - np.min(x))

    # Linear regression
    A = np.vstack([np.ones(len(x)), x]).T
    theta = np.linalg.pinv(A .T @ A) @ A.T @ y
    # Predict y using the linear model
    y_pred = A @ theta
    # Compute MSE
    MSE = np.mean((y - y_pred) ** 2)
    return theta, MSE

def Q2(dataset, return_theta=False):
    data = {k : [] for k in dataset[0].keys()}
    for _ in range(len(dataset)):
        for k in dataset[_].keys():
            data[k].append(dataset[_][k])
    # See all columns for the first few rows of the dataset
    # for col in data.columns:
    #     print(data[col].head())
    # Star ratings = y
    encoder = OneHotEncoder(drop='first') # Not using redundant features!
    y = data['rating']
    # One feature
    review_text = np.array([len(desc) for desc in data['review_text']])
    # Normalize review_text between 0 and 1
    review_text = (review_text - np.min(review_text)) / (np.max(review_text) - np.min(review_text))
    # Convert dimensions
    review_text = review_text.reshape(-1, 1)
    # print(review_text.shape)
    # Time of the review as the next feature
    review_day = np.array([dateutil.parser.parse(desc).weekday() for desc in data['date_added']])
    # Use sklearn
    x_day = encoder.fit_transform(review_day.reshape(-1, 1)).toarray()
    # print(review_day[:5])
    # print(x_day[:5])
    
    review_month = np.array([dateutil.parser.parse(desc).month for desc in data['date_added']])
    x_month = encoder.fit_transform(review_month.reshape(-1, 1)).toarray()
    # print(review_month[:5])
    # print(x_month[:5])

    # Combine all features into a single feature matrix
    x = np.hstack([np.ones_like(review_text), review_text, x_day, x_month])
    print(x.shape)

    # Linear regression
    A = x
    theta = np.linalg.pinv(A .T @ A) @ A.T @ y
    # Predict y using the linear model
    y_pred = A @ theta
    # Compute MSE
    MSE = np.mean((y - y_pred) ** 2)
    if return_theta:
        return theta
    return x, y, MSE

def Q3(dataset, return_theta=False):
    data = {k : [] for k in dataset[0].keys()}
    for _ in range(len(dataset)):
        for k in dataset[_].keys():
            data[k].append(dataset[_][k])
    y = data['rating']
    # One feature
    review_text = np.array([len(desc) for desc in data['review_text']])
    # Normalize review_text between 0 and 1
    review_text = (review_text - np.min(review_text)) / (np.max(review_text) - np.min(review_text))
    # Convert dimensions
    review_text = review_text.reshape(-1, 1)
    # print(review_text.shape)
    # Time of the review as the next feature
    
    review_day = np.array([dateutil.parser.parse(desc).weekday() for desc in data['date_added']])
    x_day = review_day.reshape(-1, 1)
    # print(review_day[:5])
    # print(x_day[:5])
    
    review_month = np.array([dateutil.parser.parse(desc).month for desc in data['date_added']])
    x_month = review_month.reshape(-1, 1)
    # print(review_month[:5])
    # print(x_month[:5])

    # Combine all features into a single feature matrix
    x = np.hstack([np.ones_like(review_text), review_text, x_day, x_month])
    # print(x.shape)

    # Linear regression
    A = x
    theta = np.linalg.pinv(A .T @ A) @ A.T @ y
    # Predict y using the linear model
    y_pred = A @ theta
    # Compute MSE
    MSE = np.mean((y - y_pred) ** 2)
    if return_theta:
        return theta
    return x, y, MSE

def Q4(dataset):
    # Splitting the dataset into train and test sets:
    dataset_train = dataset[: len(dataset) // 2]
    dataset_test = dataset[len(dataset) // 2 :]

    theta_one_hot = Q2(dataset_train, return_theta=True)
    # Process the features and extract labels the same way for one-hot encoding
    x_one_hot, y_one_hot, _ = Q2(dataset_test)
    y_pred_one_hot = x_one_hot @ theta_one_hot
    mse_one_hot = np.mean((y_one_hot - y_pred_one_hot) ** 2)

    theta_direct = Q3(dataset_train, return_theta=True)
    # Process the features and extract labels the same way for direct features
    x_direct, y_direct, _ = Q3(dataset_test)
    y_pred_direct = x_direct @ theta_direct
    mse_direct = np.mean((y_direct - y_pred_direct) ** 2)

    return mse_one_hot, mse_direct

def Q5(dataset, feature_func):
    # Logistic regression on Beer dataset
    x = feature_func(dataset)
    # Binary classification: 1 if rating >= 4, else 0
    review_overall = [d['review/overall'] for d in dataset]
    y = np.array([1 if rating >= 4 else 0 for rating in review_overall])
    # Logistic regression using sklearn
    model = linear_model.LogisticRegression(class_weight="balanced")
    model.fit(x, y)
    y_pred = model.predict(x)
    # Finding confusion matrix and Balanced Error Rate (BER)
    TP = (np.sum((y == 1) & (y_pred == 1))) 
    TN = (np.sum((y == 0) & (y_pred == 0))) 
    FP = (np.sum((y == 0) & (y_pred == 1))) 
    FN = (np.sum((y == 1) & (y_pred == 0)))
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    BER = 1 - 0.5 * (TPR + TNR)
    return TP, TN, FP, FN, BER

def featureQ5(dataset):
    review_text = [d['review/text'] for d in dataset]

    x = np.array([len(desc) for desc in review_text]).reshape(-1, 1)
    # Normalize x between 0 and 1
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x

def Q6(dataset):
    # Features
    x = featureQ5(dataset)

    # Binary labels: 1 if rating >= 4, else 0
    y = np.array([1 if d['review/overall'] >= 4 else 0 for d in dataset])

    # Train logistic regression
    model = linear_model.LogisticRegression(class_weight="balanced", fit_intercept=True)
    model.fit(x, y)

    # Get predicted probabilities for the positive class
    probs = model.predict_proba(x)[:, 1]

    # Sort examples by descending probability
    sorted_indices = np.argsort(probs)[::-1]

    # Define K values
    K = [1, 100, 1000, 10000]
    precs = []

    for k in K:
        # Select top-k indices
        topk = sorted_indices[:k]
        y_topk = y[topk]

        # Precision@k = proportion of positives in top-k
        precision = np.sum(y_topk == 1) / k
        precs.append(precision)

    return precs

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def featureQ7(dataset):
    # Using two features: length of review text and beer ABV
    review_text = [d['review/text'] for d in dataset]
    x_text= normalize(np.array([len(desc) for desc in review_text]).reshape(-1, 1))
    beer_abv = [d['beer/ABV'] for d in dataset]
    x_beer_abv = normalize(np.array(beer_abv).reshape(-1, 1))
    x = np.hstack([x_text, x_beer_abv])

    return x

def Q7(dataset, feature_func):
    return Q5(dataset, feature_func)

if __name__ == '__main__':
    dataset_text = load_data("./datasets/fantasy_10000.json.gz")
    Q2(dataset_text)
    # f = open("./datasets/beer_50000.json")
    # datasetB = []
    # for l in f:
    #     datasetB.append(eval(l))
    # # print(Q5(datasetB, featureQ5))
    # print(Q7(datasetB, featureQ5))
