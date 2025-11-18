import os
import random
import gzip
import numpy as np
from collections import defaultdict
import math

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

def cosine_similarity(s1, s2):
    """Cosine similarity between two sets"""
    intersection = len(s1.intersection(s2))
    if len(s1) == 0 or len(s2) == 0:
        return 0
    return intersection / math.sqrt(len(s1) * len(s2))

def generate_training_pairs(allRatings, ratingsPerUser, allItems):
    """Generate positive and negative training pairs using BPR-style sampling"""
    training_data = []
    
    for u, i, r in allRatings:
        # Positive pair: (u, i)
        # Negative pair: (u, j) where j is not in user's history
        
        user_items = set([item for item, _ in ratingsPerUser.get(u, [])])
        negative_items = list(allItems - user_items)
        
        if len(negative_items) > 0:
            # Sample one negative per positive
            j = random.choice(negative_items)
            training_data.append({
                'user': u,
                'pos_item': i,
                'neg_item': j
            })
    
    return training_data

def create_latent_factors(allRatings, ratingsPerUser, ratingsValid, allItems, 
                          n_factors=20, learning_rate=0.01, n_epochs=30, lambda_reg=0.01):
    """Train latent factors using BPR-style objective: sigmoid(gamma_u * gamma_i - gamma_u * gamma_j)"""
    
    # Initialize factors
    users = set([r[0] for r in allRatings])
    items = set([r[1] for r in allRatings])
    
    user_factors = {u: np.random.randn(n_factors) * 0.01 for u in users}
    item_factors = {i: np.random.randn(n_factors) * 0.01 for i in items}
    user_bias = {u: 0.0 for u in users}
    item_bias = {i: 0.0 for i in items}
    
    # Prepare validation pairs (reuse for all epochs)
    print("Preparing validation pairs...")
    val_pairs = []
    for u, i, r in ratingsValid:
        user_items = set([item for item, _ in ratingsPerUser.get(u, [])])
        neg_items = list(allItems - user_items)
        if len(neg_items) > 0:
            j = random.choice(neg_items)
            val_pairs.append({'user': u, 'pos_item': i, 'neg_item': j})
    
    print(f"Training latent factors with {n_factors} dimensions...")
    print(f"Training pairs per epoch: ~{len(allRatings)}, Validation pairs: {len(val_pairs)}")
    
    for epoch in range(n_epochs):
        # Generate training pairs
        training_pairs = generate_training_pairs(allRatings, ratingsPerUser, allItems)
        random.shuffle(training_pairs)
        
        total_loss = 0
        for pair in training_pairs:
            u = pair['user']
            i = pair['pos_item']
            j = pair['neg_item']
            
            # Get factors
            gamma_u = user_factors[u]
            gamma_i = item_factors[i]
            gamma_j = item_factors.get(j, np.zeros(n_factors))
            
            bias_u = user_bias[u]
            bias_i = item_bias[i]
            bias_j = item_bias.get(j, 0.0)
            
            # Score for positive and negative items
            # Following prof's formula: sigmoid(gamma_u * gamma_i - gamma_u * gamma_j)
            score_ui = np.dot(gamma_u, gamma_i) + bias_u + bias_i
            score_uj = np.dot(gamma_u, gamma_j) + bias_u + bias_j
            
            x_uij = score_ui - score_uj
            
            # BPR loss: -log(sigmoid(x_uij))
            sigmoid_x = 1 / (1 + np.exp(-x_uij))
            loss = -np.log(sigmoid_x + 1e-10)
            total_loss += loss
            
            # Gradient: (1 - sigmoid(x_uij))
            d_loss = sigmoid_x - 1
            
            # Update factors using gradient descent
            # d/d(gamma_u) = d_loss * (gamma_i - gamma_j) - lambda * gamma_u
            user_factors[u] -= learning_rate * (d_loss * (gamma_i - gamma_j) + lambda_reg * gamma_u)
            
            # d/d(gamma_i) = d_loss * gamma_u - lambda * gamma_i
            item_factors[i] -= learning_rate * (d_loss * gamma_u + lambda_reg * gamma_i)
            
            # d/d(gamma_j) = -d_loss * gamma_u - lambda * gamma_j
            item_factors[j] -= learning_rate * (-d_loss * gamma_u + lambda_reg * gamma_j)
            
            # Update biases
            user_bias[u] -= learning_rate * (d_loss + lambda_reg * bias_u)
            item_bias[i] -= learning_rate * (d_loss + lambda_reg * bias_i)
            item_bias[j] -= learning_rate * (-d_loss + lambda_reg * bias_j)
        
        # Compute validation loss and AUC every epoch
        val_loss = 0
        correct_rankings = 0
        
        for pair in val_pairs:
            u = pair['user']
            i = pair['pos_item']
            j = pair['neg_item']
            
            gamma_u = user_factors.get(u, np.zeros(n_factors))
            gamma_i = item_factors.get(i, np.zeros(n_factors))
            gamma_j = item_factors.get(j, np.zeros(n_factors))
            
            bias_u = user_bias.get(u, 0.0)
            bias_i = item_bias.get(i, 0.0)
            bias_j = item_bias.get(j, 0.0)
            
            score_ui = np.dot(gamma_u, gamma_i) + bias_u + bias_i
            score_uj = np.dot(gamma_u, gamma_j) + bias_u + bias_j
            
            x_uij = score_ui - score_uj
            
            # Validation loss
            sigmoid_x = 1 / (1 + np.exp(-x_uij))
            val_loss += -np.log(sigmoid_x + 1e-10)
            
            # AUC metric: correct if positive item scored higher than negative
            if score_ui > score_uj:
                correct_rankings += 1
        
        train_loss_avg = total_loss / len(training_pairs)
        val_loss_avg = val_loss / len(val_pairs)
        val_auc = correct_rankings / len(val_pairs)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val AUC: {val_auc:.4f}")
    
    return user_factors, item_factors, user_bias, item_bias

def compute_score(u, b, user_factors, item_factors, user_bias, item_bias, 
                  ratingsPerItem, ratingsPerUser, bookCount, use_heuristics=True):
    """Compute preference score for (user, book) pair"""
    
    # Get latent factor score
    n_factors = len(list(user_factors.values())[0])
    gamma_u = user_factors.get(u, np.zeros(n_factors))
    gamma_b = item_factors.get(b, np.zeros(n_factors))
    bias_u = user_bias.get(u, 0.0)
    bias_b = item_bias.get(b, 0.0)
    
    # Base latent factor score
    score = np.dot(gamma_u, gamma_b) + bias_u + bias_b
    
    if use_heuristics:
        # Add heuristic features as a boost
        # Jaccard similarity with user's history
        max_jaccard = 0
        users_b = set([x for x, _ in ratingsPerItem.get(b, [])])
        
        for i, r in ratingsPerUser.get(u, []):
            if i == b:
                continue
            users_i = set([x for x, _ in ratingsPerItem.get(i, [])])
            sim = Jaccard(users_i, users_b)
            max_jaccard = max(max_jaccard, sim)
        
        # Popularity boost
        popularity = bookCount.get(b, 0)
        
        # Combine: weight heuristics less than latent factors
        score += 0.5 * max_jaccard + 0.001 * math.log(popularity + 1)
    
    return score

def predictRead(prefix="../datasets/assignment1/"):
    allRatings = []
    userRatings = defaultdict(list)

    for user, book, r in readCSV(f"{prefix}train_Interactions.csv.gz"):
        allRatings.append((user, book, r))
        userRatings[user].append(r)

    # Split into train and validation
    train_split = 0.9
    ratingsTrain = allRatings[:int(train_split * len(allRatings))]
    ratingsValid = allRatings[int(train_split * len(allRatings)):]
    
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    bookCount = defaultdict(int)
    
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))
        bookCount[b] += 1

    allItems = set([r[1] for r in ratingsTrain])
    
    # Train latent factors on training data
    user_factors, item_factors, user_bias, item_bias = create_latent_factors(
        ratingsTrain, ratingsPerUser, ratingsValid, allItems,
        n_factors=20, learning_rate=0.01, n_epochs=10, lambda_reg=0.1
    )
    
    # Validate by computing scores on validation set
    print("\nValidating on held-out data...")
    scores_positive = []
    scores_negative = []
    
    for u, i, r in ratingsValid[:1000]:  # Sample for speed
        # Positive example score
        score_pos = compute_score(u, i, user_factors, item_factors, user_bias, item_bias,
                                  ratingsPerItem, ratingsPerUser, bookCount)
        scores_positive.append(score_pos)
        
        # Negative example (random unread book)
        user_items = set([item for item, _ in ratingsPerUser.get(u, [])])
        neg_items = list(allItems - user_items)
        if len(neg_items) > 0:
            j = random.choice(neg_items)
            score_neg = compute_score(u, j, user_factors, item_factors, user_bias, item_bias,
                                     ratingsPerItem, ratingsPerUser, bookCount)
            scores_negative.append(score_neg)
    
    print(f"Average score for positive items: {np.mean(scores_positive):.4f}")
    print(f"Average score for negative items: {np.mean(scores_negative):.4f}")
    
    # Find optimal threshold on validation set
    all_scores = []
    all_labels = []
    
    for u, i, r in ratingsValid:
        score = compute_score(u, i, user_factors, item_factors, user_bias, item_bias,
                             ratingsPerItem, ratingsPerUser, bookCount)
        all_scores.append(score)
        all_labels.append(1)
        
        # Negative sample
        user_items = set([item for item, _ in ratingsPerUser.get(u, [])])
        neg_items = list(allItems - user_items)
        if len(neg_items) > 0:
            j = random.choice(neg_items)
            score = compute_score(u, j, user_factors, item_factors, user_bias, item_bias,
                                 ratingsPerItem, ratingsPerUser, bookCount)
            all_scores.append(score)
            all_labels.append(0)
    
    # Find best threshold
    best_threshold = 0
    best_acc = 0
    for threshold in np.linspace(min(all_scores), max(all_scores), 100):
        predictions = [1 if s >= threshold else 0 for s in all_scores]
        acc = sum([p == l for p, l in zip(predictions, all_labels)]) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    
    print(f"Best validation accuracy: {best_acc:.4f} at threshold {best_threshold:.4f}")
    
    # Generate predictions for test set using the learned factors
    print("\nGenerating predictions...")
    if os.path.exists("predictions_Read.csv"):
        os.remove("predictions_Read.csv")
    
    predictions = open("predictions_Read.csv", 'w')
    
    # For per-user balancing, track predictions per user
    user_predictions = defaultdict(list)
    test_pairs = []
    
    for l in open(f"{prefix}pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u, b = l.strip().split(',')
        test_pairs.append((u, b))
        
        score = compute_score(u, b, user_factors, item_factors, user_bias, item_bias,
                             ratingsPerItem, ratingsPerUser, bookCount)
        user_predictions[u].append((score, b))
    
    # Per-user threshold: predict top 50% as read
    print("Applying per-user balancing (top 50% as read)...")
    final_predictions = {}
    
    for u in user_predictions:
        sorted_items = sorted(user_predictions[u], reverse=True)
        threshold_idx = len(sorted_items) // 2
        
        for idx, (score, b) in enumerate(sorted_items):
            if idx < threshold_idx:
                final_predictions[(u, b)] = 1
            else:
                final_predictions[(u, b)] = 0
    
    # Write predictions in order
    for u, b in test_pairs:
        pred = final_predictions.get((u, b), 0)
        predictions.write(u + ',' + b + f",{pred}\n")
    
    predictions.close()
    print("Predictions saved to predictions_Read.csv")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    predictRead()