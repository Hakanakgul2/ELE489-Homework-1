import numpy as np
from collections import Counter

def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def knn_predict(X_train, Y_train, X_test, k=3, distance_metric='euclidean'):
    dist_func = euclidean if distance_metric == 'euclidean' else manhattan
    predictions = []

    for x in X_test:
        distances = [dist_func(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:k]
        k_labels = [Y_train.iloc[i] for i in k_indices]
        predictions.append(Counter(k_labels).most_common(1)[0][0])

    return np.array(predictions)
print("knn.py loaded successfully")
