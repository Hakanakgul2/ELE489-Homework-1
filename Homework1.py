import numpy as np # Required for math operations
import pandas as pd # Important for reading data as can be seen from line 11
import matplotlib.pyplot as plt # Important for plotting graphs and figures
import seaborn as sns # From the given examples of hw1 , I found that the seaborn library makes plotting easier compared to matplotlib
from sklearn.model_selection import train_test_split # Allows us to split our data into test and train in ratio we adjust
from sklearn.preprocessing import StandardScaler # Scaling is important for calculating distance therefore I imported this function
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Allows us to plot confusion matrix and classification reports
import knn  # Importing only knn_predict function caused errors therefore I imported whole library
import os # Explained in further parts of the code
# Loading data by using pandas
wine_data = pd.read_csv("wine.data", header=None)
wine_data.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', #
                     'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',  # These features were given in wine.data file
                     'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']       #

X = wine_data.iloc[:, 1:] # I separated wine class and features into two parameters
Y = wine_data.iloc[:, 0]

# Scaling data with mean 0 variance 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2, stratify=Y) # Splitting data into 0.2 test , 0.8 train

# By using knn library I created I imported knn_predict function
k_values = range(1, 31, 2) # I choose k from 1 to 29 with increasing rate of 2 for odd value of k
accuracy_results = {'euclidean': [], 'manhattan': []}

for metric_type in ['euclidean', 'manhattan']:
    print(f"\nDistance metric: {metric_type}")
    for k in k_values:
        y_pred = knn.knn_predict(X_train, y_train, X_test, k=k, distance_metric=metric_type)
        acc = accuracy_score(y_test, y_pred) # For finding accuracy score I used built-in function of "sklearn.metrics" library
        accuracy_results[metric_type].append(acc)
        print(f"k = {k}, Accuracy = {acc:0.4f}")

# Plotting Accuracy vs K
plt.figure(figsize=(8, 5))
for metric in accuracy_results:
    plt.plot(k_values, accuracy_results[metric], marker='o', label=f'{metric.capitalize()}')


features = ['Alcohol', 'Color intensity', 'Alcalinity of ash', 'Hue', 'Total phenols']
sns.pairplot(wine_data[features + ['Class']], hue='Class', palette=['green', 'blue', 'red'])
plt.show()

plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs K for Different Distance Metrics')
plt.legend() # I added legend similar to matlab
plt.grid(True)
plt.show()

#def plot_confusion_matrices(k_values, X_train, y_train, X_test, y_test):
#    os.makedirs("outputs", exist_ok=True) # Create directory if not exists
#
#    for distance in ['euclidean', 'manhattan']:
#        for k in k_values:
#            y_pred = knn.knn_predict(X_train, y_train, X_test, k=k, distance_metric=distance)
#
#            cm = confusion_matrix(y_test, y_pred)
#            report = classification_report(y_test, y_pred, digits=2)
#
#            # --- Start plotting ---
#            plt.figure(figsize=(6, 7))
#
#            # Confusion matrix heatmap
#            plt.subplot(2, 1, 1)
#            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
#            plt.title(f'Confusion Matrix (k={k}, {distance.capitalize()})')
#           plt.xlabel('Predicted Label')
#            plt.ylabel('True Label')
#
#            # Classification report as text
#            plt.subplot(2, 1, 2)
#            plt.axis('off')
#            plt.text(0.01, 0.5, report, fontsize=10, family='monospace')
#
#            # Save combined figure
#            filename = f"outputs/confusion_report_k{k}_{distance}.png"
#            plt.savefig(filename)
#            plt.close()


def show_confusion_and_classification(k_values, X_train, y_train, X_test, y_test):
    for distance in ['euclidean', 'manhattan']:
        for k in k_values:
            y_pred = knn.knn_predict(X_train, y_train, X_test, k=k, distance_metric=distance)

            # Plotting Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix (k={k}, {distance.capitalize()})')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()

            # Printing Classification Report
            print(f"\nClassification Report (k={k}, {distance.capitalize()}):")
            print(classification_report(y_test, y_pred, digits=2))

show_confusion_and_classification(k_values, X_train, y_train, X_test, y_test)


