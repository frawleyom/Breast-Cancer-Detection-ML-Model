import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler


# Load the WDBC dataset from a local file
file_path = './data/wdbc.data'
data = pd.read_csv(file_path, header=None)

# Prompt user to input the number of features to include
num_features = int(input("Enter the number of features to include (10/20/30): "))
if num_features > 30 or num_features < 1:
    print("Invalid number of features. Please enter a value between 1 and 30.")
else:
    # Drop the ID column and select Diagnosis (column 1) as the target
    X = data.iloc[:, 2:2 + num_features].values  # Use the specified number of features
    y = data[1].values

    # Encode the target labels (M -> 0 for malignant, B -> 1 for benign)
    y = np.where(y == 'M', 0, 1)

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Perform 5-fold cross-validation to find the optimal k-value
    k_values = range(1, 31)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')
        cv_scores.append(scores.mean())

    # Determine the best k based on the highest average accuracy
    optimal_k = k_values[np.argmax(cv_scores)]
    print(f"The optimal k value is: {optimal_k} using {X.shape[1]} features")

    # Split data into 80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the K-NN classifier with the optimal k value on the training set
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)
    y_pred_proba = knn.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy with k={optimal_k} using {X.shape[1]} features: {accuracy:.2f}")

    # Print the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Print the classification report
    class_report = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])
    print("Classification Report:")
    print(class_report)
