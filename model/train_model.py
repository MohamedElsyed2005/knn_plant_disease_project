# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib

# Load the pre-saved data and labels
features = np.load('../saved_data/features.npy')
labels = np.load('../saved_data/labels.npy')
label_encoder = joblib.load('../saved_data/label_encoder.joblib')

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

# Initialize StandardScaler (better for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Define the grid of parameters for GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 15)}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_knn = grid_search.best_estimator_

# Predict on train and test sets
y_train_pred = best_knn.predict(X_train_scaled)
y_test_pred = best_knn.predict(X_test_scaled)

# Accuracy
print(f'Train Accuracy: {accuracy_score(y_train, y_train_pred):.2f}')
print(f'Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}')

# Confusion Matrix Plot
for y_true, y_pred, title in zip([y_train, y_test], [y_train_pred, y_test_pred], ['Train', 'Test']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'{title} Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f'../results/confusion_matrix_{title.lower()}.png', dpi=300)
    plt.show()

# Classification Reports
train_report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

# Save Reports as JSON
with open('../results/classification_report_train.json', 'w') as json_file:
    json.dump(train_report, json_file)

with open('../results/classification_report_test.json', 'w') as json_file:
    json.dump(test_report, json_file)

print('Classification reports saved in JSON format.')