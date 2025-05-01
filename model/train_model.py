# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib

# Load the pre-saved data and labels
# features.npy contains the feature data, labels.npy contains the labels, and label_encoder.pkl is used for label encoding
features, labels = np.load(r"../src/features.npy"), np.load(r"../src/labels.npy")
label_encoder = joblib.load(r"../src/label_encoder.pkl")

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

# Initialize MinMaxScaler to scale the feature data (between 0 and 1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train data and transform
X_test_scaled = scaler.transform(X_test)  # Only transform the test data

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Define the grid of parameters for GridSearchCV
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11]}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_knn = grid_search.best_estimator_

# Predicting the test set results using the best KNN model
y_pred_best = best_knn.predict(X_test_scaled)
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")

# Confusion Matrix
y_pred = best_knn.predict(X_test)  # Predict on test set

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the Confusion Matrix using seaborn heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the plot to a file
plt.savefig("confusion_matrix_plot.png", dpi=300)  # You can change the file name and format (e.g., .jpg, .pdf)
plt.show()  # Show the plot in the output

# Classification Report (contains precision, recall, f1-score)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("Classification Report:\n", report)

# Save Confusion Matrix and Classification Report as JSON files

# Convert confusion matrix into a list for JSON serialization
cm_dict = {"confusion_matrix": cm.tolist()}

# Save confusion matrix to a JSON file
with open("confusion_matrix.json", "w") as json_file:
    json.dump(cm_dict, json_file)

# Convert classification report into a dictionary for JSON serialization (output_dict=True)
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

# Save classification report to a JSON file
with open("classification_report.json", "w") as json_file:
    json.dump(report_dict, json_file)

print("Data saved in JSON format.")