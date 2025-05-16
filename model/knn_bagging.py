import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# Load your data
features = np.load('../saved_data/features.npy')
labels = np.load('../saved_data/labels.npy')

# Assuming label_encoder is loaded like this
label_encoder = joblib.load('../saved_data/label_encoder.joblib')

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
bag_clf = BaggingClassifier(
    KNeighborsClassifier(n_neighbors=1),
    n_estimators=100,       
    n_jobs=1,               
    bootstrap=True,
    oob_score=True,
    max_samples=0.5,
    max_features=0.5,
    random_state=42
)


# Train models
bag_clf.fit(X_train_scaled, y_train)

# Predict on train and test sets
y_train_pred = bag_clf.predict(X_train_scaled)
y_test_pred = bag_clf.predict(X_test_scaled)

# Accuracy
print(f'Bagging Train Accuracy: {accuracy_score(y_train, y_train_pred):.2f}')
print(f'Bagging Test Accuracy: {accuracy_score(y_test, y_test_pred):.2f}')

# Plot confusion matrix for Bagging
cm_bag = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm_bag, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Bagging Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('../results/confusion_matrix_Bagging.png', dpi=300)
plt.show()

# Classification Reports
bag_train_report = classification_report(y_train, y_train_pred, target_names=label_encoder.classes_, output_dict=True)
bag_test_report = classification_report(y_test, y_test_pred, target_names=label_encoder.classes_, output_dict=True)

# Save Reports as JSON
with open('../results/classification_report_train_bagging.json', 'w') as json_file:
    json.dump(bag_train_report, json_file)

with open('../results/classification_report_test_bagging.json', 'w') as json_file:
    json.dump(bag_test_report, json_file)

