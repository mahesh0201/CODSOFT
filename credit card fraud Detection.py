import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

# Load the training data
df_train = pd.read_csv(r'/content/fraudTrain.csv')
df_train.head()
df_train.info()

# Load the test data
df_test = pd.read_csv(r'/content/fraudTest.csv')
df_test.head()
df_test.info()

# Combine the training and test data
df_combined = pd.concat([df_train, df_test], axis=0)
df_combined.head()

# Check for duplicate rows in the DataFrame
duplicate_rows = df_combined[df_combined.duplicated()]

# Display the duplicate rows
print("Duplicate Rows:")
print(duplicate_rows)

# Countplot for gender distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='gender', data=df_combined)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Countplot for class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='is_fraud', data=df_combined)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Not Fraud', 'Fraud'])  # Fixed class labels
plt.show()

# Combined countplot for gender and class distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='gender', hue='is_fraud', data=df_combined)
plt.title('Gender and Class Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Class', labels=['Not Fraud', 'Fraud'])  # Fixed class labels
plt.show()

# Scatter plot of latitude and longitude with color-coded classes
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_combined, x='long', y='lat', hue='is_fraud', palette='Set1', alpha=0.5)
plt.title('Scatter Plot of Latitude and Longitude')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Class', labels=['Not Fraud', 'Fraud'])  # Fixed class labels
plt.show()

numeric_features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

# Remove duplicate values in the 'feature' column
for feature in numeric_features:
    df_combined = df_combined.drop_duplicates(subset=[feature])

for feature in numeric_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df_combined, x=feature, hue='is_fraud', element='step', bins=20, common_norm=False)
    plt.title(f'Histogram of {feature} by Class')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.legend(title='Class', labels=['Not Fraud', 'Fraud'])
    plt.show()

# Continue with the rest of your code for modeling and evaluation

# Define X and y
X = df_combined.drop('is_fraud', axis=1)
y = df_combined['is_fraud']

# Encode categorical columns
columns_to_encode = ['trans_date_trans_time', 'merchant', 'category', 'first', 'last', 'gender', 'state', 'job', 'dob', 'street', 'city', 'trans_num']
encoder = OrdinalEncoder()
X[columns_to_encode] = encoder.fit_transform(X[columns_to_encode])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply under-sampling using NearMiss
nm = NearMiss()
X_resampled, y_resampled = nm.fit_resample(X_train, y_train)

# Create and train a Logistic Regression classifier
logreg_classifier = LogisticRegression()
logreg_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set
logreg_predictions = logreg_classifier.predict(X_test)

# Calculate and print training and testing accuracies for Logistic Regression
logreg_train_accuracy = accuracy_score(y_resampled, logreg_classifier.predict(X_resampled))
logreg_test_accuracy = accuracy_score(y_test, logreg_predictions)
print("Logistic regression model:")
print(f"Training Accuracy: {logreg_train_accuracy:.2f}")
print(f"Testing Accuracy: {logreg_test_accuracy:.2f}")

# Classification report for Logistic Regression
logreg_report = classification_report(y_test, logreg_predictions)
print("\nClassification Report for Logistic Regression:")
print(logreg_report)

# Plot confusion matrix for Logistic Regression
sns.heatmap(confusion_matrix(y_test, logreg_predictions), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Create and train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_resampled, y_resampled)

# Make predictions on the test set with Random Forest
rf_predictions = rf_classifier.predict(X_test)

# Calculate and print training and testing accuracies for Random Forest
rf_train_accuracy = accuracy_score(y_resampled, rf_classifier.predict(X_resampled))
rf_test_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Training Accuracy for Random Forest: {rf_train_accuracy:.2f}")
print(f"Testing Accuracy for Random Forest: {rf_test_accuracy:.2f}")

# Classification report for Random Forest
rf_report = classification_report(y_test, rf_predictions)
print("\nClassification Report for Random Forest:")
print(rf_report)

# Plot confusion matrix for Random Forest
sns.heatmap(confusion_matrix(y_test, rf_predictions), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Calculate ROC curve and AUC for Logistic Regression
logreg_probabilities = logreg_classifier.predict_proba(X_test)[:, 1]
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg_probabilities)
roc_auc_logreg = roc_auc_score(y_test, logreg_probabilities)

# Calculate ROC curve and AUC for Random Forest
rf_probabilities = rf_classifier.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probabilities)
roc_auc_rf = roc_auc_score(y_test, rf_probabilities)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='blue', label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', label=f'Random Forest (AUC = {roc_auc_rf:.2f}')
plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Calculate precision-recall curve for Logistic Regression
logreg_precision, logreg_recall, _ = precision_recall_curve(y_test, logreg_probabilities)

# Calculate precision-recall curve for Random Forest
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probabilities)

# Plot precision-recall curves
plt.figure(figsize=(10, 6))
plt.plot(logreg_recall, logreg_precision, color='brown', label='Logistic Regression')
plt.plot(rf_recall, rf_precision, color='green', linestyle='--', label='Random Forest')
plt.title('Precision-Recall Curve')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
