import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']):
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = clf.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# Visualizations
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Churned", "Churned"],
            yticklabels=["Not Churned", "Churned"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(5, 4))
sns.countplot(data=data, x='Exited')
plt.title('Distribution of Churn vs. Non-Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='Geography', hue='Exited', data=data)
plt.title('Geography Distribution for Churned Customers')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.legend(['Not Churned', 'Churned'])

plt.figure(figsize=(10, 6))
sns.histplot(data[data['Exited'] == 0]['Age'], label='Not Churned', kde=True)
sns.histplot(data[data['Exited'] == 1]['Age'], label='Churned', kde=True)
plt.title("Age Distribution for Churned vs. Non-Churned Customers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# Correlation Matrix
correlation_matrix = X.corr()

# Plot the correlation matrix heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', square=True)
plt.title("Correlation Matrix of Features")
plt.show()
