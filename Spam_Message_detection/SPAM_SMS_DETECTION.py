# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV File
file_path = '/content/spam.csv'  # Update with the correct file path
try:
    df = pd.read_csv(file_path)
    print("Successfully read the CSV file.")
except UnicodeDecodeError:
    print("Failed to read the CSV file with utf-8 encoding.")
    try:
        # Try reading with a different encoding
        df = pd.read_csv(file_path, encoding='latin1')
        print("Successfully read the CSV file with latin1 encoding.")
    except UnicodeDecodeError:
        print("Failed to read the CSV file with latin1 encoding. Please check the file encoding.")

# Print the first few rows to verify the data is read correctly
print(df.head())

# Drop unwanted Columns and rename the rest
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)

# Replace any NaN Values in the DataFrame with a Space
df.fillna(' ', inplace=True)

# Convert the "Category" Column Values to Numerical Representation (0 for "SPAM" and 1 for "HAM")
df['Category'] = df['Category'].apply(lambda x: 0 if x == 'spam' else 1)

# Separate the Feature (Message) and Target (Category) Data
X = df["Message"]
Y = df["Category"]

# Split the Data into Training and Testing Sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Create a TF-IDF Vectorizer to Convert Text Message into Numerical Features
feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

# Create the Training and Testing Text Message into Numerical Features Using TF-IDF
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Using Support Vector Machine (SVM) Model
model = SVC(kernel='linear')  # You can change the kernel type as needed
model.fit(X_train_features, Y_train)

# Prediction on the Training Data and Calculate the Accuracy
prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print("Accuracy on Training Data:", accuracy_on_training_data)

# Prediction on the Testing Data and Calculate the Accuracy
prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print("Accuracy on Testing Data:", accuracy_on_test_data)

# Test the Model with Some Custom Email Messages
input_your_mail = ["Congratulations! You Have Won a Free Vacation to an Exotic Destination. Click the Link to Claim Your Prize"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

# Prediction Result
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")

input_your_mail = ["Meeting Reminder: Tomorrow, 10 AM, Conference Room. See You There!"]
input_data_features = feature_extraction.transform(input_your_mail)
prediction = model.predict(input_data_features)
print(prediction)

# Prediction Result
if prediction[0] == 1:
    print("Ham Mail")
else:
    print("Spam Mail")

# Data Visualization: Distribution of Spam and Ham Emails
spam_count = df[df['Category'] == 0].shape[0]
ham_count = df[df['Category'] == 1].shape[0]

plt.bar(['Spam', 'Ham'], [spam_count, ham_count])
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Spam and Ham Emails')
plt.show()

# Confusion Matrix
cm = confusion_matrix(Y_test, prediction_on_test_data)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Oranges', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
probabilities = model.decision_function(X_test_features)
fpr, tpr, thresholds = roc_curve(Y_test, probabilities)
roc_auc = roc_auc_score(Y_test, probabilities)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
