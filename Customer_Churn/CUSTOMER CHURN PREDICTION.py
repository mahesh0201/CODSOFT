# Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Importing CSV File
data = pd.read_csv(r'/content/Churn_Modelling.csv')

# Showing Data
data.head()
data.isnull().sum()
corre = data.corr()

data.info()

# Dropping the Columns
data1 = data.drop(['Surname','CustomerId','RowNumber'],axis=1)

# Checking DataTypes
data1.dtypes

# Describing the data1
data1.describe()

co = data1.corr()

# This is Showing that the dataset is imbalanced
data1['Exited'].value_counts()
data1.hist(figsize=(20,20))

plt.figure(figsize=(10,8))
sns.heatmap(co, annot = True, vmax = 8, square = True)
plt.show()

data1.sample(10)
data1.head()

# Importing the seaborn library
sns.boxplot(data1['Balance'])
sns.boxplot(data1['EstimatedSalary'])
sns.boxplot(data1['Tenure'])

data1['Tenure'].value_counts()
sns.countplot(x = 'Tenure', hue = "Exited", data = data1)
sns.countplot(x = data1['Exited'], data = data1)

# For Geography
print("Before Encoding", data1['Geography'].unique())
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data1['Geography']=label.fit_transform(data1['Geography'])
print("After Encoding", data1['Geography'].unique())

# For Gender
print("Before Encoding", data1['Gender'].unique())
label = LabelEncoder()
data1['Gender']=label.fit_transform(data1['Gender'])
print("After Encoding", data1['Gender'].unique())

data1.head()
data1.dtypes

zeros,ones = data1.Exited.value_counts()
fr1 = data1[data1['Exited']==1]
nfr1 = data1[data1['Exited']==0]
nfr1=nfr1.sample(n=ones, replace= False)
pdata = pd.concat([fr1,nfr1],axis=0)

data1['Exited'].value_counts()

x = data1.drop('Exited', axis=1)
y = data1['Exited']
x.dtypes


# Spliting the test and test data 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state = 2)
x.shape
y.shape

from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
x_train_Scaled = sclr.fit_transform(x_train)
x_test_Scaled = sclr.fit_transform(x_test)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate = 0.17, random_state = 2)
print(model.fit(x_train, y_train))
y_pred = model.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Considering you've "y_pred" containing your model's predictions
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
sns.heatmap(conf_matrix, annot = True, fmt = "d", cmap = "Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc 
 # Get Predicted probabilities for the positive class
y_probs = model.predict_proba(x_test)[:, 1]

# Calculating ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Ploting ROC Curve
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc = "lower right")
plt.show()

# Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_train, y_train)

y_pred1 = lg.predict(x_test)


# After Fitting the X & Y with Logistic Regression Algorithm Now let's Check the Confusion Matrix, Accuracy Score with its Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred1))

from sklearn.metrics import roc_curve, auc 
 # Get Predicted probabilities for the positive class
y_probs = model.predict_proba(x_test)[:, 1]

# Calculating ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Ploting ROC Curve
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc = "lower right")
plt.show()

# Support Vector Classifier 
from sklearn import svm
sv = svm.SVC()
sv.fit(x_train, y_train)

y_pred2 = sv.predict(x_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred2))

from sklearn.metrics import roc_curve, auc 
 # Get Predicted probabilities for the positive class
y_probs = model.predict_proba(x_test)[:, 1]

# Calculating ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Ploting ROC Curve
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, label = 'ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = 3, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc = "lower right")
plt.show()

# Handling Imbalance Data Using SMOTE Module

from imblearn.over_sampling import SMOTE
x_re, y_re = SMOTE().fit_resample(x,y)
y_re.value_counts()

x_res = sclr.fit_transform(x_re)
sv.fit(x_re,y_re)

import joblib
joblib.dump(sv, 'Prediction_Model_File')

prediction_part = joblib.load('Prediction_Model_File')

data1.columns

# So, Based Upon the Above Given Columns We Give Input as 11 Values for Prediction
# We import Warnings Module Inorder to Supress the Warnings in the Output
import warnings
warnings.filterwarnings('ignore')

# Prediction_Part
prediction_part.predict([[210,92,2,1.0,1,0,301.231,1,1,0]])
