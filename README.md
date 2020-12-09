# BUS_256_Project

## Introduction

For this final project, I am going to work on heart disease prediction using Machine Learning.

Heart diseases have drawn attention from all over the world since it is the leading cause of death. I want to figure out which model can best predict whether a person has heart disease. I collected this **_Heart Disease Dataset_** from [**_Kaggle_**](https://www.kaggle.com/ronitf/heart-disease-uci). There are 14 arrtibutes in this dataset, 1 response variabe (Y) and 13 explanatory variables (X). Y is a dummy variable named "target", with 0 represents no heart disease detactedand 1 represents heart disease detacted. The Xs are age of patients, sex, chest pain type, resting blood pressure, serum cholestoral, fasting blood sugar, resting electrocardiographic results, max heart rate achieved, exercise induced angina, and ST depression induced by exercise relative to rest.

## Library

The first step is to import necessary libraries.
```
import pandas as pd
import os
```
Here I imported pandas and os for the next step, which is importing the dataset. The other necessary libraries which are related to regressions and models evolved in machine learning will be imported later in each section.

## Import Dataset

Importing the dataset to Python
```
os.chdir('C:\\Users\\yulur\\iCloudDrive\\Brandeis University\\Bus 256\\BUS_256_Project')

heart_data = pd.read_csv('heart.csv')
heart_data
```
Then, I allocated the response variabe (Y) and explanatory variables (X).
```
X = heart_data.iloc[:, 0:12].values
X
Y = heart_data.iloc[:, 13].values
Y
```
With the following code, I checked missing values in the dataset
```
heart_data.isnull().sum()
heart_data.isna().sum()
```
The result indicated there is no missing value.

## Data Processing

Usually, the datasets downloaded are not ready to be used for building models. There are few steps to do before jumping into machine learning. I first created the histograms of each variable to check if they have the same scale
```
heart_data.hist()
```
![alt text](https://github.com/luruihatescoding/BUS_256_Project/blob/main/histogram.png?raw=true)
The histograms indicated that a data scaling is necessary.
Before data scaling, I split the training data (75%) and test data (25%) by
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
```
and followed by data processing including transfering categorical variables to dummy variables
```
heart_data = pd.get_dummies(heart_data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Model Testing

I tested 4 models: KNN, Naive Bayes, Regression Trees and Neural Network, which are learned from class. Then I got a confusion matrix from each model cand calculated the accuracy rate

### KNN

```
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric='minkowski', p = 2)
classifier.fit(X_train, Y_train)

Y_valid = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_valid)
```
Confusion Matrix
(25,  8
  6, 37)
Accuracy Rate: 81.58%

### Naive Bayes

```
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_valid = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_valid)
```
Confusion Matrix
(22, 11
  6, 37)
Accuracy Rate: 77.63%

### Regression Trees

```
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)

Y_valid = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_valid)
```
Confusion Matrix
(22, 11
  8, 35)
Accuracy Rate: 75.00%

### Neural Network

```
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier()
classifier.fit(X_train, Y_train)

Y_valid = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_valid)
```
Confusion Matrix
(25,  8
  6, 37)
Accuracy Rate: 81.58%

## Conclusion

The accuracy rates are:
KNN             : 81.58%
Naive Bayes     : 77.63%
Regression Trees: 75.00%
Neural Network  : 81.58%

For this dataset, Key Nearest Neighbor and Neural Network generated the same results and they have the best accuracy rate. The result showed we can use KNN and Neural Network to predict whether a patient has heart disease and better improve the effiency of hospital.
