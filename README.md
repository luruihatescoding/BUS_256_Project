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
![alt text](https://github.com/luruihatescoding/BUS_256_Project/blob/[branch]/image.jpg?raw=true)

