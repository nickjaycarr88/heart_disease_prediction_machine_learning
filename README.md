# **heart disease prediction-machine learning**
---
### The aim of this project is to create a machine learning web application from a data set from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). This data set contains a collection of data from 5 different hospitals and over 900 patients records collected. 

## Data Collection
### Once importing thr data, the standard process of checking to see what data is available, missing, data types, categorical/ numerical etc is performed. This allows me to get familiar woth the data and start generating some ideas as to what to look for.

## Data exploration
### A few different graphs were created in order to see if there are any findings between the columns offered. For this data set, as there are male and female inputs, i thought it would be a good idea to explore any possible findings from this. 
![Image Link](https://github.com/nickjaycarr88/heart_disease_prediction_machine_learning/blob/main/images/ECG.png)

## Machine Learning
### I applied various machine learning models on the data, including Linear, Logistic regression, SVC, KNN, Neurol Network and Random Forrest. I first ran these models using the default hyperparameters, then I ran again adjusting them to get the model with the highest accuracy. The Model which worked the best for me was the random forrest. The code below shows the parameters best used on this model
```python
RF_grid_best_params = { 'max_depth': [6],
 'max_features': ['sqrt'],
 'min_samples_leaf': [1],
 'min_samples_split':[5],
 'n_estimators': [33],
 'bootstrap': [True]
    
}
``` 
### This gave a training score of 92% and a testing score of 91%. 

## Database
### The idea behind the application is for the user to input data, then be returned a result of their likelihood of having or not having heart disease. However, the idea for the database allows for the data to be collected and then be used again in the model. This will allow for the model to become more accurate. I used postgres, wil the folling inputs recorded: 
```python
class Heart_Disease_data(db.Model):
    __tablename__ = 'heart_disease_data'
    name = db.Column(db.String, primary_key=True)
    age = db.Column(db.Integer)
    sex = db.Column(db.Integer)
    chest_Pain_Type = db.Column(db.Integer)
    resting_BP = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    fasting_bs = db.Column(db.Integer)
    resting_ECG = db.Column(db.Integer)
    maxHR = db.Column(db.Integer)
    exercise_Angina = db.Column(db.Integer)
    old_Peak = db.Column(db.Float)
    sT_Slope = db.Column(db.Integer)
```
## Web application
### Once landing on the page, the user inputs their name, age, sex, chest pain type, resting BP, cholesterol, fasting bs, resting ecg, maximum heart rate, exercise angina, old peak and st slope.

