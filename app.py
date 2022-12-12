from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

heart_disease_data = pd.read_csv('heart.csv')

y = heart_disease_data["HeartDisease"].values

heart_disease_data = heart_disease_data.drop('HeartDisease', axis=1)

heart_disease_data = pd.get_dummies(heart_disease_data)

X = heart_disease_data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.preprocessing import StandardScaler

# Create a StandardScater model and fit it to the training data

X_scaler = StandardScaler()
X_scaler.fit(X_train)

# Transform the training and testing data by using the X_scaler and y_scaler models

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

y_true = y_test
y_pred = model.predict(X_test)
confusion_matrix(y_true, y_pred)

print(f"Training Data Score: {model.score(X_train, y_train)}")
print(f"Testing Data Score: {model.score(X_test, y_test)}")



app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
