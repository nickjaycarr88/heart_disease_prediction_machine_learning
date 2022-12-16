from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle
from flask_sqlalchemy import SQLAlchemy
import psycopg2
import psycopg2.extras
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


heart_disease_data = pd.read_csv('heart.csv')

# Transform Chest Pain Type
def changeChestPainType(ChestPainType):
    if ChestPainType == "ASY":
        return 0
    elif ChestPainType == "NAP":
        return 1
    elif ChestPainType == "ATA":
        return 2
    else:
        return 3

heart_disease_data['ChestPainType'] = heart_disease_data['ChestPainType'].apply(changeChestPainType)

# Transform Resting ECG
def changeRestingECG(RestingECG):
    if RestingECG == "Normal":
        return 0
    elif RestingECG == "LVH":
        return 1
    else:
        return 2


heart_disease_data['RestingECG'] = heart_disease_data['RestingECG'].apply(changeRestingECG)



# Transform ST SLope
def changeST_Slope(ST_Slope):
    if ST_Slope == "Flat":
        return 0
    elif ST_Slope == "Up":
        return 1
    else:
        return 2

heart_disease_data['ST_Slope'] = heart_disease_data['ST_Slope'].apply(changeST_Slope)

# Transform Sex
def changeSex(sex):
    if sex == "M":
        return 0
    
    else:
        return 1

heart_disease_data['Sex'] = heart_disease_data['Sex'].apply(changeSex)

# Transform ExerciseAngina
def changeExerciseAngina(ExerciseAngina):
    if ExerciseAngina == "Y":
        return 0
    
    else:
        return 1

heart_disease_data['ExerciseAngina'] = heart_disease_data['ExerciseAngina'].apply(changeExerciseAngina)

#Set the X and y variables
y = heart_disease_data['HeartDisease']
X = heart_disease_data.drop(['HeartDisease'], axis = 1)

#Train test and split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Set the standard scaler
X_scaler = StandardScaler()
X_scaler.fit(X_train)

#Create scaler variable, then transform
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

#fit the machine learning model

RF_model = RandomForestClassifier()

RF_grid_best_params = { 'max_depth': [6],
 'max_features': ['sqrt'],
 'min_samples_leaf': [1],
 'min_samples_split':[5],
 'n_estimators': [33],
 'bootstrap': [True]
    
}

RF_grid_params = GridSearchCV(estimator=RF_model, param_grid =RF_grid_best_params, cv=3, verbose=2, n_jobs = 4 )

RF_grid_params.fit(X_train, y_train)

app = Flask(__name__)

#Set up the postgres connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/heart_disease_data'

db = SQLAlchemy(app)
app.app_context().push()
#DB class for the table input fields
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

#Home page route
@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')
#Prediction page route
@app.route('/predict', methods=["POST"])
def predict():
    
    Name=request.form['Name']
    Age = request.form['Age']
    Sex = request.form['Sex']
    Chest_Pain_Type = request.form['Chest_Pain_Type']
    Resting_BP = request.form['Resting_BP']
    Cholesterol = request.form['Cholesterol']
    Fasting_bs = request.form['Fasting_bs']
    Resting_ECG = request.form['Resting_ECG']
    MaxHR = request.form['MaxHR']
    Exercise_Angina = request.form['Exercise_Angina']
    Old_Peak = request.form['Old_Peak']
    ST_Slope = request.form['ST_Slope']
    
    predict_example = (Age, Sex, Chest_Pain_Type, Resting_BP, Cholesterol, Fasting_bs, Resting_ECG, MaxHR, Exercise_Angina, Old_Peak, ST_Slope)    
    input_data_as_numpy_array = np.asarray(predict_example)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    std_data = X_scaler.transform(input_data_reshape)
    prediction = RF_grid_params.predict(std_data)[0]
 
    data=Heart_Disease_data(name = Name, age=Age, sex=Sex, chest_Pain_Type=Chest_Pain_Type, resting_BP=Resting_BP, cholesterol=Cholesterol, fasting_bs=Fasting_bs, resting_ECG=Resting_ECG, maxHR=MaxHR, exercise_Angina=Exercise_Angina, old_Peak=Old_Peak, sT_Slope=ST_Slope)
    db.session.add(data)
    db.session.commit()

    return render_template('prediction_result.html', data=prediction)

if __name__ == '__main__':
    app.run(debug=True)
