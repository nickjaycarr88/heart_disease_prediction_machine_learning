from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
import pickle
from flask_sqlalchemy import SQLAlchemy
import psycopg2
import psycopg2.extras


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

# Along with replace() and map(), this is another way to encode the gender column into numbers.
heart_disease_data['ChestPainType'] = heart_disease_data['ChestPainType'].apply(changeChestPainType)

# Transform Resting ECG
def changeRestingECG(RestingECG):
    if RestingECG == "Normal":
        return 0
    elif RestingECG == "LVH":
        return 1
    else:
        return 2

# Along with replace() and map(), this is another way to encode the gender column into numbers.
heart_disease_data['RestingECG'] = heart_disease_data['RestingECG'].apply(changeRestingECG)



# Transform ST SLope
def changeST_Slope(ST_Slope):
    if ST_Slope == "Flat":
        return 0
    elif ST_Slope == "Up":
        return 1
    else:
        return 2

# Along with replace() and map(), this is another way to encode the gender column into numbers.
heart_disease_data['ST_Slope'] = heart_disease_data['ST_Slope'].apply(changeST_Slope)

# Transform Sex
def changeSex(sex):
    if sex == "M":
        return 0
    
    else:
        return 1

# Along with replace() and map(), this is another way to encode the gender column into numbers.
heart_disease_data['Sex'] = heart_disease_data['Sex'].apply(changeSex)

# Transform ExerciseAngina
def changeExerciseAngina(ExerciseAngina):
    if ExerciseAngina == "Y":
        return 0
    
    else:
        return 1

heart_disease_data['ExerciseAngina'] = heart_disease_data['ExerciseAngina'].apply(changeExerciseAngina)


y = heart_disease_data['HeartDisease']
X = heart_disease_data.drop(['HeartDisease'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_scaler = StandardScaler()
X_scaler.fit(X_train)
# Transform the training and testing data by using the X_scaler and y_scaler models

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

SVC_best_params = SVC(C=1, kernel='linear')
SVC_best_params.fit(X_train, y_train)

# model = pickle.load(open("SVC_pickle.pkl", "rb"))
# X_scaler = pickle.load(open("scaler_pickle.pkl", "rb"))

app = Flask(__name__)

# ENV = 'dev'
# if ENV == 'dev':
#     app.debug = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/heart_disease_data'
# else:
#     app.debug = False
#     app.config['SQLALCHEMY_DATABASE_URI'] = ''

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.app_context().push()

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

    # def __init__(self, Age, Sex, Chest_Pain_Type, Resting_BP, Cholesterol, Fasting_bs, Resting_ECG, MaxHR, Exercise_Angina, Old_Peak, ST_Slope ):
    #     self.age=Age
    #     self.sex=Sex
    #     self.chest_Pain_Type=Chest_Pain_Type
    #     self.resting_BP=Resting_BP
    #     self.cholesterol=Cholesterol
    #     self.fasting_bs=Fasting_bs
    #     self.resting_ECG=Resting_ECG
    #     self.maxHR=MaxHR
    #     self.exercise_Angina=Exercise_Angina
    #     self.old_Peak=Old_Peak
    #     self.sT_Slope=ST_Slope




@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # pickled_model = pickle.load(open('knnmodel.pkl', 'rb'))
    # scaler = pickle.load(open('scaled.pkl', 'rb'))
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
    prediction = SVC_best_params.predict(std_data)[0]

   
        
       
        
    data=Heart_Disease_data(name = Name, age=Age, sex=Sex, chest_Pain_Type=Chest_Pain_Type, resting_BP=Resting_BP, cholesterol=Cholesterol, fasting_bs=Fasting_bs, resting_ECG=Resting_ECG, maxHR=MaxHR, exercise_Angina=Exercise_Angina, old_Peak=Old_Peak, sT_Slope=ST_Slope)
    db.session.add(data)
    db.session.commit()



    return render_template('prediction_result.html', data=prediction)

# @app.route('/submit', methods=["GET", "POST"])
# def submit():
#     if request.method == 'POST':
#         Age = request.form['Age']
#         Sex = request.form['Sex']
#         Chest_Pain_Type = request.form['Chest_Pain_Type']
#         Resting_BP = request.form['Resting_BP']
#         Cholesterol = request.form['Cholesterol']
#         Fasting_bs = request.form['Fasting_bs']
#         Resting_ECG = request.form['Resting_ECG']
#         MaxHR = request.form['MaxHR']
#         Exercise_Angina = request.form['Exercise_Angina']
#         Old_Peak = request.form['Old_Peak']
#         ST_Slope = request.form['ST_Slope']
#         # if Age == '' or Sex == '' or Chest_Pain_Type == '' or Resting_BP == '' or Cholesterol == '' or Fasting_bs == '' or Resting_ECG == '' or MaxHR == '' or Exercise_Angina == '' or Old_Peak == '' or ST_Slope == '':
#         #     return render_template('index.html', message='Please fill out all of the cells')
#         print(Age, Sex, Chest_Pain_Type)
#         data=Heart_Disease_data(age=Age, sex=Sex, chest_Pain_Type=Chest_Pain_Type, resting_BP=Resting_BP, cholesterol=Cholesterol, fasting_bs=Fasting_bs, resting_ECG=Resting_ECG, maxHR=MaxHR, exercise_Angina=Exercise_Angina, old_Peak=Old_Peak, sT_Slope=ST_Slope)
#         db.session.add(data)
#         db.session.commit()


if __name__ == '__main__':
    app.run(debug=True)
