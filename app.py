from flask import Flask, render_template, url_for, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
import pickle
from flask_sqlalchemy import SQLAlchemy

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

# Along with replace() and map(), this is another way to encode the gender column into numbers.
heart_disease_data['ChestPainType'] = heart_disease_data['ChestPainType'].apply(changeChestPainType)

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
print(heart_disease_data)

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

ENV = 'dev'
if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost/Heartdisease'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = ''

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.app_context().push()

class Heart_Disease_data(db.Model):
    __tablename__ = 'heart disease data'
    Age = db.Column(db.Integer, primary_key=True)
    Sex = db.Column(db.Integer)
    Chest_Pain_Type = db.Column(db.Integer)
    Resting_BP = db.Column(db.Integer)
    Cholesterol = db.Column(db.Integer)
    Fasting_bs = db.Column(db.Integer)
    Resting_ECG = db.Column(db.Integer)
    MaxHR = db.Column(db.Integer)
    Exercise_Angina = db.Column(db.Integer)
    Old_Peak = db.Column(db.Float)
    ST_Slope = db.Column(db.Integer)

    def __init__(self, Sex, Chest_Pain_Type, Resting_BP, Cholesterol, Fasting_bs, Resting_ECG, MaxHR, Exercise_Angina, Old_Peak, ST_Slope ):
        
        self.Sex=Sex
        self.Chest_Pain_Type=Chest_Pain_Type
        self.Resting_BP=Resting_BP
        self.Cholesterol=Cholesterol
        self.Fasting_bs=Fasting_bs
        self.Resting_ECG=Resting_ECG
        self.MaxHR=MaxHR
        self.Exercise_Angina=Exercise_Angina
        self.Old_Peak=Old_Peak
        self.ST_Slope=ST_Slope




@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    # pickled_model = pickle.load(open('knnmodel.pkl', 'rb'))
    # scaler = pickle.load(open('scaled.pkl', 'rb'))
    
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



    return render_template('prediction_result.html', data=prediction)

@app.route('/submit', methods=["GET", "POST"])
def submit():
    if request.method == 'POST':
        
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
        # if Age == '' or Sex == '' or Chest_Pain_Type == '' or Resting_BP == '' or Cholesterol == '' or Fasting_bs == '' or Resting_ECG == '' or MaxHR == '' or Exercise_Angina == '' or Old_Peak == '' or ST_Slope == '':
        #     return render_template('index.html', message='Please fill out all of the cells')
        # print(Age, Sex, Chest_Pain_Type)
        data=Heart_Disease_data(Sex, Chest_Pain_Type, Resting_BP, Cholesterol, Fasting_bs, Resting_ECG, MaxHR, Exercise_Angina, Old_Peak, ST_Slope)
        db.session.add(data)
        db.session.commit


if __name__ == '__main__':
    app.run(debug=True)
