from flask import Flask, render_template, request
### loading model 

import pickle

file = open("diabetes.pkl", "rb")
model = pickle.load(file)
file.close()

app = Flask(__name__)

@app.route('/',methods=["GET", "POST"])

def homePage():
    if request.method == "POST":

        myDict = request.form

        Pregnancies = float(myDict['Pregnancies'])
        Glucose = float(myDict['Glucose'])
        BloodPressure = float(myDict['BloodPressure'])
        SkinThickness = float(myDict['SkinThickness'])
        Insulin = float(myDict['Insulin'])
        BMI = float(myDict['BMI'])
        DiabetesPedigreeFunction = float(myDict['DiabetesPedigreeFunction'])
        Age = float(myDict['Age'])

        inputfeatures = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,DiabetesPedigreeFunction, Age]

        infprob = model.predict_proba([inputfeatures])[0][1]
        print(infprob)
        return render_template('show.html', inf=infprob*100)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)