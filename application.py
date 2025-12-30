import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
SVM=pickle.load(open('Notebooks/SVM_final.pkl','rb'))
standard_scaler=pickle.load(open('Notebooks/Scaler_new.pkl','rb'))

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        no_of_dependents=int(request.form.get('no_of_dependents'))
        education = int(request.form.get('education'))
        self_employed = int(request.form.get('self_employed'))
        income_annum = int(request.form.get('income_annum'))
        loan_amount = int(request.form.get('loan_amount'))
        loan_term = int(request.form.get('loan_term'))
        cibil_score= int(request.form.get('cibil_score'))
        residential_assets_value = int(request.form.get('residential_assets_value'))
        commercial_assets_value = int(request.form.get('Commercial_assets_value'))

        new_data_scaled=standard_scaler.transform([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value]])

        result=SVM.predict(new_data_scaled)
        print(result[0])
        return render_template('home.html',result=int(result[0]))

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)