from flask import Flask,render_template, redirect,request, url_for
from pycaret.classification import *
import pandas as pd
import numpy as np

app = Flask(__name__,
            template_folder="templates")

model = load_model('lr_final_model')
cols =  ['loan_amnt', 'term', 'int_rate', 'installment', 'grade',
       'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_util', 'total_acc']

@app.route('/predictions')
def predictions():
    return render_template("predictions.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = predict_model(model, data=data_unseen)
    output=prediction.Label[0]
    if (output == 1):
        state = "approved."
    else:
        state = "not approved."
    return render_template("predictions.html",pred='The loan request status is {}'.format(state))
        

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/dataset')
def dataset():
    return render_template("dataset.html")

@app.route('/overview')
def overview():
    return render_template("overview.html")
  
@app.route('/preprocess')
def preprocess():
    return render_template("preprocess.html")

@app.route('/ML')
def ml():
    return render_template("ML.html")

@app.route('/members')
def members():
    return render_template("members.html")


if __name__ == '__main__':
    app.run(debug=True)