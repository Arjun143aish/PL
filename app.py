import pickle
from flask import Flask,render_template,request
import pandas as pd
import numpy as np

app = Flask(__name__)
Model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    feature_names = ['Age','Experience','Income',
                     'Zip Code','CCAvg','Mortgage','Family_one','Family_three',
                     'Family_two','Education_Professional','Education_Undergrad',
                     'Securities Account_Yes','CD Account_Yes','Online_Yes','CreditCard_Yes']
    
    df = pd.DataFrame(final_features,columns =feature_names)
    prediction = Model.predict(df)
    
    output = prediction
    
    if output == 0:
        status = 'Customer is not going for Personal Loan'
    else:
        status = 'Customer will opt for Personal Loan'
    
    return render_template('index.html',prediction_text = 'Status is {}'.format(status))


if __name__ == '__main__':
    app.run(debug=True)
    
    
        
    
        
