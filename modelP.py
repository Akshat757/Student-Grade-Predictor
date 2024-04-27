from flask import Flask, render_template,request
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import os
app=Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/',methods=['GET'])
def hello():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        features=[]
        for i in range(1,20):
            feature_name='input'+str(i)
            feature_value=float(request.form[feature_name])
            features.append(feature_value)  
        
        input_features = np.array([features]).reshape(1, -1)
        prediction = model.predict(input_features)
        pred_int = int(prediction)

        return render_template('result.html',predictionValue=prediction, predictionInt=pred_int)
        

if __name__=='__main__':
    app.run(port=3000,debug=True)