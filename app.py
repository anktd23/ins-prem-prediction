from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors=CORS(app)

file = open("/config/workspace/saved_models/2/model/model.pkl", 'rb')
model = pickle.load(file)

data = pd.read_csv('/config/workspace/insurance.csv')

@app.route('/',methods=['GET','POST'])
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex= sex, smoker= smoker, region= region)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    prediction = model.predict(pd.DataFrame(columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'],data=np.array(['age', 'sex', 'bmi', 'children', 'smoker', 'region']).reshape(1, 6)))

    return str(prediction[0])           

if __name__=="__main__":
    app.run(debug=True)