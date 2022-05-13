from copyreg import pickle
from distutils.log import debug
from flask import *
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)

df = pd.read_csv(r"C:\Users\admin\Desktop\Chennai_House_Price_Prediction_Final\Chennai.csv")
pipe = pickle.load(open(r"C:\Users\admin\Desktop\Chennai_House_Price_Prediction_Final\model.pkl",'rb'))
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    location = sorted(df['Location'].unique())
    return render_template("prediction.html",location=location)

@app.route("/house_price",methods=['POST','GET'])
def house_price():
    sqft = request.form['sqft']
    bhk = request.form['bhk']
    location = request.form['location']
    parking = request.form['parking']
    pool = request.form['pool']
    track = request.form['track']
    water = request.form['water']
    columns=['Sqtft','Location','No. of Bedrooms','SwimmingPool','JoggingTrack','RainWaterHarvesting','CarParking']
    values=np.array([sqft,location,bhk,pool,track,water,parking])
    pred=pd.DataFrame(values.reshape(-1,len(values)),columns=columns)
    prediction = pipe.predict(pred)
    return render_template("result.html",prediction=prediction[0])
    
if __name__ == '__main__':
    app.run(debug=True)