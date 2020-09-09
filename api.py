from flask import Flask, request, redirect, url_for, flash, jsonify
import json
import pandas as pd
from sklearn.metrics import  roc_auc_score

app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def predict():
    df = request.files['file']
    df = pd.read_csv(df)
    df = df['SeriousDlqin2yrs']
    df2 = pd.read_csv('test2-predictions.csv')
    df2 = df2['SeriousDlqin2yrs']
    auc_score = roc_auc_score(df, df2)
    return jsonify(auc_score)

if __name__ == "__main__" :
    app.run()