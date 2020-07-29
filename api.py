from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd

app = Flask(__name__)

@app.route('/api/test', methods=['POST'])
def predict():
    df = pd.read_csv('./test2.csv')
    df = df['SeriousDlqin2yrs']
    df2 = pd.read_csv('./test2-predictions.csv')
    df2 = df2['SeriousDlqin2yrs']
    df['comparaison'] = np.where(df['SeriousDlqin2yrs'] == df2['SeriousDlqin2yrs'], 'True', 'False')

    return jsonify(df.comparaison.value_counts())

if __name__ == "__main__" :
    app.run()