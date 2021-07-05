import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

train = pd.read_csv('train2.csv')
test = pd.read_csv('test2.csv')
target = 'SeriousDlqin2yrs'
y_train = train[target]
x_train = train.drop(columns=[target])
x_test = test.drop(columns=[target]) 

model = SGDClassifier()
model.fit(x_train,y_train)
data = model.predict(x_test)
data.to_csv('test2-predictions.csv')