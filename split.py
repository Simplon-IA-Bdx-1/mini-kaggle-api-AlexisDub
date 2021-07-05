from pandas import read_csv
from sklearn.model_selection import train_test_split

df = read_csv('cs-training.csv', index_col=0)

train, test = train_test_split(df, test_size=0.20)
train.to_csv('train2.csv')
test.to_csv('test2.csv')