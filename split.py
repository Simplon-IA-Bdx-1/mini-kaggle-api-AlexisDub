import pandas as pd
df = pd.read_csv('./train-original.csv')
test2 = df.iloc[1:30001]
test2-predictions = df['SeriousDlqin2yrs']
test2-predictions.to_csv("test2-predictions.csv", index=False)
test2.to_csv("test2.csv", index=False)