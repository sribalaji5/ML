import pandas as pd
filename = 'CarPrice.csv'
dataset = pd.read_csv(filename)
num_attributes = len(dataset.columns) - 1
hypothesis = ['0'] * num_attributes
for index, row in dataset.iterrows():
    if row.iloc[-1] == 'Yes':
        for i in range(num_attributes):
            if hypothesis[i] == '0' or hypothesis[i] == row[i]:
                hypothesis[i] = row[i]
            else:
                hypothesis[i] = '?'
print("Final Hypothesis:")
print(hypothesis)
