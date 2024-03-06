import pandas as pd
filename = 'diabetes.csv'
dataset = pd.read_csv(filename)
num_attributes = len(dataset.columns) - 1
specific_hypothesis = ['0'] * num_attributes
for index, row in dataset.iterrows():
    if row.iloc[-1] == 'yes':  # If instance is positive
        for i in range(num_attributes):
            if specific_hypothesis[i] == '0' or specific_hypothesis[i] == row[i]:
                specific_hypothesis[i] = row[i]
print("Final Specific Hypothesis:")
print(specific_hypothesis)
