import pandas as pd

# Read training data from CSV file
csv_file_path = 'diabetes.csv'
training_data = pd.read_csv(csv_file_path)

# Initialize hypotheses
num_attributes = len(training_data.columns) - 1
specific_hypothesis = ['0'] * num_attributes
general_hypothesis = ['?'] * num_attributes

# Candidate-Elimination algorithm
for index, row in training_data.iterrows():
    instance = row[:-1].tolist()
    label = row.iloc[-1]

    if label == 'yes':  # Positive instance
        for i in range(num_attributes):
            if specific_hypothesis[i] == '0' or specific_hypothesis[i] == instance[i]:
                specific_hypothesis[i] = instance[i]
            else:
                specific_hypothesis[i] = '?'

        for i in range(num_attributes):
            if general_hypothesis[i] != specific_hypothesis[i] and specific_hypothesis[i] != '?':
                general_hypothesis[i] = specific_hypothesis[i]
    else:  # Negative instance
        for i in range(num_attributes):
            if instance[i] != specific_hypothesis[i] and specific_hypothesis[i] != '0':
                general_hypothesis[i] = specific_hypothesis[i]

# Display the final hypotheses
print("Specific Hypothesis:")
print(specific_hypothesis)
print("\nGeneral Hypothesis:")
print(general_hypothesis)
