import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import ExpectationMaximization

from pgmpy.estimators import ExpectationMaximization as EM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df= pd.read_csv("breast-cancer.csv")

model = BayesianNetwork([('Age', 'Menopause'),
                         ('Age', 'Deg_malig'),
                         ('Irradiat', 'Deg_malig'),
                         ('Irradiat', 'Breast'),
                         ('Menopause', 'Deg_malig'),
                         ('Tumor_size', 'Deg_malig'),
                         ('Tumor_size', 'Class'),
                         ('Inv_nodes', 'Class'),
                         ('Inv_nodes', 'Node_caps'),

                         ('Node_caps', 'Deg_malig'),
                         ('Tumor_size', 'Inv_nodes'),
                         ('Node_caps', 'Class'),
                         ('Breast', 'Breast_quad'),
                         ('Breast_quad', 'Node_caps'),
                         ('Class', 'Deg_malig')
                         ])

estimator = EM(model, df)
estimator.get_parameters()

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

for cpd in model.get_cpds():
    print("CPD de {variable}:".format(variable=cpd.variable))
    print(cpd)

# Check unique state names in the training data
unique_states = {}
for column in train_data.select_dtypes(include=['object']).columns:
    unique_states[column] = train_data[column].unique()

# Update state names in the test data
for column, states in unique_states.items():
    most_frequent_state = train_data[column].mode()[0]  # Get the most frequent state in training data
    test_data[column] = test_data[column].apply(lambda x: x if x in states else most_frequent_state)

# Make predictions on the updated test data
predicted_labels = []
threshold = 0.2  # meilleur seuil
for index, row in test_data.iterrows():
    # Construct a DataFrame with a single row containing the values from the test row
    test_row = pd.DataFrame([row.drop('Class')], columns=row.index.drop('Class'))
    # Predict the probabilities for 'Class' for the current row
    predicted_probabilities = model.predict_probability(test_row)
    # Assign label based on threshold
    predicted_label = 1 if predicted_probabilities.iloc[0, 1] > threshold else 0
    predicted_labels.append(predicted_label)

# Convert predicted labels back to class names
predicted_labels = ['recurrence-events' if label == 1 else 'no-recurrence-events' for label in predicted_labels]

# Calculate evaluation metrics
true_labels = test_data['Class']
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, pos_label='recurrence-events')
recall = recall_score(true_labels, predicted_labels, pos_label='recurrence-events')
f1 = f1_score(true_labels, predicted_labels, pos_label='recurrence-events')

# Print evaluation metrics
print("Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")