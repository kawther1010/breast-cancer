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
# train_df || df

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# Afficher les distributions de probabilité conditionnelle apprises pour chaque nœud    
for cpd in model.get_cpds():
    print("CPD de {variable}:".format(variable=cpd.variable))
    print(cpd)


# Vous pouvez maintenant utiliser le modèle entraîné pour l'inférence et le diagnostic médical sur l'ensemble de test

"""
# Prédictions sur l'ensemble de test
#  || Deg_malig
predictions = model.predict(test_data.drop(columns=['Class']))

# Comparaison avec les étiquettes réelles
correct_predictions = (predictions['Class'] == test_data['Class']).sum()
total_samples = len(test_data)
accuracy = correct_predictions / total_samples

print(f"Précision sur l'ensemble de test : {accuracy}")

"""