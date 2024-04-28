import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization

from pgmpy.estimators import ExpectationMaximization as EM

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