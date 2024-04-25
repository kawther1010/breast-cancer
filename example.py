import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization

from pgmpy.estimators import ExpectationMaximization as EM

data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
estimator = ExpectationMaximization(model, data)

data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 3)), columns=['A', 'C', 'D'])
model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')], latents={'B'})
estimator = EM(model, data)
estimator.get_parameters(latent_card={'B': 3})

"""
[<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
<TabularCPD representing P(B:3 | C:2, A:2) at 0x7f7b4dfd4da0>,
<TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
<TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
"""