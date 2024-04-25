# Breast Cancer Diagnosis using Bayesian Networks

## Overview
This project aims to diagnose breast cancer using a Bayesian network constructed from data provided by the University Medical Center, Institute of Oncology, in Ljubljana, Yugoslavia. The dataset contains information on 286 tumor instances, some malignant and others benign, described by 10 attributes.

## Dataset
The dataset includes the following attributes:
- Class: Diagnosis of the tumor (no-recurrence-events, recurrence-events)
- Age: Patient's age (10-19, 20-29, ..., 90-99)
- Menopause: Patient's menopausal status (lt40, ge40, premeno)
- Tumor-size: Tumor size (0-4, 5-9, ..., 55-59)
- Inv-nodes: Number of invaded lymph nodes (0-2, 3-5, ..., 36-39)
- Node-caps: Presence or absence of lymph node capsules (yes, no)
- Deg-malig: Degree of tumor malignancy (1, 2, 3)
- Breast: Affected breast (left, right)
- Breast-quad: Quadrant of the affected breast (left-up, left-low, ..., central)
- Irradiat: Irradiation treatment (yes, no)

## Structure Learning
Before constructing the Bayesian network structure, consultation with a medical professional was undertaken to understand the relationships between tumor characteristics and breast cancer diagnosis. The following steps were performed:
- Identification of relevant variables for breast cancer diagnosis from the dataset.
- Construction of the Bayesian network structure based on identified variables and information from the medical consultation.
- Graphical representation of the Bayesian network structure.

## Parameter Learning
The Expectation-Maximization (EM) algorithm was used to learn the parameters of the Bayesian network from the dataset.

## Inference and Diagnosis
The final Bayesian network was utilized for inference and medical diagnosis, estimating the probability of a particular diagnosis based on observed tumor characteristics.

## Report
The project report includes:
- Introduction
- Description of selected variables and their importance for breast cancer diagnosis, based on the medical consultation.
- Bayesian network structure with explanations justifying the links between variables.
- Explanation of the EM algorithm and its use in learning Bayesian network parameters.
- Explanation of inference and medical diagnosis based on the final Bayesian network.
- Graphical representation of the Bayesian network (showing structure and parameters).
- Any relevant observations or conclusions regarding the construction of the network and its application to medical diagnosis.

## Conclusion
This project demonstrates the application of Bayesian networks in medical diagnosis, specifically for breast cancer. The constructed network can assist in the diagnostic process by providing probabilistic assessments based on tumor characteristics.
