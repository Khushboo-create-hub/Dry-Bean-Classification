# Dry Bean Classification

## Project Overview

This project is part of the Machine Learning Assignment.

The objective is to implement and compare five machine learning classification algorithms using the UCI Dry Bean dataset.

## Dataset

Source: UCI Machine Learning Repository

Dataset: Dry Bean Dataset

Total original instances: 13,611

Instances after duplicate removal: 13,543

Input features: 16

Target variable: Class

Classification type: Multiclass (7 classes)

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Project Status

Dataset inspection and preprocessing completed.

## Performance Observations

1. Logistic Regression

Achieved 91.92% accuracy and 99.34% AUC.
Demonstrated strong overall classification performance.
Classified the BOMBAY class perfectly.
Some confusion was observed between DERMASON and SIRA.

2. Decision Tree

Achieved 89.66% accuracy and 93.63% AUC.
Classified the BOMBAY class perfectly.
Demonstrated good performance for HOROZ and SEKER.
Had difficulty distinguishing between DERMASON and SIRA.

3. KNN Model

Achieved 91.55% accuracy and 98.11% AUC.
Demonstrated strong classification performance, particularly for BOMBAY, HOROZ and SEKER. 
Some confusion was observed between DERMASON and SIRA.

4. Gaussian Naive Bayes

Achieved 76.30% accuracy and 96.44% AUC.
Classified the BOMBAY class perfectly.
Demonstrated good performance for DERMASON.
Had difficulty distinguishing between BARBUNYA and CALI, and between DERMASON and SEKER.

5. Support Vector Machine (SVM)

Achieved 92.10% accuracy and 99.33% AUC.
Demonstrated strong overall classification performance.
Classified the BOMBAY class perfectly.
Some confusion was observed between DERMASON and SIRA.


