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



## Model Comparison

All five classification models were evaluated using the same Dry Bean dataset and test data. The evaluation metrics include Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

### Performance Comparison

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.9192 | 0.9934 | 0.9197 | 0.9192 | 0.9193 | 0.9023 |
| Decision Tree | 0.8966 | 0.9363 | 0.8965 | 0.8966 | 0.8964 | 0.8750 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | **0.9210** | 0.9933 | **0.9214** | **0.9210** | **0.9211** | **0.9045** |

### Comparative Observations

1. SVM achieved the highest accuracy of 92.10%, along with the highest Precision, Recall, F1-score and MCC.
2. Logistic Regression achieved the highest AUC score of 99.34%, slightly exceeding SVM's AUC of 99.33%.
3. KNN demonstrated strong classification performance with 91.55% accuracy, closely following Logistic Regression and SVM.
4. Decision Tree achieved 89.66% accuracy, showing good classification performance but lower overall scores than SVM, Logistic Regression and KNN.
5. Gaussian Naive Bayes achieved the lowest accuracy of 76.30%, although its AUC remained relatively high at 96.44%.
6. All five models classified the BOMBAY class perfectly.
7. Confusion between DERMASON and SIRA was observed across multiple models, indicating difficulty distinguishing these two bean classes.

### Best Performing Model

**Support Vector Machine (SVM)** was identified as the best-performing model based on the evaluation results.
SVM achieved the highest Accuracy (92.10%), Precision (92.14%), Recall (92.10%), F1-score (92.11%) and MCC (0.9045).
Although Logistic Regression achieved a marginally higher AUC score, SVM demonstrated the strongest overall performance across the majority of evaluation metrics.
Therefore, SVM was selected as the best-performing model for the Dry Bean classification task under the evaluated configuration.

Added the Saved Models and Preprocessing Artifacts section we prepared earlier to your README.
