# Diabetes Progression Prediction using Artificial Neural Networks


## Project Overview

This project focuses on predicting diabetes progression using the Diabetes dataset from the sklearn library.
An Artificial Neural Network (ANN) model is designed, trained, evaluated, and optimized to understand how different medical factors influence the disease progression.

## The project includes:

✔ Data loading and preprocessing
✔ Exploratory Data Analysis (EDA)
✔ ANN model building
✔ Model training and evaluation
✔ Hyperparameter tuning & model improvement
✔ Comparison with machine learning models
✔ Final observations and conclusion

## Dataset

Source: sklearn.datasets.load_diabetes()

Samples: 442

Features: 10 numeric medical predictors

Target: A quantitative measure of diabetes progression one year after baseline

##  1. Data Loading & Preprocessing

Loaded the dataset from sklearn

No missing values found

Data normalized using StandardScaler

Train-test split performed (80:20)

Why normalization?
ANNs perform significantly better when input features have similar scales.

## 2. Exploratory Data Analysis (EDA)

EDA included:

Feature distributions

Correlation heatmap

Scatter plots for feature–target relationships

Key Insights:

BMI, BP, S4, and S5 showed stronger correlation with diabetes progression

No multicollinearity issues

Dataset is small and slightly noisy → impacts model performance

## 3. Building the ANN Model

A baseline ANN was built using:

Dense layers

ReLU activation

MSE loss

Adam optimizer

The base model performed moderately, giving:

MSE ≈ 3343

R² ≈ 0.368

This served as a starting point for improvements.

## 4. Training & Evaluation

Trained for 100 epochs

Training and validation loss tracked

Minor overfitting observed in baseline model

Performance Metrics Used:

MSE (Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

## 5. Model Improvements

Several advanced techniques were applied:

### Improvements Included:

Additional hidden layers

Batch Normalization

Dropout layers

Learning rate scheduling

Early stopping

Smaller batch size

## 6. Best Model (Model 3)
###  Architecture

Dense(128) + BN + Dropout

Dense(64) + BN + Dropout

Dense(32)

Output: Dense(1)

## Best Performance:

MSE: 2811.19

R²: 0.4694 (highest among all models)

This model explains 47% of the variability in diabetes progression.

## 7. Comparison With Random Forest

A Random Forest regressor was trained for comparison.

MSE: 3016

R²: 0.431

Although good, it performed below the ANN Model 3, indicating the ANN captured non-linear patterns slightly better.

##  Model Comparison Summary
Model	MSE ↓	R² ↑
Base ANN	3343	0.368
Improved ANN	2879	0.456
Model 3 (Best)	2811	0.469
Random Forest	3016	0.431
## Final Conclusion

The ANN Model 3 delivered the best prediction accuracy.

Regularization (Dropout + BatchNorm) significantly improved generalization.

Learning rate tuning and early stopping stabilized the training.

Tree-based methods performed well but did not surpass the customized ANN.

With only 442 samples, collecting more data or using advanced models (e.g., XGBoost) could further boost performance.
