# Car Price Prediction 🚙 | Multiple Regression Models 📊

## Project Overview

The goal of this project is to predict the price of used cars based on various attributes using multiple regression models. The project applies machine learning techniques to analyze the dataset and provide accurate price predictions.

## Table of Contents

1. [Installation](#installation)
2. [Data Description](#data-description)
3. [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Missing Value Imputation](#missing-value-imputation)
4. [Modeling](#modeling)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [License](#license)

## Installation

Either clone the repo and run it locally,

```
https://github.com/YashsTiwari/Used-Car-Price-Prediction.git
```
Or you can run it on colab using the link in the notebook.

The dataset is included in this repository, which can be directly loaded into the Colab notebook (adjust the import path for dataset)

## Data Description

The dataset consists of various features related to used cars, including:

- `brand`: The brand of the car.
- `model`: The car's model.
- `engine`: Information on engine specifications.
- `ext_col`: Exterior color of the car.
- `int_col`: Interior color of the car.
- `transmission`: Type of transmission (automatic/manual).
- `fuel_type`: Type of fuel (e.g., petrol, diesel).
- `accident`: Whether the car has been in an accident (binary).
- `clean_title`: Indicates if the car has a clean title (binary).
- `model_year`: The release year of the car model.
- `price`: The target variable representing the price of the car.

## Data Preprocessing

### Feature Engineering

Several features were derived from existing ones:

- **Engine Features**: Extracted components from the `engine` feature:
  - `Horsepower`
  - `Liters_engine`
  - `Cylinders_count`
  
- **Model Age**: Calculated the age of the car based on the `model_year`.

### Missing Value Imputation

KNN imputer failed so other imputation techniques were applied for various features:

- The `accident` feature was filled with "None reported" for missing values.
- `IterativeImputer` was applied for `Horsepower` and `Liters_engine`, while `SimpleImputer` was used for `Cylinders_count`.
- Manual imputation of fuel_type, clean_title, etc were done.

### Encoding and Scaling

- Categorical features were encoded using `LabelEncoder`.
- Continuous features were scaled using `StandardScaler`.
- The target variable `price` was log-transformed for normality.

## Modeling

Multiple regression models were implemented to predict car prices, including:

1. **Decision Tree Regressor**
2. **Random Forest Regressor**
3. **AdaBoost Regressor**
4. **XGBoost Regressor**
5. **LightGBM Regressor**
6. **CatBoost Regressor**
7. **Elastic Net Regressor**
8. **Linear Regression**
9. **Voting Regressor**: Combining XGBoost, LightGBM, CatBoost, and Random Forest.

### Hyperparameter Tuning

RandomizedSearchCV was used for hyperparameter tuning of several models. Below are the details of hyperparameter tuning for each:

#### 1. Decision Tree Regressor
- **Best Parameters**: 
  ```json
  {
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_depth": 10,
    "criterion": "absolute_error"
  }
  ```

#### 2. Random Forest Regressor
- **Best Parameters**:
  ```json
  {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "criterion": "squared_error"
  }
  ```

#### 3. AdaBoost Regressor
- **Base Estimator (Decision Tree) Best Parameters**:
  ```json
  {
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_depth": 7
  }
  ```
- **Best AdaBoost Parameters**:
  ```json
  {
    "n_estimators": 50,
    "loss": "square",
    "learning_rate": 0.01
  }
  ```

#### 4. XGBoost Regressor
- **Best Parameters**:
  ```json
  {
    "subsample": 1.0,
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "min_child_weight": 3,
    "max_depth": 5,
    "learning_rate": 0.1,
    "gamma": 0,
    "colsample_bytree": 0.7
  }
  ```

#### 5. LightGBM Regressor
- **Best Parameters**:
  ```json
  {
    "subsample": 0.8,
    "reg_lambda": 0,
    "reg_alpha": 0.5,
    "num_leaves": 100,
    "n_estimators": 500,
    "min_child_samples": 10,
    "max_depth": 10,
    "learning_rate": 0.01,
    "colsample_bytree": 0.6
  }
  ```

#### 6. CatBoost Regressor
- **Best Parameters**:
  ```json
  {
    "depth": 6,
    "iterations": 300,
    "learning_rate": 0.03,
    "l2_leaf_reg": 3
  }
  ```

#### 7. Elastic Net Regressor
- **Best Parameters**:
  ```json
  {
    "alpha": 0.1,
    "l1_ratio": 0.5
  }
  ```

#### 8. Linear Regression
- No hyperparameter tuning was required for Linear Regression.

#### 9. Voting Regressor
- Combines the predictions of the best-tuned models (XGBoost, LightGBM, CatBoost, Random Forest) for an ensemble prediction.

## Results

### Performance Metrics for Each Model

- **Decision Tree Regressor**:
  - Validation RMSE: 69882.79
  - R-squared: 64.72%

- **Random Forest Regressor**:
  - Validation RMSE: 69412.67
  - R-squared: 65.09%

- **AdaBoost Regressor**:
  - Validation RMSE: 69676.99
  - R-squared: 65.10%

- **XGBoost Regressor**:
  - Validation RMSE: 68988.40
  - R-squared: 65.36%

- **LightGBM Regressor**:
  - Validation RMSE: 69044.88
  - R-squared: 65.48%

- **CatBoost Regressor**:
  - Validation RMSE: 69215.12
  - R-squared: 65.28%

- **Elastic Net Regressor**:
  - Validation RMSE: 74110.72
  - R-squared: 61.50%

- **Linear Regression**:
  - Validation RMSE: 74955.38
  - R-squared: 60.76%

- **Voting Regressor (Ensemble)**:
  - Training RMSE: 73917.81
  - Validation RMSE: 68848.18
  - R-squared: 66.00%
  - RMSE Ratio to Mean: 6689.78

### Observations

- **Voting Regressor** combining XGBoost, LightGBM, CatBoost, and Random Forest delivered the best performance with an R-squared of 66.00%.
- Ensemble methods significantly improve predictive performance by combining the strengths of individual models.
- There is still room for improvement.

## Conclusion

The project successfully implemented multiple regression models for predicting car prices, with the Voting Regressor performing best overall. 
Future work could involve exploring more advanced feature engineering, alternative algorithms, and further hyperparameter tuning to refine the model's accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
