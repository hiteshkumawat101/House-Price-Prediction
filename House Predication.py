import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import sklearn.datasets
from sklearn import metrics

# Load the California Housing dataset
house_data = sklearn.datasets.fetch_california_housing()

# Convert the dataset into a DataFrame
house_dataframe = pd.DataFrame(house_data.data, columns=house_data.feature_names)
house_dataframe['Price'] = house_data.target

# Display the first few rows of the DataFrame
print(house_dataframe.head())

# Check the shape of the DataFrame
print(house_dataframe.shape)

# Generate descriptive statistics
print(house_dataframe.describe())

# Check for missing values
print(house_dataframe.isna().sum())

# Compute the correlation matrix
corre = house_dataframe.corr()
print(corre)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(corre, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 0.8}, cmap='Greens')

# Separate features and target variable
X = house_dataframe.drop('Price', axis=1)
Y = house_dataframe['Price']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
print(x_train.shape, x_test.shape)

# Initialize the XGBRegressor model
model = XGBRegressor()

# Train the model
model.fit(x_train, y_train)

# Predict on the training set
predict_train = model.predict(x_train)
print(predict_train)

# Evaluate the model on the training set using R-squared
print(metrics.r2_score(y_train, predict_train))

# Predict on the testing set
predict_test = model.predict(x_test)
print(predict_test)

# Evaluate the model on the testing set using R-squared
print(metrics.r2_score(y_test, predict_test))
