import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load data
data = pd.read_csv("D:/TY/Sem 6/AML Project/Assignment-2_Data.csv")

# Drop missing values correctly
data = data.dropna()

# Fill missing values for numerical columns
data['age'] = data['age'].fillna(data['age'].median())
data['balance'] = data['balance'].fillna(data['balance'].median())
data['day'] = data['day'].fillna(data['day'].mean())
data['duration'] = data['duration'].fillna(data['duration'].mean())
data['campaign'] = data['campaign'].fillna(data['campaign'].mean())
data['previous'] = data['previous'].fillna(data['previous'].mean())
data['pdays'] = data['pdays'].fillna(data['pdays'].mean())

# Fill missing values for categorical columns
data['education'] = data['education'].fillna(data['education'].mode()[0])
data['default'] = data['default'].fillna(data['default'].mode()[0])
data['housing'] = data['housing'].fillna(data['housing'].mode()[0])
data['loan'] = data['loan'].fillna(data['loan'].mode()[0])
data['contact'] = data['contact'].fillna(data['contact'].mode()[0])
data['month'] = data['month'].fillna(data['month'].mode()[0])
data['poutcome'] = data['poutcome'].fillna(data['poutcome'].mode()[0])
data['y'] = data['y'].fillna(data['y'].mode()[0])

# Drop the ID column
data = data.drop(["Id"], axis=1)

# Convert categorical columns into numeric using Label Encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'loan', 
                       'contact', 'housing', 'month', 'poutcome', 'y']

lb = LabelEncoder()
for column in categorical_columns:
    data[column] = lb.fit_transform(data[column])

# Define predictors and target
predictors = data.columns[:-1]
target = 'y'

# Split data into training and testing sets
train, test = train_test_split(data, test_size=0.3, random_state=42)

# Define model
model = DT(criterion='entropy')

# GridSearchCV
param_grid = {
    'min_samples_leaf': [1, 5, 10, 20],
    'max_depth': [2, 4, 6, 8, 10],
    'max_features': ['sqrt']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='accuracy', n_jobs=-1, cv=5,
                           refit=True, return_train_score=True)

grid_search.fit(train[predictors], train[target])

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Save the model to a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Evaluate the model on the test set
test_predictions = best_model.predict(test[predictors])
print('Test Accuracy:', accuracy_score(test[target], test_predictions))
print(confusion_matrix(test[target], test_predictions))
