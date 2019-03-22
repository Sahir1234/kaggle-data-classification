# This code was written for a Kaggle Data Classification Challenge.
# This is the first attempt using a logistic regression algorithm. There
# is a second attempt where I used a K-Nearest Neighbors classifier to
# accomplish the task. The goal of this challenge is to predict whether a
# customer will make a payment based on their payment history and other
# demographic data. This is a binary classification problem and the training
# and testing data are stored in csv files.


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale


# We read the data we need to use for training and for our final
# submission into 
train_data = pd.read_csv('../input/train.csv')
pred_data = pd.read_csv('../input/test.csv')


X = train_data.drop('PAID_NEXT_MONTH', axis = 1).values
y = train_data['PAID_NEXT_MONTH'].values
X_pred = pred_data.drop('PAID_NEXT_MONTH', axis = 1).values


# Scale the training data so that the model can learn faster and fit the 
# data more easily.
X_scaled = scale(X)


# Instnatiate the model as a Logistic Regression model.
model = LogisticRegression()


# We use cross validation scoring to perform a train_test_split 5 different split
# possibilities and score each of them. Then, we take the mean of these scores to 
# get a fairly accurate metric of how well the model performs.
print(cross_val_score(model, X, y, scoring = 'accuracy', cv = 5).mean())


# To prevent overfitting, we split the data into train and test sets and fit
# the model with the training sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 52)
model.fit(X_train,y_train)


# We actually use the model to predict the results based on the unfamiliar test data.
y_pred = model.predict(X_pred)


# To submit a csv to the competition, I converted the sample submission to a 
# dataframe, overwrote the last column with my own model's prediction, and
# then converted the dataframe back to a 
df = pd.read_csv('../input/sample-submission.csv')
df['PAID_NEXT_MONTH'] = y_pred
df.to_csv('submission.csv',index=False)


# Upon submitting this model, it gave a score of approximately 80% accuracy
# on data it had never seen before. This is pretty good, although it could
# still be improved further.
