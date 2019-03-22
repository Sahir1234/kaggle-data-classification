# This is the second attempt to complete the data classification challenge.
# In this attempt, I will use a K-nearest neighbors classifier. This
# method will probably not be as accurate as the logistic regression we used 
# in the first attempt because there we are analyzing too many features.
# However, since the point of this exercise is to experiment with the
# different classifiers, we will try it anyway.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report


# We read the csv files and store them in dataframes. train_data contains the
# data we will use to train our model. pred_data is the data we have to use
# the model and get legitimate results for.
train_data = pd.read_csv('../input/train.csv')
pred_data = pd.read_csv('../input/test.csv')


# We first isolate the features and the outputs of the training data
# into X and y as numpy arrays, respectively.
X = train_data.drop('PAID_NEXT_MONTH', axis = 1).values
y = train_data['PAID_NEXT_MONTH'].values


# We then isolate the data we need to make a prediction into another numpy array.
X_pred = pred_data.drop('PAID_NEXT_MONTH', axis = 1).values


# To prevent overfitting we need to split the training data into training and test sets,
# which can also be used to evaluate the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 52)


error = []
smallestError = 1
neighbors = 1


# To figure out what the ideal number of neighbors would be to use with
# our KNearestNeighbors classifier, we loop through many value, fit a model,
# and predict results to get error values. We also save the number of neighbors
# where the mean error is the least in the variable neighbors.
for i in range(1,60):
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    currentError = np.mean(pred != y_test)
    error.append(currentError)
    if(smallestError > currentError):
        smallestError = currentError
        neighbors = i


# Here, we graph the error list we just filled to visualize which value of
# K is produces the smallest error in the model predictions.
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 60), error, color='black', marker='o', markerfacecolor='red', markersize=5)
plt.title('Error Based on K-Value')  
plt.xlabel('K Value')  
plt.ylabel('Error')


# We instantiate the model and set the number of meighbors to the value
# that will produce the smallest error.
model = KNeighborsClassifier(n_neighbors = neighbors)
model.fit(X_train , y_train)


# The model score will tell us roughly how accurate the model is on
# unfamiliar data.
print(model.score(X_test,y_test))


# We now use the model to make prediction on the test data and then
# print the classification report based on this prediction. This gives us
# more detailed information about how well the model classifies data to the
# two different classes.
y_test_pred = model.predict(X_test)
print(classification_report(y_test, y_test_pred))


# Finally, we make a prediction on the actual data that we need to analyze.
y_pred = model.predict(X_pred)


# We now have to set up an output csv file for submission. We simply take
# the sample submission, convert it to a dataframe, override the last
# column with our own model's output, and then convert it back to a csv.
df = pd.read_csv('../input/sample-submission.csv')
df['PAID_NEXT_MONTH'] = y_pred
df.to_csv('submission.csv',index=False)

# After running this and submitting the output file to the competition, we found that 
# this method has an approximately 78% accuracy, which is better than expected.
