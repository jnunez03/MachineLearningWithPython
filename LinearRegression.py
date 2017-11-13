import statsmodels.api as sm 
from sklearn import datasets #Imports data sets
import numpy as np
import pandas as pd
from sklearn import linear_model

data = datasets.load_boston() #loads boston dataset

                           

print(data.DESCR)  #only works for sklearn datasets.

     
print(data.feature_names) # Column names of variables
print(data.target)  
"""
this shows CRIM to LSTAT , which means they already set 
the house value/price as target variable
others are predictors.
"""
 # Define the data/predictors as pre-set feature names
df = pd.DataFrame(data.data, columns = data.feature_names)

# Put the target (housing value - MEDV)
# in another data frame

target = pd.DataFrame(data.target, columns=["MEDV"])

# We want to check which variables will be
# good predictors. * Check correlations
# by plotting the data and searching for visual 
#by relationship,

# We'll use LSTAT and RM, statsmodels does not 
# add a constant by default.
# we will check it without a constant.
#-----
X = df["RM"]
y = target["MEDV"]
# NOTE Difference in argument order
# sm.OLS(output, input)
model = sm.OLS(y,X).fit()
predictions = model.predict(X) # Use model

  #Print stats
print(model.summary())

#TABLE
"""
df = degrees of freedom
the number of variables that are free to vary.

coefficient 3.634 means as RM increases by 1,
the predicted value of MDEV increases
by 3.634.
- Rsquared- % of variance model explains
std err is deviation of samp dist
of statistic, most commonly the mean.
- t score, p-value.
pval is significant
"""
# X = sm.add_constant(X) x -> name of dataFrame
"""
with constant    v v v v v v v v v v v v
"""  
X = df["RM"] # X is input var
y = target["MEDV"] ## Y output var

        # Add intercept Beta_0 to model
X = sm.add_constant(X)

model = sm.OLS(y,X).fit()
predictions = model.predict()
print(model.summary())
"""
Now we have an intercept at -34.6706, slope changes from
3.634, to 9.1021.
"""

#     *  We will try fitting a regression model with more
#        then 1 variable.

X = df[["RM", "LSTAT"]]
y = target["MEDV"]

model = sm.OLS(y,X).fit()
predictions = model.predict(X)
print(model.summary())

"""
R-squared is higher.
RM inc by 1, MEDV inc by 4.9
LSTAT inc by 1, MEDV dec by -.65

"""
X = df    # using all variables
y = target["MEDV"]

lm  = linear_model.LinearRegression()

model = lm.fit(X,y)

predictions = lm.predict(X)
print(predictions)[0:5]  #print first 5 predictions for y

print(lm.score(X,y))  #rsquared
print(lm.coef_, lm.intercept_)

""" Testing/Training Cross Validation
Just uncomment these imports ,
"""
#import pandas as pd
#from sklearn import datasets, linear_model
#from sklearn.model_selection import train_test_split
#from matplotlib import pyplot as plt

#   LOAD DATASET 
columns = "age sex bmi map tc ldl hdl tch ltg glu".split() #columns
diabetes = datasets.load_diabetes() 
df = pd.DataFrame(diabetes.data, columns = columns)
y = diabetes.target  #define target variable

#Create training and testing vars

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = .2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Fit a model
lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

print(predictions[0:5]) # Only first 5 are predicted
## THE LINE / MODEL 

plt.scatter(y_test,predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print("Score: ", model.score(X_test,y_test))
# .4389
"""

1) loaded data, split it into train and test, fitted regression model on
the training data, made predictions based on this data and tested
the predictions on the test data. 

:What if the split was not random. what if one subset of our
data has only people from a certain state, employees with a certain
income level, or imagine a file ordered by one of these. this will
result in overfitting, even though we are trying to avoid it.
"""

# CROSS VALIDATION !
""" We split data into k subsets, train on k-1 of the subsets, and 
test on last subset.
2 methods.
K - Folds Cross Validation and Leave one out Cross Validation.
"""
# K-Folds.
# split our data in k subsets(folds).
#Use K-1 subsets to train our data, leave last one out.

#LOOCV 
# # of folds = number of observations we have in the dataset.
""" we then average all of these folds and build our model
with the average. We then test on the last fold. This
is very computational, should be used on small datasets. 

%% . .  The more folds we have we will reduce error due to bias,
but increase the error due to variance. also computation increases
with folds. less folds we reduce error due to variance, but
error due to bias would increase. K=3 is best for big datasets.
smaller datasets, use LOOCV.
"""

# NECESSARY IMPORTS ..     JUST UNCOMMENT THEM 
#from sklearn.cross_validation import cross_val_score, cross_val_predict
#from sklearn import metrics

# Perform 6-fold CV
scores = cross_val_score(model, df, y, cv=6)
print("Cross Val Score:", scores)

# It improved from previously .4389 to .569
# Make CV predictions.
predictions = cross_val_predict(model, df, y, cv=6)
plt.scatter(y,predictions)

# Has 6 times as many points because CV = 6.
accuracy = metrics.r2_score(y, predictions)
print(accuracy)
# .4908
