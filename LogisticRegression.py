"""
Logistic Regression
: Titanic Dataset from Kaggle
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

rcParams['figure.figsize'] = 10, 8
sb.set_style('whitegrid')

# Read the data set in

url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'

# naming all columns
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId', 'survived', 'Pclass', 'Name',
                   'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                   'Cabin', 'Embarked']
# titanic.head()    #See first 5 data points.

#Check that our target variable is binary
#sb.countplot(x='survived', data=titanic,palette='hls')

#Check for missing values.
# .sum() adds up the missing values,
titanic.isnull().sum()

# titanic.info() gives us the number of records in the data frame for each variable

# We see age is missing 177 values. Cabin is missing almost all the values
# so we will drop that variable!. Age intuitively is a good predictor so 
# we will keep it in the model. We need to fill those values in. 

# Dropping variables that don't help for prediction.
titanic_data = titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 1)
# titanic_data.head() #shows first 5 for reduced data

# See how age is related to class.

#sb.boxplot(x='Pclass',y='Age',data=titanic_data,palette='hls')

# Fill all null values in age with the average age value based on the classes.

def age_approx(cols):
    Age = cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
#print(titanic_data.isnull().sum())

# 2 missing values in embarked, we can drop those.
titanic_data.dropna(inplace=True)
# titanic_data.isnull().sum()  #Check to see if they were dropped. Good!

                     # Convert Categorical Variables to dummy indicators.
# Sex and Embarked Variables.

gender = pd.get_dummies(titanic_data['Sex'], drop_first=True)
#gender.head()
embark_location = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
#embark_location.head()
# Drop dummies from titanic data set and add new ones back into it.

titanic_data.drop(['Sex', 'Embarked'],axis=1, inplace=True)
#titanic_data.head()

titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
# titanic_dmy.head()

"""Checking for independence within variables"""

sb.heatmap(titanic_dmy.corr())

# Fare and Pclass are not independent so drop these variables!
titanic_dmy.drop(['Fare','Pclass'], axis=1, inplace=True)
#titanic_dmy.head()

# We now have 6 predictive variables and rule is have at least
# 50 records per feature, so we need 300.

titanic_dmy.info()   #WE have 889

X = titanic_dmy.ix[:,(1,2,3,4,5,6)].values
y = titanic_dmy.ix[:,0].values
                  
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.3, random_state=25)
              
# Deploy - Evaluate 

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print(classification_report(y_test, y_pred))
