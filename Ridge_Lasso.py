""" 

Ridge and Lasso 

Used for 'large' number of features.

1) Large enough to enhance the tendency of a model to overfit (as low as
10 variables might cause overfitting.)

2) Large enough to cause computational challenges. millions/billions of features

They work by penalizing the 
magnitude of coefficients of features along with minimizing the error between
predicted and actual observations. 

Ridge Regression:
    performs L2 regularization, adds
    penalty equivalent to square of the magnitude of
    coefficients.
Lasso Regression:
    L1 regularization, adds penalty to absolute value
    of the magnitude of coefficients. 


# WHY PENALIZE THE MAGNITUDE OF COEFFICIENTS ?  

"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize']= 12, 10
        
    # Define input array with angles 60 - 300 in radians
    
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10) # for reproducability.
#
y = np.sin(x) + np.random.normal(0,0.15,len(x))
#
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

# Polynomial Regression Example form powers 1 to 15.

for i in range(2,16): #Power of 1 is already there
    colname = 'x_%d'%i # new var will be x_power
    data[colname] = data['x']**i
#print data.head()

"""
 lets make 15 different linear regression models.
with each model containing variables with powers of x from 1 to the particular 
model number. 

"""
# Import linear regression libraries
from sklearn.linear_model import LinearRegression

def linear_regression(data,power,models_to_plot):
    #initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
        
    # Fit model 
    
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    
    #check if a plot is to be made for entered power
    
    if power in  models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for Power: %d'%power)
        
    # return the result in pre-defined format
    
    rss = sum((y_pred - data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret

""" 
Note that this function will not plot the model 
fit for all the powers but will return the RSS and 
coefficients for all the models.
"""

"""
Now, we can make all 15 models and compare the results.
store in a Pandas dataframe and plot 6 models to get an idea of the trend.
"""
# Initialize the data frame to store results.

col = ['rss','intercept'] + ['coef_x%d'%i for i in range(1,16)]

ind = ['model_pow_%d'%i for i in range(1,16)]

#coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

#Define powers for which a plot is required.

#models_to_plot = {1:231, 3:232, 6:233, 9:234, 12:235, 15:236}

# Iterate through all powers and assimilate results
#for i in range(1,16):
#    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data,power=i,models_to_plot=models_to_plot)





# AS model complexity increases, the model fits even smaller deviations
#in the training data set. This leads to over-fitting. 

# Set display format to be scientific

#pd.options.display.float_format = '{:,.2g}'.format
# print(coef_matrix_simple)

""" 
We see that the size of the coefficients increase exponentially with
increase in model complexity. 

That's why we should put a constraint on magnitude of coefficients.
 
A large coefficient means we are putting a lot of emphasis 
on that feature. Meaning it is a good predictor for the outcome.
"""
# --------

""" RIDGE 
alpha = 0 - same as simple linear regression
alpha - infinity - coeffs will be 0
0 < alpha < infinity , between 0 and simple lin regress

"""

from sklearn.linear_model import Ridge

def ridge_regression(data, predictors, alpha, models_to_plot={}):
    # Fit model
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for entered alpha
    
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('plot for alpha: %.3g'%alpha)
    
    #Return result in defined format.
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret

""" NOTE: Each of these 10 models will contain all the 15 variables and only
the value of alpha would differ. Different from linear regression, which would have
a subset of features.. """

# initialize predictors to be set of 15 powers of x.
#predictors = ['x']
#predictors.extend(['x_%d'%i for i in range(2,16)])

#Set different values of alpha to be tested.
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

#Make data frame to store coefficients.

#col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
#ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
#coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

#models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
#for i in range(10):
#    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i],models_to_plot) 

"""

As alpha increases, model complexity decreases. Higher values
of alpha reduce overfitting (ex: alpha=5).

   Cross validation - the value of alpha is iterated over a range of
   values and the one giving the higher CV score is chosen.
"""
# Set display format to table for values of alpha-coeff to be seen.

#pd.options.display.float_format = '{:,.2g}'.format
#print(coef_matrix_ridge)

""" We can see the following:
    1. The RSS increases with increase in alpha, the model complexity dec.
    2. Alpha as small as 1e-15 gives us significant reduction in magnitude
    of coefficients. How? compare with simple lin regression table.
    3. High alphas lead to significant underfitting.
    4. coefficient are NOT ZERO!! 
    
    lets confirm that.. """
#coef_matrix_ridge.apply(lambda x: sum(x.values==0), axis=1)
    
    # shows that all 15 coefficients are greater than zero.
    
    #### _ _ __ ### _ _  ##### ____##  NEXT ..
    
    
    
    
""" LASSO 

alpha = 0 - same as simple linear regression
alpha - infinity - coeffs will be 0
0 < alpha < infinity , between 0 and simple lin regress
"""
from sklearn.linear_model import Lasso

def lasso_regression(data,predictors, alpha, models_to_plot={}):
    # Fit Model
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    # Check if a plot is needed for alpha..
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    # return result in format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

# Initialize predictors to all 15 powers of x
predictors = ['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

# Define alpha values to test
alpha_lasso=[1e-15,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1,5,10]

#Initialize dataframe to store coefficients. 
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

# Define models to plot
models_to_plot = {1e-10:231, 1e-5:232, 1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#iterate over the 10 alpha values

for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data,predictors,alpha_lasso[i],models_to_plot)
    

    
# This again tells un model complexity decreases with
# increase alpha values. Lets consider further analysis
pd.options.display.float_format = '{:,.2g}'.format
print(coef_matrix_lasso)

# higher RSS for higher values of alpha.
"""    Observations
1. For same values of alpha, coefficients of lasso regression
are much smaller compared to Ridge, compare row 1 of the tables.

2. for same alpha, lasso has higher RSS (poorer fit) as compared
to ridge.

3. Many coefficients are zero even for small alpha values.
Higher sparsity.
1,2, might not generalize always but will hold for many cases.
The real difference from ridge is coming out in the last
inference. 
"""
coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis = 1)


# Check correlation with data.corr()



# OBSERVE 

""" Even a small value of alpha, a significant # of coeff's
are zero. This explains the baseline model for alpha=1, the 
horizontal line.
"""



"""
Use Cases 
Ridge....It is used to prevent overfitting. It includes all the
features is computationally expensive.
Lasso.... Since it provides sparse solutions, it is generally
the model of choice, and performs feature selection.
"""

"""
Presence of highly correlated features.

Ridge: It generally works well even in presence of
highly correlated features as it will include all of them
in the model, but the coefficients will be distributed among
them depending on the correlation.

Lasso: It arbitrarily selects any one feature among
the highly correlated ones and reduced the coefficients
of the rest to zero. Also, the chosen variable changes
randomly with change in model parameters. This
generally doesnt work that well compared to Ridge. 
