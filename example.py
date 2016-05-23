
# coding: utf-8

# ## Example:  tuning xgboost for multinomial regression

# Here is a basic example of using `xgbtuner` to optimize the parameters of an xgboost instance that does multinomial prediction. First import some modules and load the example data:

# In[3]:

import xgbtuner
import pkg_resources
import pandas as pd
import xgboost as xgb

f = pkg_resources.resource_filename('xgbtuner', 'data/fake_multinomial_data.csv')
df = pd.read_csv(f)
xdat = xgb.DMatrix(
    data = df.drop('target', axis=1).values,
    label= df['target'].values
    )


# To search for the best xgboost parameters, you first have to decide what "best" means by setting up a loss function to be minimized (using the negative of the objective function if bigger is better):

# In[4]:

loss = xgbtuner.Loss(xdat)


# `Loss` is a class that holds the data and all model parameters and includes a method to evaluate the cross validation score for a given objective function.  By default, `Loss` uses the default xgboost model, which uses a tree as the base learner and minimizes the mean squared error of the predictions (treating the response as numeric, not multinomial/categorical!).  To use `Loss` with multinomial regression, therefore, it is essential to change at least some of the defaults.
# 
# `Loss` groups the loss function parameters into two types:
# * fixed_params: Overrides xgboost defaults.  Any specified values will be held constant throughout the tuning process.
# * tuning_params: Specifies any parameters that should be tuned.
# 

# In[5]:

fixed_params = {
    'nfold':5,
    'n_early_stop':20,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3
}
# Evaluate the objective on a set of params:
params = {
    'num_boost_round':10,
    'bst:max_depth':6, 
    'bst:eta':0.3,
    'bst:min_child_weight':3,
}
loss.evaluate(params)

# 
# 
# # Fit the final xgboost model
# gbm = xgb.train(param, xdat, num_boost_round = wm, verbose_eval=False)
# 
# # Use the model to make predictions on new data
# predictions = gbm.predict(xdat)


# In[ ]:



