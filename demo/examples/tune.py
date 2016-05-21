
'''
Minimal example of parameter tuning
'''
import xgbtuner as tuner
import pkg_resources
pkg_resources.resource_filename('xgbtuner', 'data/fake_multinomial_data.csv')
import pandas as pd
import xgboost as xgb
#import os
import operator

# Load sample data
df = pd.read_csv('../data/fake_multinomial_data.csv')
xdat = xgb.DMatrix(
    data = df.drop('target', axis=1).values,
    label= df['target'].values
    )

# Define xgb parameters
nfold = 5
nround = 20
n_early_stop = 20
param = {
    'bst:max_depth':6, 
    'bst:eta':0.3,
    'bst:min_child_weight':3,
    'objective':'multi:softprob',
    'eval_metric':'mlogloss',
    'num_class':3
}

# Do xgboost cross validation to estimate how well xgboost will perform for this
#   dataset and the chosen parameters
res = xgb.cv(param, xdat, nround, nfold)
loss = res.ix[:,0]
wm = loss.idxmin()
loss[wm]
wm

# Fit the final xgboost model
gbm = xgb.train(param, xdat, num_boost_round = wm, verbose_eval=False)

# Use the model to make predictions on new data
predictions = gbm.predict(xdat)


