
'''
Minimal example of parameter tuning
'''
import pandas as pd
import xgboost as xgb
import os
import operator

# Prepare the training data
home = os.path.expanduser("~")
datdir = os.path.join(home, 'Documents/kagdata/santander/')

train_df = pd.read_csv(datdir + 'rawdata/train.csv')
test_df = pd.read_csv(datdir + 'rawdata/test.csv')
n_train = train_df.shape[0]

# Feature engineering
targ = train_df['TARGET'].values # make TARGET the final columns
train_df.drop('TARGET', axis=1, inplace = True)
all_df = train_df.append(test_df)
all_df.index = range(all_df.shape[0])
stats_df = pd.DataFrame({'mean': all_df.mean(axis=1)})
stats_df['var'] = all_df.var(axis=1)
stats_df['zeros'] = (all_df == 0).astype(int).sum(axis=1)
# stats_df['median'] = all_df.median(axis=1) ALWAYS ZERO!
stats_df['skew'] = all_df.skew(axis=1)
stats_df['kurtosis'] = all_df.kurtosis(axis=1)
stats_df['max'] = all_df.max(axis=1)
#stats_df.describe()

norm_df = (all_df - all_df.mean())/all_df.std()
norm_stats_df = pd.DataFrame({'norm_mean': norm_df.mean(axis=1)})
norm_stats_df['norm_var'] = norm_df.var(axis=1)
norm_stats_df['norm_median'] = norm_df.median(axis=1)
norm_stats_df['norm_skew'] = norm_df.skew(axis=1)
norm_stats_df['norm_kurtosis'] = norm_df.kurtosis(axis=1)
norm_stats_df['max'] = norm_df.max(axis=1)
#norm_stats_df.describe()

#col_sets = []
np.random.seed(34)
normax_df = pd.DataFrame()
for k in range(100):
    cols = list(np.random.choice(all_df.columns, 40, replace=False))
    normax_df['normax_' + str(k)] = norm_df[cols].max(axis = 1)

all_df = pd.concat([all_df, stats_df], axis = 1)
all_df = pd.concat([all_df, norm_stats_df], axis = 1)
all_df = pd.concat([all_df, normax_df], axis = 1)

train_df = all_df.iloc[0:n_train].copy()
test_df = all_df.iloc[n_train::].copy()
train_df['TARGET'] = targ

dtrain = xgb.DMatrix(
    data = train_df.drop('TARGET', axis=1).copy().values,
    label= train_df['TARGET'].copy().values
    )

# Define xgb parameters
nfold = 5
nround = 120
n_early_stop = 20
param = {
    'bst:max_depth':6, 
    'bst:eta':0.06,
    'bst:min_child_weight':3,
    'bst:lambda':0.25,
    'bst:alpha':0,
    'eval_metric':'auc', 
    'objective':'binary:logistic'
    }

# Do xgboost cross validation to estimate how well xgboost will perform for this
#   dataset and the chosen parameters
res = xgb.cv(param, dtrain, nround, nfold)
wm = res['test-auc-mean'].idxmax()
res['test-auc-mean'][(wm - 10):(wm + 10)]
print('expecting out-of-sample performance to be around '
        + str(res['test-auc-mean'][wm]))
# actual best turned out to be around 0.838

# Fit the final xgboost model
gbm = xgb.train(param, dtrain, num_boost_round = wm, 
    evals = [(dtrain,'train')],
    early_stopping_rounds = n_early_stop)

# Insight:  which variables are most important?
imps = gbm.get_fscore()
imps = sorted(imps.items(), key=operator.itemgetter(1))
pd.DataFrame(imps).sort(0)

# Use the model to make predictions on new data
dnew = xgb.DMatrix(data = test_df.values)
predictions = gbm.predict(dnew, ntree_limit=gbm.best_ntree_limit)

# Save predictions in the format required for a Kaggle submission
exam_df = pd.read_csv(datdir + 'rawdata/sample_submission.csv')
exam_df['TARGET'] = predictions
exam_df.to_csv(datdir + 'submissions/xgboost5.csv', index = False)



