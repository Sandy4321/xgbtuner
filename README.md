## xgbtuner

[xgboost](https://github.com/dmlc/xgboost) is a leading implementation of gradient boosting machines.  UNDER CONSTRUCTION:  xgbtuner offers a hybridization of grid search, random search, gradient descent, and Bayesian optimization methods to help find the parameters of the xgboost algorithm that lead to optimal predictions for a given dataset.

## Background and references

As of 2016/04/30, the official [xgboost notes on parameter tuning](https://github.com/dmlc/xgboost/blob/master/doc/param_tuning.md) call the tuning problem a "dark art", and offer only a few heuristics to guide the practicioner. [Wikipedia](https://en.wikipedia.org/wiki/Hyperparameter_optimization) summarizes various approaches to parameter tuning, and [several parameter tuning packages](http://fastml.com/optimizing-hyperparams-with-hyperopt/) exist for general-purpose parameter tuning applications.  xgbtuner will draw from many of these to design a tuner specific to xgboost.


## Notes

Hyperparameter tuning is only a small subset of the model building problem.  Don't spend to much time on this before you've considered other ways to boost performance such as feature engineering, model averaging, or trying entirely different models.  