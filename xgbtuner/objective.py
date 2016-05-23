
import pdb
import xgboost as xgb

class Loss(object):
    ''' Parameters and methods for evaluating the objective function '''
    def __init__(self, data):
        '''
        Args:
            data (xgboost.Dmatrix): the data on which to tune the parameters
        '''
        self.data = data
    
    def evaluate(self, params):
        '''
        Returns: A dictionary containing the best tried number of rounds and 
            the associated optimum loss function value
        '''
        assert params['num_boost_round']
        nround = params.pop('num_boost_round')
        params.update(self.fixed_params)
        res = xgb.cv(params = params, 
                     dtrain = self.data, 
                     num_boost_round = nround,
                     nfold = self.fixed_params['nfold'],
                     show_progress=False)
        loss = res.ix[:,0]
        wm = loss.idxmin() # generalize this line for convex/concave objectives
        return {'loss':loss[wm], 'wm':wm}

