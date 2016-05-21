class Objective(object):
    ''' Parameters and methods for evaluating the objective function '''
    def __init__(self, data, fixed_params, old_results = None):
        '''
            Args:
                data (xgboost.Dmatrix): the data on which to tune the parameters
                fixed_params (dictionary):  parameters that will not be tuned
                old_results (string): the path/name of a file containing previous
                    objective evaluations
            '''
        self.data = data
        self.fixed_params = fixed_params
        if old_results:
            self.results = pd.read_csv(old_results)
    
    def evaluate(params):
        '''
            Args:
                params (dictionary):  parameters over which to tune
            
            Returns: A dictionary containing the best tried number of rounds and 
                the associated optimum loss function value
            '''
        nround = params.pop('nround')
        res = xgb.cv(params = params, 
                     dtrain = self.data, 
                     nround = nround,
                     nfold = self.fixed_params['nfold'])
        loss = res.ix[:,0]
        wm = loss.idxmin()
        return {'loss':loss[wm], 'wm':wm}

