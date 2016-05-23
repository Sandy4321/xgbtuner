
class Tuner(object):
    '''Proposes new parameter settings, taking into account any past performance'''
    
    def __init__(self, loss, fixed_params = None, tuning_params = None):
        '''
        Args:
            loss (xgbtuner.Loss): A loss function class with an 'evaluate'
                submethod that accepts a parameter dictionary and
                returns a performance value
            fixed_params (dict): values of xgboost parameters; these override
                xgboost defaults
            tuning_params (dict): initial values for xgboost parameters to tune
        '''
        self.loss = loss
        params = load_xgb_default_parameters()
        
        self.fixed_params = fixed_params
        self.which_to_tune = tuning_params.keys()
        self.results = pd.DataFrame(params)
    
    def propose(self):
        # function of self.which_to_tune
        self.tuning_params = self.tuning_params # the interesting part eventually
        
    def evaluate(self):
        return self.loss(parameters)
    
    def iterate(self, n):
        '''Perform n iterations of propose(), evaluate(), while saving results'''
        while n:
            print("iteration " + str(n))
            self.propose()
            self.evaluate()
            n += -1
        
    
