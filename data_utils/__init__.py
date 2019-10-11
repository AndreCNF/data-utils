__version__ = '0.1.0'

from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations

# [TODO] Check if the random seed is working properly
# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
# Automatic seed
random_seed = np.random.get_state()
np.random.set_state(random_seed)
torch.manual_seed(random_seed[1][0])

# Methods

def set_random_seed(num):
    '''Set a user specified seed to use in stochastic (i.e. random) processes.
    This method should be called before importing packages which use a
    random seed.

    Parameters
    ----------
    num : int
        Number that will serve as the random seed.

    Returns
    -------
    None
    '''
    global random_seed
    random_seed = num
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    return
