__version__ = '0.1.0'

from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
from importlib import reload                            # Allows to reload (import again) modules, which make them rerun their initialization

# [TODO] Check if the random seed is working properly
# Random seed used in PyTorch and NumPy's random operations (such as weight initialization)
# Automatic seed
random_seed = np.random.get_state()
np.random.set_state(random_seed)
torch.manual_seed(random_seed[1][0])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Boolean that sets whether to use the original Pandas library or the Modin distributed version
use_modin = True

from . import utils                                     # Generic and useful methods
from . import datasets                                  # PyTorch dataset classes
from . import search_explore                            # Methods to search and explore data
from . import data_processing                           # Data processing and dataframe operations
from . import padding                                   # Padding and variable sequence length related methods
from . import embedding                                 # Embeddings and other categorical features handling methods
from . import deep_learning                             # Common and generic deep learning related methods
from . import machine_learning                          # Machine learning focused pipeline methods

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


def set_pandas_library(lib='modin'):
    global use_modin
    if lib.lower() == 'modin':
        use_modin = True
    elif lib.lower() == 'pandas':
        use_modin = False
    else:
        raise Exception(f'ERROR: {lib} is an invalid pandas library. Must either use `pandas` or `modin`.')
    # Reload the modules, to update their pandas package
    reload(utils)
    reload(datasets)
    reload(search_explore)
    reload(data_processing)
    reload(padding)
    reload(embedding)
    reload(deep_learning)
    reload(machine_learning)
