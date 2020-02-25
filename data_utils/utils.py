from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm.auto import tqdm                              # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

# Methods

def apply_dict_convertion(x, conv_dict, nan_value=0):
    '''Safely apply a convertion through a dictionary.

    Parameters
    ----------
    x : anything
        Object that will be converted through the dictionary.
    conv_dict : dict
        Dictionary used to convert the input object.
    nan_value: anything
        Value or object that repressents missingness.

    Returns
    -------
    x : anything
        Converted object.
    '''
    # Check if it's a missing value (NaN)
    if isinstance(x, numbers.Number):
        if np.isnan(x):
            return nan_value
    # Must be a convertable value
    else:
        return conv_dict[x]


def invert_dict(x):
    '''Invert a dictionary, switching its keys with its values.

    Parameters
    ----------
    x : dict
        Dictionary to be inverted

    Returns
    -------
    x : dict:
        Inverted dictionary
    '''
    return {v: k for k, v in x.items()}


def replace_dict_strings(dct, str_to_replace=';', new_str='_', replace_keys=True,
                         replace_vals=True, inplace=False):
    '''Replace strings in a dictionary, in keys and/or values, with a new,
    desired string.

    Parameters
    ----------
    dct : dict
        Dictionary that will have its keys and/or values modified.
    str_to_replace : str, default ';'
        String to replace with a new one.
    new_str : str, default ';'
        String to replace the old one.
    replace_keys : bool, default True
        If set to True, the dictionary's keys will have their strings edited
        according to the string replacement set by the user.
    replace_values : bool, default True
        If set to True, the dictionary's values will have their strings edited
        according to the string replacement set by the user.
    inplace : bool, default False
        If set to True, the original dictionary will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dictionary.

    Returns
    -------
    data_dct : dict:
        Inverted dictionary
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_dct = dct.copy()
    else:
        # Use the original dataframes
        data_dct = dct
    if replace_keys is True:
        for key in dct.keys():
            # Replace undesired string with the new one
            new_key = str(key).replace(str_to_replace, new_str)
            if new_key != key:
                # Remove the old key and replace with the new one
                dct[new_key] = dct.pop(key)
    if replace_vals is True:
        for key, val in dct.items():
            # Replace undesired string with the new one
            new_val = str(val).replace(str_to_replace, new_str)
            if new_val != val:
                # Replace the old value with the new one, in the same key
                dct[key] = new_val
    return data_dct


def merge_dicts(dict1, dict2=None):
    '''Merge two or more dictionaries into one. The second dictionary can
    overwrite the first one if there are overlapping keys.

    Parameters
    ----------
    dict1 : dict or list of dicts
        Dictionary 1 that will be merged with dictionary 2 or list of
        dictionaries that will be merged.
    dict2 : dict, default None
        Dictionary 2 that will be merged with dictionary 1. If not specified,
        the user must define a list of dictionaries in parameter `dict1` to merge.

    Returns
    -------
    dict3 : dict
        New dictionary resulting from the merge.
    '''
    if isinstance(dict1, dict):
        if dict2 is not None:
            if isinstance(dict2, dict):
                # Merge the two input dictionaries
                return {**dict1, **dict2}
            else:
                raise Exception(f'ERROR: When `dict1` is specified as a single dictionary, the second argument `dict2` must also be a dictionary. Instead, received `dict2` of type {type(dict2)}.')
        else:
            raise Exception(f'ERROR: When `dict1` is specified as a single dictionary, the second argument `dict2` must also be set.')
    elif isinstance(dict1, list):
        # Initialize the new dictionary with the first one on the list
        new_dict = dict1[0]
        for i in range(len(dict1)):
            try:
                # Try to merge with the next dictionary, if there is any
                new_dict = {**new_dict, **dict1[i+1]}
            except:
                break
        return new_dict
    else:
        return Exception(f'ERROR: The first parameter `dict1` must be set as either a dictionary or a list of dictionaries. Instead, received `dict1` of type {type(dict1)}.')


def is_definitely_string(x):
    '''Reports if a value is actually a real string or if it has some number in it.

    Parameters
    ----------
    x
        Any value which will be judged to be either a real string or numeric.

    Returns
    -------
    boolean
        Returns a boolean, being it True if it really is a string or False if it's
        either numeric data or a string with a number inside.
    '''
    if isinstance(x, int) or isinstance(x, float):
        return False

    try:
        float(x)
        return False

    except Exception:
        return isinstance(x, str)


def is_string_nan(x, specific_nan_strings=[]):
    '''Indicates if a string corresponds to a missing value.

    Parameters
    ----------
    x : string
        A string that will be compared with possible missing value
        representations.
    specific_nan_strings : list of strings, default []
        Parameter where the user can specify additional strings that
        should correspond to missing values.

    Returns
    -------
    boolean
        Returns a boolean, being it True if the string corresponds to a missing
        value representation or False if it doesn't.
    '''
    # Only considering strings for the missing values search
    if isinstance(x, str):
        # Considering the possibility of just 3 more random extra characters
        # in NaN-like strings
        if (('other' in x.lower() and len(x) < 9)
            or ('null' in x.lower() and len(x) < 7)
            or (x.lower() == 'nan')
            or ('discrepancy' in x.lower() and len(x) < 14)
            or all([char == ' ' for char in x])
            or all([char == '_' for char in x])
            or all([char == '.' for char in x])
            or ('unknown' in x.lower())
            or ('not obtainable' in x.lower())
            or ('not obtained' in x.lower())
            or ('not applicable' in x.lower())
            or ('not available' in x.lower())
            or ('not evaluated' in x.lower())
            or (x in specific_nan_strings)):
            return True
        else:
            return False
    else:
        warnings.warn(f'Found a non string value of type {type(x)}. As we\'re \
                        expecting a string, any other format will be considered \
                        a missing value.')
        return True


def get_full_number_string(x, decimal_digits=0):
    '''Gets a full number's representation in a string.
    Particularly useful when one has very large float values,
    possibly too big to be represented as an integer.

    Parameters
    ----------
    x : float or double or int
        A numeric value that one wants to represent in a string,
        with all it's numbers visible.
    decimal_digits : int, default 0
        Number of decimal digits to account for in the number.
        Considering the value as a natural number, without
        decimals, by default.

    Returns
    -------
    x : string
        A numeric value that one wants to represent in a string,
        with all it's numbers visible.
    '''
    return f'{x:.{decimal_digits}f}'


def in_ipynb():
    '''Detect if code is running in a IPython notebook, such as in Jupyter Lab.'''
    try:
        return str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"
    except Exception:
        # Not on IPython if get_ipython fails
        return False


def iterations_loop(x, see_progress=True):
    '''Determine if a progress bar is shown or not.'''
    if see_progress is True:
        # Use a progress bar
        return tqdm(x)
    else:
        # Don't show any progress bar if see_progress is False
        return x


def convert_dataframe(df, to='pandas', return_library=True):
    '''Converts a dataframe to the desired dataframe library format.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.DataFrame
        Original dataframe which will be converted.
    to : string, default 'pandas'
        The data library to which format the dataframe will be converted to.
    return_library : bool, default True
        If set to True, the new dataframe library is also returned as an output.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.dataframe.DataFrame
        Converted dataframe, in the desired type.

    If return_library == True:

    new_pd : pandas or modin.pandas
        The dataframe library to which the input dataframe is converted to.
    '''
    lib = str(to).lower()
    if lib == 'pandas':
        import pandas as new_pd
    elif lib == 'modin':
        import modin.pandas as new_pd
    else:
        raise Exception(f'ERROR: Currently, convertion to a dataframe of type {to} is not supported. Availabale options are "pandas" and "modin".')
    converted_df = new_pd.DataFrame(data=df.to_numpy(), columns=df.columns)
    du.set_pandas_library(lib)
    if return_library is True:
        return converted_df, new_pd
    else:
        return converted_df


def set_bar_color(values, ids, seq_len, threshold=0,
                  neg_color='rgba(30,136,229,1)', pos_color='rgba(255,13,87,1)'):
    '''Determine each bar's color in a bar chart, according to the values being
    plotted and the predefined threshold.

    Parameters
    ----------
    values : numpy.Array
        Array containing the values to be plotted.
    ids : int or list of ints
        ID or list of ID's that select which time series / sequences to use in
        the color selection.
    seq_len : int or list of ints
        Single or multiple sequence lengths, which represent the true, unpadded
        size of the input sequences.
    threshold : int or float, default 0
        Value to use as a threshold in the plot's color selection. In other
        words, values that exceed this threshold will have one color while the
        remaining have a different one, as specified in the parameters.
    pos_color : string
        Color to use in the bars corresponding to threshold exceeding values.
    neg_color : string
        Color to use in the bars corresponding to values bellow the threshold.

    Returns
    -------
    colors : list of strings
        Resulting bar colors list.'''
    if type(ids) is list:
        # Create a list of lists, with the colors for each sequences' instances
        return [[pos_color if val > 0 else neg_color for val in values[id, :seq_len]]
                for id in ids]
    else:
        # Create a single list, with the colors for the sequence's instances
        return [pos_color if val > 0 else neg_color for val in values[ids, :seq_len]]
