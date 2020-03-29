from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
from tqdm.auto import tqdm                              # tqdm allows to track code execution progress
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
import itertools                                        # Flatten lists
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
        if is_num_nan(x):
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


def reverse(data):
    '''Reverse the order of a tensor or list.

    Parameters
    ----------
    data : torch.Tensor or list
        PyTorch tensor or list to revert.

    Returns
    -------
    data : torch.Tensor or list
        Reversed tensor or list.
    '''
    return data[::-1]


def replace_dict_strings(dct, str_to_replace='0', new_str='_', replace_keys=True,
                         replace_vals=True, inplace=False):
    '''Replace strings in a dictionary, in keys and/or values, with a new,
    desired string.

    Parameters
    ----------
    dct : dict
        Dictionary that will have its keys and/or values modified.
    str_to_replace : str, default '0'
        String to replace with a new one.
    new_str : str, default '_'
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
                # Find if there are any overlapping keys
                dict1_keys = set(dict1.keys())
                dict2_keys = set(dict2.keys())
                overlap_keys = dict1_keys.intersection(dict2_keys)
                for key in overlap_keys:
                    if ((isinstance(dict1[key], list) or isinstance(dict1[key], set))
                    or (isinstance(dict2[key], list) or isinstance(dict2[key], set))):
                        # Merge the lists
                        dict1[key] = set(dict1[key]) | set(dict2[key])
                        if isinstance(dict2[key], list):
                            dict1[key] = list(dict1[key])
                        dict2[key] = dict1[key]
                    else:
                        warnings.warn(f'Found an overlapping key {key} when merging two dictionaries which, as it doesn\'t point to a list or a set, can\'t be merged. As such, the value from the dictionary on the right will be kept.')
                # Merge the two input dictionaries
                return {**dict1, **dict2}
            else:
                raise Exception(f'ERROR: When `dict1` is specified as a single dictionary, the second argument `dict2` must also be a dictionary. Instead, received `dict2` of type {type(dict2)}.')
        else:
            raise Exception(f'ERROR: When `dict1` is specified as a single dictionary, the second argument `dict2` must also be set.')
    elif isinstance(dict1, list) and dict2 is None:
        # Initialize the new dictionary with the first one on the list
        new_dict = dict1[0]
        for i in range(len(dict1)):
            try:
                dict2 = dict1[i+1]
                # Find if there are any overlapping keys
                new_dict_keys = set(new_dict.keys())
                dict2_keys = set(dict2.keys())
                overlap_keys = new_dict_keys.intersection(dict2_keys)
                for key in overlap_keys:
                    if ((isinstance(new_dict[key], list) or isinstance(new_dict[key], set))
                    or (isinstance(dict2[key], list) or isinstance(dict2[key], set))):
                        # Merge the lists
                        new_dict[key] = set(new_dict[key]) | set(dict2[key])
                        if isinstance(dict2[key], list):
                            new_dict[key] = list(new_dict[key])
                        dict2[key] = new_dict[key]
                    else:
                        warnings.warn(f'Found an overlapping key {key} when merging two dictionaries which, as it doesn\'t point to a list or a set, can\'t be merged. As such, the value from the dictionary on the right will be kept.')
                # Try to merge with the next dictionary, if there is any
                new_dict = {**new_dict, **dict2}
            except:
                break
        return new_dict
    else:
        return Exception(f'ERROR: The first parameter `dict1` must be set as either a dictionary or a list of dictionaries. Instead, received `dict1` of type {type(dict1)}.')


def merge_lists(lists):
    '''Merge two or more lists into one.

    Parameters
    ----------
    lists : list of lists
        List containing all the lists that we want to merge.

    Returns
    -------
    lists : lists
        New list with all the input lists flatten in a single list.
    '''
    return list(itertools.chain.from_iterable(lists))


def remove_from_list(data, to_remove, update_idx=False):
    '''Remove values from a list, with the option to update the remaining values
    everytime one is removed.

    Parameters
    ----------
    data : list
        Data list to update by removing specified values.
    to_remove : list or int or float or str
        Values to remove from the list.
    update_idx : bool, default False

    Returns
    -------
    data : list
        Updated data list.
    '''
    if isinstance(to_remove, int) or isinstance(to_remove, float) or isinstance(to_remove, str):
        # Make sure that the values to remove are in a list format, even if it's just one
        to_remove = [to_remove]
    # Check if we need to update the values, in case they'll be used as indeces
    update_idx = all([isinstance(val, int) for val in to_remove]) and update_idx is True
    for val in to_remove:
        data.remove(val)
        if update_idx is True:
            for i in range(len(data)):
                if data[i] > val:
                    # Update value (which could be an index) to decrease its value
                    data[i] -= 1
    return data


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


def is_num_nan(x):
    '''Indicates if a number corresponds to a missing value.

    Parameters
    ----------
    x : int or float or string
        A numeric value that will be compared with possible missing value
        representations.

    Returns
    -------
    boolean
        Returns a boolean, being it True if the number corresponds to a missing
        value representation or False if it doesn't.
    '''
    str_val = str(x).lower()
    if str_val == 'nan' or str_val == '<na>':
        return True
    else:
        return False


def is_integer(x):
    '''Indicates if a number is an integer.

    Parameters
    ----------
    x : int or float or string
        A numeric value that will be checked if it's an integer.

    Returns
    -------
    boolean
        Returns a boolean, being it True if the number corresponds to an integer
        or False if it doesn't.
    '''
    try:
        float(x)
    except ValueError:
        return False
    return float(x).is_integer()


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


def iterations_loop(x, see_progress=True, desc=None, leave=True):
    '''Determine if a progress bar is shown or not.'''
    if see_progress is True:
        # Use a progress bar
        return tqdm(x, desc=desc, leave=leave)
    else:
        # Don't show any progress bar if see_progress is False
        return x


def convert_dtypes(df, dtypes=None, inplace=False):
    '''Converts a dataframe's data types to the desired ones.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.DataFrame
        Original dataframe which will be converted.
    dtypes : dict, default None
        Dictionary that indicates the desired dtype for each column.
        e.g. {'Var1': 'float64', 'Var2': 'UInt8', 'Var3': str}

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.dataframe.DataFrame
        Converted dataframe, in the desired data type.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframes
        data_df = df
    # Only use the dictionary keys that correspond to column names in the current dataframe
    dtype_dict = dict()
    df_columns = list(data_df.columns)
    for key, val in dtypes.items():
        if key in df_columns:
            dtype_dict[key] = dtypes[key]
        elif key.lower() in df_columns:
            dtype_dict[key.lower()] = dtypes[key]
    try:
        # Set the desired dtypes
        data_df = data_df.astype(dtype_dict)
    except:
        # Replace the '<NA>' objects with NumPy's NaN
        data_df = data_df.applymap(lambda x: x if str(x) != '<NA>' else np.nan)
        # Set the desired dtypes
        data_df = data_df.astype(dtype_dict)
    return data_df


def convert_dataframe(df, to='pandas', return_library=True, dtypes=None):
    '''Converts a dataframe to the desired dataframe library format.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.DataFrame
        Original dataframe which will be converted.
    to : string, default 'pandas'
        The data library to which format the dataframe will be converted to.
    return_library : bool, default True
        If set to True, the new dataframe library is also returned as an output.
    dtypes : dict, default None
        Dictionary that indicates the desired dtype for each column.
        e.g. {'Var1': 'float64', 'Var2': 'UInt8', 'Var3': str}

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
    if dtypes is None:
        # Infer adequate dtypes for the dataframe's columns
        converted_df = converted_df.infer_objects()
    else:
        # Set the desired dtypes
        converted_df = convert_dtypes(converted_df, dtypes=dtypes, inplace=True)
    if return_library is True:
        return converted_df, new_pd
    else:
        return converted_df


def convert_pyarrow_dtypes(df, inplace=False):
    '''Converts a dataframe's data types to a pyarrow supported version.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.DataFrame
        Original dataframe which will have its data types converted.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame or modin.pandas.dataframe.DataFrame
        Converted dataframe, in pyarrow compatible data types.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframes
        data_df = df
    # Create a columns data type dictionary
    dtype_dict = dict(data_df.dtypes)
    # Replace the pyarrow incompatible data types with similar, compatible ones
    for key, val in dtype_dict.items():
        val = str(val)
        if (val == 'UInt8' or val == 'UInt16' or val == 'UInt32'
        or val == 'Int8' or val == 'Int16' or val == 'Int32'
        or val == 'boolean'):
            dtype_dict[key] = 'float32'
        elif val == 'UInt64' or val == 'Int64':
            dtype_dict[key] = 'float64'
        elif val == 'string':
            dtype_dict[key] = str
    # Apply the new data types
    data_df = data_df.astype(dtype_dict)
    return data_df
