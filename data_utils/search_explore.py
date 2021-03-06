import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
from . import utils                                     # Generic and useful methods
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def dataframe_missing_values(df, column=None):
    '''Returns a dataframe with the percentages of missing values of every column
    of the original dataframe.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original dataframe which the user wants to analyze for missing values.
    column : string, default None
        Optional argument which, if provided, makes the function only return
        the percentage of missing values in the specified column.

    Returns
    -------
    missing_value_df : pandas.DataFrame or dask.DataFrame
        DataFrame containing the percentages of missing values for each column.
    col_percent_missing : float
        If the "column" argument is provided, the function only returns a float
        corresponfing to the percentage of missing values in the specified column.
    '''
    if column is None:
        columns = df.columns
        percent_missing = df.isnull().sum() * 100 / len(df)
        if isinstance(df, dd.DataFrame):
            # Make sure that the values are computed, in case we're using Dask
            percent_missing = percent_missing.compute()
        missing_value_df = pd.DataFrame({'column_name': columns,
                                         'percent_missing': percent_missing})
        missing_value_df.sort_values('percent_missing', inplace=True)
        return missing_value_df
    else:
        col_percent_missing = df[column].isnull().sum() * 100 / len(df)
        return col_percent_missing


def is_boolean_column(df, column, n_unique_values=None):
    '''Checks if a given column is one hot encoded.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used, which contains the specified column.
    column : string
        Name of the column that will be checked for boolean format.
    n_unique_values : int, default None
        Number of the column's unique values. If not specified, it will
        be automatically calculated.

    Returns
    -------
    bool
        Returns true if the column is in boolean format.
        Otherwise, returns false.
    '''
    if str(df[column].dtype) == 'boolean':
        return True
    if n_unique_values is None:
        # Calculate the number of unique values
        n_unique_values = df[column].nunique()
        if isinstance(df, dd.DataFrame):
            # Make sure that the number of unique values are computed, in case we're using Dask
            n_unique_values = n_unique_values.compute()
    # Check if it only has, at most, 2 possible values
    if n_unique_values <= 2:
        unique_values = df[column].unique()
        if isinstance(df, dd.DataFrame):
            # Make sure that the unique values are computed, in case we're using Dask
            unique_values = unique_values.compute()
        # Remove NaNs from the list of unique values
        unique_values = [val for val in unique_values if not utils.is_num_nan(val)]
        # Check if the possible values are all numeric
        if (all([isinstance(x, numbers.Number) for x in unique_values])
        or all([isinstance(x, bool) or isinstance(x, np.bool_) for x in unique_values])):
            # Check if the only possible values are 0 and 1 (and ignore NaN's)
            unique_values = list(set(np.nan_to_num(unique_values)))
            unique_values.sort()
            unique_values = [val for val in unique_values if str(val).lower() != 'nan' ]
            if ((n_unique_values == 2 and unique_values == [0, 1])
            or (n_unique_values == 1 and unique_values == [0])
            or (n_unique_values == 1 and unique_values == [1])):
                return True
    return False


def list_boolean_columns(df, search_by_dtypes=False):
    '''Lists the columns in a dataframe which are in a boolean format.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used checked for one hot encoded columns.
    search_by_dtypes : bool, default False
        If set to True, the method will only look for boolean columns based on
        their data type. This is only reliable if all the columns' data types
        have been properly set.

    Returns
    -------
    list of strings
        Returns a list of the column names which correspond to one hot encoded columns.
    '''
    if search_by_dtypes is True:
        return [col for col in df.columns if str(df[col].dtype) == 'boolean' or df[col].dtype == 'UInt8']
    else:
        # Calculate the columns' number of unique values
        n_unique_values = df.nunique()
        if isinstance(df, dd.DataFrame):
            # Make sure that the number of unique values are computed, in case we're using Dask
            n_unique_values = n_unique_values.compute()
        if n_unique_values.min() > 2:
            # If there are no columns with just 2 unique values, then there are no binary columns
            return []
        else:
            return [col for col in df.columns if is_boolean_column(df, col, n_unique_values[col])]


def find_col_idx(df, feature):
    '''Find the index that corresponds to a given feature's column number on
    a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe on which to search for the feature idx
    feature : string
        Name of the feature whose index we want to find.

    Returns
    -------
    idx : int
        Index where the specified feature appears in the dataframe.'''
    return df.columns.get_loc(feature)


def find_val_idx(data, value, column=None):
    '''Find the index that corresponds to a given unique value in a data tensor.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the data on which the desired value will
        be searched for.
    value : numeric
        Unique value whose index on the data tensor one wants to find out.
    column : int, default None
        The number of the column in the data tensor that will be searched.

    Returns
    -------
    idx : int
        Index where the specified value appears in the data tensor.'''
    if len(data.shape) == 1:
        val = (data == value).nonzero().squeeze()
    elif column is not None:
        if len(data.shape) == 2:
            val = (data[:, column] == value).nonzero().squeeze()
        elif len(data.shape) == 3:
            val = (data[:, :, column] == value).nonzero().squeeze()
        else:
            raise Exception(f'ERROR: Currently this method only supports up to tree-dimensional data. User submitted data with {len(data.shape)} dimensions.')
    else:
        raise Exception('ERROR: If multidimensional data is being used, the column to search for must be specified in the `column` parameter.')
    if len(val.shape) > 1 or len(val) > 1:
        # Convert to a NumPy array
        return val.numpy()
    else:
        if val.shape[0] == 0:
            # No results found
            return None
        else:
            # Convert to a Python scalar
            return val.item()


def find_subject_idx(data, subject_id, subject_id_col=0):
    '''Find the index that corresponds to a given subject in a data tensor.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the data on which the subject's index will be
        searched for.
    subject_id : int or string
        Unique identifier of the subject whose index on the data tensor one
        wants to find out.
    subject_id_col : int, default 0
        The number of the column in the data tensor that stores the subject
        identifiers.

    Returns
    -------
    idx : int
        Index where the specified subject appears in the data tensor.'''
    return (data[:, 0, subject_id_col] == subject_id).nonzero().item()


def find_seq_len(labels, padding_value=999999):
    '''Find the lengths of the sequences based on the padding values present in
    a labels tensor.

    Parameters
    ----------
    labels : torch.Tensor
        PyTorch tensor containing the data on which the subject's index will be
        searched for.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    seq_lengths : torch.Tensor
        List of sequence lengths, relative to the input data.'''
    # Find when in each sequence the padding starts
    padding_start = ((labels == padding_value).cumsum(dim=1) == 1)
    # Detect the sequence lengths
    seq_lengths = padding_start.nonzero()[:, 1]
    if len(seq_lengths) < labels.shape[0]:
        # Assign the maximum sequence length to the sequences that aren't padded
        all_zero_seq_len = (padding_start.sum(dim=1) == 0) * labels.shape[1]
        all_zero_seq_len[all_zero_seq_len == 0] = seq_lengths
        seq_lengths = all_zero_seq_len
    return seq_lengths


def find_row_contains_word(df, feature, words):
    '''Find if each row in a specified dataframe string feature contains some
    word from a list.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe containing the feature on which to run the words search.
    feature : string
        Name of the feature through which the method will search if strings
        contain any of the specified words.
    words : list of strings or string
        List of the words to search for in the feature's rows.

    Returns
    -------
    row_contains_word : pandas.Series or dask.Series
        Boolean series indicating for each row of the dataframe if its specified
        feature contains any of the words that the user is looking for.'''
    row_contains_word = None
    if not df[feature].dtype == 'object':
        raise Exception(f'ERROR: The specified feature should have type "object", not type {df[feature].dtype}.')
    if isinstance(words, str):
        # Make sure that the words are in a list format, even if it's just one word
        words = [words]
    if any([not isinstance(word, str) for word in words]):
        raise Exception('ERROR: All words in the specified words list should be strings.')
    if isinstance(df, dd.DataFrame):
        row_contains_word = df[feature].apply(lambda row: any([word.lower() in row.lower() for word in words]),
                                              meta=('row', bool))
    elif isinstance(df, pd.DataFrame):
        row_contains_word = df[feature].apply(lambda row: any([word.lower() in row.lower() for word in words]))
    else:
        raise Exception(f'ERROR: `df` should either be a Pandas or Dask dataframe, not {type(df)}.')
    return row_contains_word


def get_element(x, n, till_the_end=False):
    '''Try to get an element from a list. Useful for nagging apply and map
    dataframe operations.

    Parameters
    ----------
    x : list or numpy.ndarray
        List from which to get an element.
    n : int
        Index of the element from the list that we want to retrieve.
    till_the_end : bool, default False
        If set to true, all elements from index n until the end of the list will
        be fetched. Otherwise, the method only returns the n'th element.

    Returns
    -------
    y : anything
        Returns the n'th element of the list or NaN if it's not found.
    '''
    try:
        if till_the_end is True:
            return x[n:]
        else:
            return x[n]
    except Exception:
        return np.nan


def get_element_from_split(orig_string, n, separator='|', till_the_end=False):
    '''Split a string by a specified separator and return the n'th element of
    the obtained list of words.

    Parameters
    ----------
    orig_string : string
        Original string on which to apply the splitting and element retrieval.
    n : int
        The index of the element to return from the post-split list of words.
    separator : string, default '|'
        Symbol that concatenates each string's words, which will be used in the
        splitting.
    till_the_end : bool, default False
        If set to true, all elements from index n until the end of the list will
        be fetched. Otherwise, the method only returns the n'th element.

    Returns
    -------
    n_element : string
        The n'th element from the split string.
    '''
    # Split the string, by the specified separator, to get the list of all words
    split_list = orig_string.split(separator)
    # Get the n'th element of the list
    n_element = get_element(split_list, n, till_the_end)
    if till_the_end is True:
        # Rejoin the elements of the list by their separator
        n_element = separator.join(n_element)
    return n_element
