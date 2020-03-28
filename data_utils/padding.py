from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
from . import utils                                     # Generic and useful methods
from . import search_explore                            # Methods to search and explore data
from . import embedding                                 # Embeddings and other categorical features handling methods

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def get_sequence_length_dict(df, id_column='subject_id', ts_column='ts'):
    '''Creates a dictionary with the original sequence lengths of a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    id_column : string or int, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    ts_column : string or int, default 'ts'
        Name of the column which corresponds to the timestamp in the
        dataframe.

    Returns
    -------
    seq_len_dict : dictionary, default None
        Dictionary containing the original sequence lengths of the dataframe.
        The keys should be the sequence identifiers (the numbers obtained from
        the id_column) and the values should be the length of each sequence.
    '''
    if isinstance(id_column, int) and isinstance(ts_column, int):
        # Convert the column indeces to the column names
        column_names = list(df.columns)
        id_column = column_names[id_column]
        ts_column = column_names[ts_column]
    # Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
    seq_len_df = df.groupby(id_column)[ts_column].count().to_frame().sort_values(by=ts_column, ascending=False)
    seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.to_numpy()))])
    return seq_len_dict


def dataframe_to_padded_tensor(df, seq_len_dict=None, id_column='subject_id',
                               ts_column='ts', bool_feat=None, data_type='PyTorch',
                               padding_value=999999, inplace=False):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    seq_len_dict : dictionary, default None
        Dictionary containing the original sequence lengths of the dataframe.
        The keys should be the sequence identifiers (the numbers obtained from
        the id_column) and the values should be the length of each sequence.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    ts_column : string, default 'ts'
        Name of the column which corresponds to the timestamp in the
        dataframe.
    bool_feat : string or list of strings, default None
        Name(s) of the boolean feature(s) of the dataframe. In order to prevent
        confounding padding values with encodings, these features must have
        their padding values replaced with 0. If not specified, the method
        will automatically look for boolean columns in the dataframe. If you
        don't want any feature to be treated as a boolean dtype, set `bool_feat=[]`
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    arr : torch.Tensor or numpy.ndarray
        PyTorch tensor or NumPy array version of the dataframe, after being
        padded with the specified padding value to have a fixed sequence
        length.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    if seq_len_dict is None:
        # Find the sequence lengths and store them in a dictionary
        seq_len_dict = get_sequence_length_dict(data_df, id_column, ts_column)
    # Fetch the number of unique sequence IDs
    n_ids = data_df[id_column].nunique()
    if isinstance(df, dd.DataFrame):
        # Make sure that the number of unique values are computed, in case we're using Dask
        n_ids = n_ids.compute()
    # Get the number of columns in the dataframe
    n_inputs = len(data_df.columns)
    # Max sequence length (e.g. patient with the most temporal events)
    max_seq_len = seq_len_dict[max(seq_len_dict, key=seq_len_dict.get)]
    if n_ids > 1:
        # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
        arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value
        # Fetch a list with all the unique identifiers (e.g. each patient in the dataset)
        unique_ids = data_df[id_column].unique()
        # Iterator that outputs each unique identifier
        id_iter = iter(unique_ids)
        # Count the iterations of ids
        count = 0
        # Assign each value from the dataframe to the numpy array
        for idt in id_iter:
            arr[count, :seq_len_dict[idt], :] = data_df[data_df[id_column] == idt].to_numpy()
            arr[count, seq_len_dict[idt]:, :] = padding_value
            count += 1
    else:
        # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
        arr = np.ones((max_seq_len, n_inputs)) * padding_value
        # Assign each value from the dataframe to the numpy array
        idt = data_df[id_column].iloc[0]
        arr[:seq_len_dict[idt], :] = data_df.to_numpy()
        arr[seq_len_dict[idt]:, :] = padding_value
    if bool_feat is None:
        # Find the boolean columns in the dataframe
        bool_feat = search_explore.list_one_hot_encoded_columns(data_df)
        # Make sure that none of the ID columns are considered boolean
        bool_feat = list(set(bool_feat) - set([id_column, ts_column]))
        # Get the indeces of the boolean features
        bool_feat = [search_explore.find_col_idx(data_df, feature) for feature in bool_feat]
    if isinstance(bool_feat, str):
        # Get the index of the boolean feature
        bool_feat = search_explore.find_col_idx(data_df, bool_feat)
        # Make sure that the boolean feature names are in a list format
        bool_feat = [bool_feat]
    if not isinstance(bool_feat, list):
        raise Exception(f'ERROR: The `bool_feat` argument must be specified as either a single string or a list of strings. Received input with type {type(bool_feat)}.')
    if len(bool_feat) > 0:
        if n_ids > 1:
            # Iterator that outputs each unique identifier
            id_iter = iter(unique_ids)
            # Count the iterations of ids
            count = 0
            # Replace each padding value in the boolean features with zero
            for idt in id_iter:
                arr[count, seq_len_dict[idt]:, bool_feat] = 0
                count += 1
        else:
            # Replace each padding value in the boolean features with zero
            idt = data_df[id_column].iloc[0]
            arr[seq_len_dict[idt]:, bool_feat] = 0
    # Make sure that the data type asked for is a string
    if not isinstance(data_type, str):
        raise Exception('ERROR: Please provide the desirable data type in a string format.')
    if data_type.lower() == 'numpy':
        return arr
    elif data_type.lower() == 'pytorch':
        return torch.from_numpy(arr)
    else:
        raise Exception('ERROR: Unavailable data type. Please choose either NumPy or PyTorch.')


def sort_by_seq_len(data, seq_len_dict, labels=None, id_column=0):
    '''Sort the data by sequence length in order to correctly apply it to a
    PyTorch neural network.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor on which sorting by sequence length will be applied.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    labels : torch.Tensor, default None
        Labels corresponding to the data used, either specified in the input
        or all the data that the interpreter has.
    id_column : int, default 0
        Number of the column which corresponds to the subject identifier in
        the data tensor.

    Returns
    -------
    sorted_data : torch.Tensor, default None
        Data tensor already sorted by sequence length.
    sorted_labels : torch.Tensor, default None
        Labels tensor already sorted by sequence length. Only outputed if the
        labels data is specified in the input.
    x_lengths : list of int
        Sorted list of sequence lengths, relative to the input data.
    '''
    # Get the original lengths of the sequences, for the input data
    x_lengths = [seq_len_dict[id] for id in list(data[:, 0, id_column].numpy())]
    is_sorted = all(x_lengths[i] >= x_lengths[i+1] for i in range(len(x_lengths)-1))
    if is_sorted is True:
        # Do nothing if it's already sorted
        sorted_data = data
        sorted_labels = labels
    else:
        # Sorted indeces to get the data sorted by sequence length
        data_sorted_idx = list(np.argsort(x_lengths)[::-1])
        # Sort the x_lengths array by descending sequence length
        x_lengths = [x_lengths[idx] for idx in data_sorted_idx]
        # Sort the data by descending sequence length
        sorted_data = data[data_sorted_idx, :, :]
        if labels is not None:
            # Sort the labels by descending sequence length
            sorted_labels = labels[data_sorted_idx, :]
    if labels is None:
        return sorted_data, x_lengths
    else:
        return sorted_data, sorted_labels,  x_lengths


def pad_list(x_list, length, padding_value=999999):
    '''Pad a list with a specific padding value until the desired length is
    met.

    Parameters
    ----------
    x_list : list
        List which will be padded.
    length : int
        Desired length for the final padded list.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    x_list : list
        Resulting padded list'''
    return x_list + [padding_value] * (length - len(x_list))
