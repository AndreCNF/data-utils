from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
from . import utils                                     # Generic and useful methods
from . import embedding                                 # Embeddings and other categorical features handling methods

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def get_sequence_length_dict(df, id_column='subject_id', ts_column='ts'):
    '''Converts a Pandas dataframe into a padded NumPy array or PyTorch Tensor.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Data in a Pandas dataframe format which will be padded and converted
        to the requested data type.
    id_column : string, default 'subject_id'
        Name of the column which corresponds to the subject identifier in the
        dataframe.
    ts_column : string, default 'ts'
        Name of the column which corresponds to the timestamp in the
        dataframe.

    Returns
    -------
    seq_len_dict : dictionary, default None
        Dictionary containing the original sequence lengths of the dataframe.
        The keys should be the sequence identifiers (the numbers obtained from
        the id_column) and the values should be the length of each sequence.
    '''
    # Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
    seq_len_df = df.groupby(id_column)[ts_column].count().to_frame().sort_values(by=ts_column, ascending=False)
    seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])
    return seq_len_dict


def dataframe_to_padded_tensor(df, seq_len_dict=None, id_column='subject_id', 
                               ts_column='ts', data_type='PyTorch', 
                               padding_value=999999, cat_feat=None, 
                               inplace=False):
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
    data_type : string, default 'PyTorch'
        Indication of what kind of output data type is desired. In case it's
        set as 'NumPy', the function outputs a NumPy array. If it's 'PyTorch',
        the function outputs a PyTorch tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    cat_feat : string or list of strings, default None
        Name(s) of the categorical encoded feature(s) which will have their
        semicolon separators converted into its binary ASCII code. If not
        specified, the method will look through all columns, processing
        the ones that might have semicolons.
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
    # Make sure that all possible categorical encoded columns are in numeric format
    data_df = embedding.string_encod_to_numeric(df, cat_feat=cat_feat, inplace=inplace)
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
    # Making a padded numpy array version of the dataframe (all index has the same sequence length as the one with the max)
    arr = np.ones((n_ids, max_seq_len, n_inputs)) * padding_value
    # Iterator that outputs each unique identifier (e.g. each patient in the dataset)
    id_iter = iter(data_df[id_column].unique())
    # Count the iterations of ids
    count = 0
    # Assign each value from the dataframe to the numpy array
    for idt in id_iter:
        arr[count, :seq_len_dict[idt], :] = data_df[data_df[id_column] == idt].to_numpy()
        arr[count, seq_len_dict[idt]:, :] = padding_value
        count += 1
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
