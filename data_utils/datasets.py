from torch.utils.data import Dataset                    # Pytorch base dataset class
import inspect                                          # Inspect methods and their arguments
from glob import glob                                   # List files that follow certain path and name rules
from . import padding                                   # Padding and variable sequence length related methods
from .embedding import embedding_bag_pipeline           # Categorical embedding method
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

class Tabular_Dataset(Dataset):
    '''A dataset object for tabular data, which separates features from
    labels, and allows to iteratively load data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe used to search for the label column.
    arr : torch.Tensor or numpy.ndarray
        The data itself which will be separated into features and labels,
        to the be loaded in this dataset or following dataloader objects.
    label_name : string, default None
        Name of the label column.
    '''
    def __init__(self, df, arr, label_name=None):
        # Counter that indicates in which column we're in when searching for the label column
        col_num = 0
        for col in df.columns:
            if 'label' in col or col == label_name:
                # Column name corresponding to the label
                self.label_column_name = col
                # Column number corresponding to the label
                self.label_column_num = col_num
                break
            col_num += 1
        # Column numbers corresponding to the features
        self.features_columns = (list(range(self.label_column_num))
                                 + list(range(self.label_column_num + 1, arr.shape[1])))
        # Features
        self.X = arr[:, self.features_columns]
        # Labels
        self.y = arr[:, self.label_column]

    def __getitem__(self, item):
        x_i = self.X[item]
        y_i = self.y[item]
        return x_i, y_i

    def __len__(self):
        return len(self.X)


class Time_Series_Dataset(Dataset):
    '''A dataset object for time series data, which separates features from
    labels, and allows to iteratively load data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe used to search for the label column.
    arr : torch.Tensor or numpy.ndarray, default None
        The data itself which will be separated into features and labels,
        to the be loaded in this dataset or following dataloader objects.
        If it is not specified, the data will be used directly from the
        dataframe `df`.
    label_name : string, default None
        Name of the label column.
    id_column : string, default None
        Name of the sequence ID column.
    ts_column : string, default None
        Name of the timestamp column.

    If arr is None:

    seq_len_dict : dictionary, default None
        Dictionary containing the original sequence lengths of the dataframe.
        The keys should be the sequence identifiers (the numbers obtained from
        the id_column) and the values should be the length of each sequence.
    embedding_layer : torch.nn.EmbeddingBag or torch.nn.ModuleList or torch.nn.ModuleDict
        PyTorch layer(s) that applies the embedding bag, i.e. calculates the
        average embedding based on multiple encoded values.
    embed_features : int or list of int or str or list of str or list of list of int
        Index (or indices) or name(s) of the categorical column(s) that will be
        ran through its (or their) respective embedding layer(s). This feature(s)
        is (are) removed from the data tensor after the embedding columns are
        added. In case the input data is in a dataframe format, the feature(s)
        can be specified by name.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    '''
    def __init__(self, df, arr=None, label_name=None, id_column='subject_id',
                 ts_column='ts', seq_len_dict=None, embedding_layer=None,
                 embed_features=None, padding_value=999999):
        if arr is None:
            self.data_type = 'dataframe'
            self.id_column_name = id_column
            self.ts_column_name = ts_column
            self.padding_value = padding_value
            if embedding_layer is not None and embed_features is not None:
                self.embedding_layer = embedding_layer
                self.embed_features = embed_features
        else:
            self.data_type = 'tensor'
        # Counter that indicates in which column we're in when searching for the label column
        col_num = 0
        for col in df.columns:
            if 'label' in col or col == label_name:
                # Column name corresponding to the label
                self.label_column_name = col
                # Column number corresponding to the label
                self.label_column_num = col_num
                break
            col_num += 1
        if self.data_type == 'tensor':
            # Column numbers corresponding to the features
            self.features_columns_num = (list(range(self.label_column_num))
                                         + list(range(self.label_column_num + 1, arr.shape[2])))
            # Features
            self.X = arr[:, :, self.features_columns_num]
            # Labels
            self.y = arr[:, :, self.label_column_num]
        elif self.data_type == 'dataframe':
            # Column names corresponding to the features
            self.features_columns = list(df.columns)
            self.features_columns.remove(self.label_column_name)
            # Column numbers corresponding to the features
            self.features_columns_num = (list(range(self.label_column_num))
                                         + list(range(self.label_column_num + 1, len(df.columns))))
            # Features
            self.X = df[self.features_columns]
            # Labels
            self.y = df[self.label_column_name]
            # List of items (sequences)
            self.seq_items = df[id_column].unique()
            # Sequence length dictionary
            if seq_len_dict is None:
                self.seq_len_dict = padding.get_sequence_length_dict(df, id_column=id_column,
                                                                     ts_column=ts_column)
            else:
                self.seq_len_dict = seq_len_dict

    def __getitem__(self, item):
        if self.data_type == 'tensor':
            # Get the data
            x_t = self.X[item]
            y_t = self.y[item]
        elif self.data_type == 'dataframe':
            # Get the sequence ID
            seq = self.seq_items[item]
            # Find the indices of the dataframes
            idx = self.X.index[self.X[self.id_column_num] == seq]
            # Get the data
            x_t = self.X.iloc[idx]
            y_t = self.y.iloc[idx]
            if self.embed_features is not None:
                # Run each embedding layer on each respective feature, adding the
                # resulting embedding values to the tensor and removing the original,
                # categorical encoded columns
                x_t = embedding_bag_pipeline(x_t, self.embedding_layer, self.embed_features,
                                             inplace=True)
            # Pad the data (both X and y)
            df = pd.concat([x_t, y_t], axis=1)
            df = padding.dataframe_to_padded_tensor(df, seq_len_dict=self.seq_len_dict,
                                                    id_column=self.id_column_name,
                                                    ts_column=self.ts_column_name,
                                                    padding_value=self.padding_value,
                                                    inplace=True)
            # Features
            x_t = df[:, :-1]
            # Labels
            y_t = df[:, -1]
        return x_t, y_t

    def __len__(self):
        if self.data_type == 'tensor':
            return len(self.X)
        elif self.data_type == 'dataframe':
            return len(self.seq_items)


class Large_Dataset(Dataset):
    '''A generic dataset object for any data type, including large datasets,
    which allows to iteratively load data from individual data files, in a
    lazy loading way.

    Parameters
    ----------
    files_name : string, default None
        Core name that is shared by the data files.
    process_pipeline : function
        Python function that preprocesses data in each data loading
        iteration.
    id_column : string, default None
        Name of the sequence or sample ID column.
    initial_analysis : function
        Python function that runs in the dataset initialization, which can
        be used to retrieve additional parameters.
    files_path : string
        Name of the directory where the data files are stored.
    '''
    def __init__(self, files_name, process_pipeline, id_column,
                 initial_analysis=None, files_path='', **kwargs):
        # Load the file names
        self.files = glob(f'{files_path}{files_name}_*.ftr')
        # Data preprocessing pipeline function
        self.process_pipeline = process_pipeline
        # Other basic data information
        self.id_column_name = id_column
        # Add aditional data that the user might have specified
        self.__dict__.update(kwargs)
        # Initial analysis pipeline to fetch important, context specific information
        self.initial_analysis = initial_analysis
        if self.initial_analysis is not None:
            # Run the initial analysis
            self.initial_analysis(self)

    def __getitem__(self, item):
        # Load a data file
        df = pd.read_feather(self.files[item])
        # Run the data preprocessing pipeline, which should return the features
        # and label tensors
        features, labels = self.process_pipeline(self, df)
        return features, labels

    def __len__(self):
        return len(self.files)
