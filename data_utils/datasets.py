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
    def __init__(self, df, arr, label_name=None):
        # [TODO] Add documentation
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
    def __init__(self, df, arr=None, label_name=None, id_column='subject_id',
                 ts_column='ts', seq_len_dict=None, embed_layers=None,
                 embed_features=None, padding_value=999999):
        # [TODO] Add documentation
        if arr is None:
            self.data_type = 'dataframe'
            self.id_column_name = id_column
            self.ts_column_name = ts_column
            self.padding_value = padding_value
            if embed_layers is not None and embed_features is not None:
                self.embed_layers = embed_layers
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
                x_t = embedding_bag_pipeline(x_t, self.embed_layers, self.embed_features,
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
    def __init__(self, files_name, process_pipeline, id_column,
                 initial_analysis=None, files_path='', **kwargs):
        # [TODO] Add documentation
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
