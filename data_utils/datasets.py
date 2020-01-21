from torch.utils.data import Dataset                    # Pytorch base dataset class
from . import padding                                   # Padding and variable sequence length related methods

class Tabular_Dataset(Dataset):
    def __init__(self, arr, df, label_name=None):
        # Counter that indicates in which column we're in when searching for the label column
        col_num = 0
        for col in df.columns:
            if 'label' in col or col == label_name:
                # Column number corresponding to the label
                self.label_column = col_num
                break
            col_num += 1
        # Column numbers corresponding to the features
        self.features_columns = (list(range(self.label_column)) 
                                 + list(range(self.label_column + 1, arr.shape[1])))
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

        from torch.utils.data import Dataset


class Time_Series_Dataset(Dataset):
    def __init__(self, arr, df, label_name=None, id_column='subject_id',
                 ts_column='ts', seq_len_dict=None):
        # Counter that indicates in which column we're in when searching for the label column
        col_num = 0
        for col in df.columns:
            if 'label' in col or col == label_name:
                # Column number corresponding to the label
                self.label_column = col_num
                break
            col_num += 1
        # Column numbers corresponding to the features
        self.features_columns = (list(range(self.label_column))
                                 + list(range(self.label_column + 1, arr.shape[2])))
        # Features
        self.X = arr[:, :, self.features_columns]
        # Labels
        self.y = arr[:, :, self.label_column]
        # Sequence length dictionary
        if seq_len_dict is None:
            seq_len_dict = padding.get_sequence_length_dict(df, id_column=id_column, 
                                                            ts_column=ts_column)
        self.seq_len_dict = seq_len_dict

    def __getitem__(self, item):
        x_t = self.X[item]
        y_t = self.y[item]
        return x_t, y_t

    def __len__(self):
        return len(self.X)
