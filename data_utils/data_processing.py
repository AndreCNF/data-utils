from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
import warnings                                         # Print warnings for bad practices
from . import utils                                     # Generic and useful methods
from . import search_explore                            # Methods to search and explore data
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def get_clean_label(orig_label, clean_labels, column_name=None):
    '''Gets the clean version of a given label.

    Parameters
    ----------
    orig_label : string
        Original label name that needs to be converted to the new format.
    clean_labels : dict
        Dictionary that converts each original label into a new, cleaner designation.
    column_name : string, default None
        Optional parameter to indicate a column name, which is used to specify better the
        missing values.

    Returns
    -------
    key : string
        Returns the dictionary key from clean_labels that corresponds to the translation
        given to the input label orig_label.
    '''
    for key in clean_labels:
        if orig_label in clean_labels[key]:
            return key

    # Remaining labels (or lack of one) are considered as missing data
    if column_name is not None:
        return f'{column_name}_missing_value'
    else:
        return 'missing_value'


def rename_index(df, name):
    '''Renames the dataframe's index to a desired name. Specially important
    for dask dataframes, as they don't support any elegant, one-line method
    for this.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe whose index column will be renamed.
    name : string
        The new name for the index column.

    Returns
    -------
    df : dask.DataFrame
        Dataframe with a renamed index column.
    '''
    if isinstance(df, dd.DataFrame):
        feat_names = set(df.columns)
        df = df.reset_index()
        orig_idx_name = set(df.columns) - feat_names
        orig_idx_name = orig_idx_name.pop()
        df = df.rename(columns={orig_idx_name: name})
        df = df.set_index(name)
    elif isinstance(df, pd.DataFrame):
        df.index.names = [name]
    else:
        raise Exception(f'ERROR: Input "df" should either be a pandas dataframe or a dask dataframe, not type {type(df)}.')
    return df


def standardize_missing_values(x, specific_nan_strings=[]):
    '''Apply function to be used in replacing missing value representations with
    the standard NumPy NaN value.

    Parameters
    ----------
    x : str, int or float
        Value to be analyzed and replaced with NaN, if it has a missing value
        representation.
    specific_nan_strings : list of strings, default []
        Parameter where the user can specify additional strings that
        should correspond to missing values.

    Returns
    -------
    x : str, int or float
        Corrected value, with standardized missing value representation.
    '''
    if isinstance(x, str):
        if utils.is_string_nan(x, specific_nan_strings):
            return np.nan
        else:
            return x
    else:
        return x


def standardize_missing_values_df(df, see_progress=True, specific_nan_strings=[]):
    '''Replace all elements in a dataframe that have a missing value
    representation with the standard NumPy NaN value.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe to be analyzed and have its content replaced with NaN,
        wherever a missing value representation is found.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.
    specific_nan_strings : list of strings, default []
        Parameter where the user can specify additional strings that
        should correspond to missing values.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Corrected dataframe, with standardized missing value representation.
    '''
    for feature in utils.iterations_loop(df.columns, see_progress=see_progress):
        if isinstance(df, dd.DataFrame):
            df[feature] = df[feature].apply(lambda x: standardize_missing_values(x, specific_nan_strings), 
                                            meta=df[feature]._meta.dtypes)
        elif isinstance(df, pd.DataFrame):
            df[feature] = df[feature].apply(lambda x: standardize_missing_values(x, specific_nan_strings))
        else:
            raise Exception(f'ERROR: Input "df" should either be a pandas dataframe or a dask dataframe, not type {type(df)}.')
    return df


def remove_cols_with_many_nans(df, nan_percent_thrsh=40, inplace=False):
    '''Remove columns that have too many NaN's (missing values).

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be processed, to remove columns with high
        percentages of missing values.
    nan_percent_thrsh : int or float, default 40
        Threshold value above which it's considered a column with too
        many missing values. Measured in percentage of missing values,
        in 100% format.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Corrected dataframe, with columns removed that had too many 
        missing values.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    # Find each column's missing values percentage
    nan_percent_df = search_explore.dataframe_missing_values(data_df)
    # Remove columns that exceed the missing values percentage threshold
    many_nans_cols = list(nan_percent_df[nan_percent_df.percent_missing > nan_percent_thrsh].column_name)
    data_df = data_df.drop(many_nans_cols, axis = 1)
    return data_df


def clean_naming(x):
    '''Change strings to only have lower case letters and underscores.

    Parameters
    ----------
    x : string or list of strings
        String(s) on which to clean the naming, standardizing it.

    Returns
    -------
    x : string or list of strings
        Cleaned string(s).
    '''
    if isinstance(x, list):
        x = [string.lower().replace('  ', '')
                           .replace(' ', '_')
                           .replace(',', '_and') for string in x]
    elif (isinstance(x, pd.DataFrame)
    or isinstance(x, pd.Series)
    or isinstance(x, dd.DataFrame)
    or isinstance(x, dd.Series)):
        raise Exception('ERROR: Wrong method. When using dataframes or series, use clean_categories_naming() method instead.')
    else:
        x = (str(x).lower().replace('  ', '')
                           .replace(' ', '_')
                           .replace(',', '_and'))
    return x


def clean_categories_naming(df, column, clean_missing_values=True,
                            specific_nan_strings=[]):
    '''Change categorical values to only have lower case letters and underscores.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that contains the column to be cleaned.
    column : string
        Name of the dataframe's column which needs to have its string values
        standardized.
    clean_missing_values : bool, default True
        If set to True, the algorithm will search for missing value
        representations and replace them with the standard, NumPy NaN value.
    specific_nan_strings : list of strings, default []
        Parameter where the user can specify additional strings that
        should correspond to missing values.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe with its string column already cleaned.
    '''
    if isinstance(df, dd.DataFrame):
        df[column] = (df[column].map(clean_naming, meta=('x', str)))
        if clean_missing_values is True:
            df[column] = df[column].apply(lambda x: standardize_missing_values(x, specific_nan_strings),
                                          meta=df[column]._meta.dtypes)
    else:
        df[column] = (df[column].map(clean_naming))
        if clean_missing_values is True:
            df[column] = df[column].apply(lambda x: standardize_missing_values(x, specific_nan_strings))
    return df


def one_hot_encoding_dataframe(df, columns, clean_name=True, clean_missing_values=True,
                               specific_nan_strings=[], has_nan=False, join_rows=True,
                               join_by=['patientunitstayid', 'ts'],
                               get_new_column_names=False, inplace=False):
    '''Transforms specified column(s) from a dataframe into a one hot encoding
    representation.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe that will be used, which contains the specified column.
    columns : list of strings
        Name of the column(s) that will be conveted to one hot encoding. Even if
        it's just one column, please provide inside a list.
    clean_name : bool, default True
        If set to true, changes the name of the categorical values into lower
        case, with words separated by an underscore instead of space.
    clean_missing_values : bool, default True
        If set to True, the algorithm will search for missing value
        representations and replace them with the standard, NumPy NaN value.
    specific_nan_strings : list of strings, default []
        Parameter where the user can specify additional strings that
        should correspond to missing values.
    has_nan : bool, default False
        If set to true, will first fill the missing values (NaN) with the string
        f'{column}_missing_value'.
    join_rows : bool, default True
        If set to true, will group the rows created by the one hot encoding by
        summing the boolean values in the rows that have the same identifiers.
    join_by : string or list, default ['subject_id', 'ts'])
        Name of the column (or columns) which serves as a unique identifier of
        the dataframe's rows, which will be used in the groupby operation if the
        parameter join_rows is set to true. Can be a string (single column) or a
        list of strings (multiple columns).
    get_new_column_names : bool, default False
        If set to True, the names of the new columns will also be outputed.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Raises
    ------
    ColumnNotFoundError
        Column name not found in the dataframe.

    Returns
    -------
    ohe_df : pandas.DataFrame or dask.DataFrame
        Returns a new dataframe with the specified column in a one hot encoding
        representation.
    new_column_names : list of strings
        List of the new, one hot encoded columns' names.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    for col in columns:
        # Check if the column exists
        if col not in data_df.columns:
            raise Exception('ERROR: Column name not found in the dataframe.')
        if has_nan is True:
            # Fill NaN with "missing_value" name
            data_df[col] = data_df[col].fillna(value='missing_value')
        if clean_name is True:
            # Clean the column's string values to have the same, standard format
            data_df = clean_categories_naming(data_df, col, clean_missing_values, specific_nan_strings)
        # Cast the variable into the built in pandas Categorical data type
        if 'pandas' in str(type(data_df)):
            data_df[col] = pd.Categorical(data_df[col])
    if isinstance(data_df, dd.DataFrame):
        data_df = data_df.categorize(columns)
    if get_new_column_names is True:
        # Find the previously existing column names
        old_column_names = data_df.columns
    # Apply the one hot encoding to the specified columns
    if isinstance(data_df, dd.DataFrame):
        ohe_df = dd.get_dummies(data_df, columns=columns)
    else:
        ohe_df = pd.get_dummies(data_df, columns=columns)
    if join_rows is True:
        # Columns which are one hot encoded
        ohe_columns = search_explore.list_one_hot_encoded_columns(ohe_df)
        # Group the rows that have the same identifiers
        ohe_df = ohe_df.groupby(join_by).sum(min_count=1).reset_index()
        # Clip the one hot encoded columns to a maximum value of 1
        # (there might be duplicates which cause values bigger than 1)
        ohe_df.loc[:, ohe_columns] = ohe_df[ohe_columns].clip(upper=1)
    if get_new_column_names is True:
        # Find the new column names and output them
        new_column_names = list(set(ohe_df.columns) - set(old_column_names))
        return ohe_df, new_column_names
    else:
        return ohe_df


def category_to_feature(df, categories_feature, values_feature, min_len=None,
                        see_progress=True, inplace=False):
    '''Convert a categorical column and its corresponding values column into
    new features, one for each category.
    WARNING: Currently not working properly on a Dask dataframe. Apply .compute()
    to the dataframe to convert it to Pandas, before passing it to this method.
    If the data is too big to run on Pandas, use the category_to_feature_big_data
    method.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe on which to add the new features.
    categories_feature : string
        Name of the feature that contains the categories that will be converted
        to individual features.
    values_feature : string
        Name of the feature that has each category's corresponding value, which
        may or may not be a category on its own (e.g. it could be numeric values).
    min_len : int, default None
        If defined, only the categories that appear on at least `min_len` rows
        are converted to features.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data_df : pandas.DataFrame or dask.DataFrame
        Dataframe with the newly created features.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    # Find the unique categories
    categories = data_df[categories_feature].unique()
    if isinstance(df, dd.DataFrame):
        categories = categories.compute()
    # Create a feature for each category
    for category in utils.iterations_loop(categories, see_progress=see_progress):
        if min_len is not None:
            # Check if the current category has enough data to be worth it to convert to a feature
            if len(data_df[data_df[categories_feature] == category]) < min_len:
                # Ignore the current category
                continue
        # Convert category to feature
        data_df[category] = data_df.apply(lambda x: x[values_feature] if x[categories_feature] == category
                                                    else np.nan, axis=1)
    return data_df


def category_to_feature_big_data(df, categories_feature, values_feature,
                                 min_len=None, see_progress=True):
    '''Convert a categorical column and its corresponding values column into
    new features, one for each category. Optimized for very big Dask dataframes,
    which can't be processed as a whole Pandas dataframe.

    Parameters
    ----------
    df : dask.DataFrame
        Dataframe on which to add the new features.
    categories_feature : string
        Name of the feature that contains the categories that will be converted
        to individual features.
    values_feature : string
        Name of the feature that has each category's corresponding value, which
        may or may not be a category on its own (e.g. it could be numeric values).
    min_len : int, default None
        If defined, only the categories that appear on at least `min_len` rows
        are converted to features.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.

    Returns
    -------
    data_df : dask.DataFrame
        Dataframe with the newly created features.
    '''
    # Create a list with Pandas dataframe versions of each partition of the
    # original Dask dataframe
    df_list = []
    print('Converting categories to features in each partition...')
    for n in utils.iterations_loop(range(df.npartitions), see_progress=see_progress):
        # Process each partition separately in Pandas
        tmp_df = df.get_partition(n).compute()
        tmp_df = category_to_feature(tmp_df, categories_feature=categories_feature,
                                     values_feature=values_feature, min_len=min_len, 
                                     see_progress=see_progress)
        df_list.append(tmp_df)
    # Rejoin all the partitions into a Dask dataframe with the same number of
    # partitions it originally had
    print('Rejoining partitions into a Dask dataframe...')
    data_df = dd.from_pandas(pd.concat(df_list, sort=False), npartitions=df.npartitions)
    print('Done!')
    return data_df


def remove_rows_unmatched_key(df, key, columns):
    '''Remove rows corresponding to the keys that weren't in the dataframe merged at the right.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe resulting from a asof merge which will be searched for missing values.
    key : string
        Name of the column which was used as the "by" key in the asof merge. Typically
        represents a temporal feature from a time series, such as days or timestamps.
    columns : list of strings
        Name of the column(s), originating from the dataframe which was merged at the
        right, which should not have any missing values. If it has, it means that
        the corresponding key wasn't present in the original dataframe. Even if there's
        just one column to analyze, it should be received in list format.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Returns the input dataframe but without the rows which didn't have any values
        in the right dataframe's features.
    '''
    for k in utils.iterations_loop(df[key].unique()):
        # Variable that counts the number of columns which don't have any value
        # (i.e. all rows are missing values) for a given identifier 'k'
        num_empty_columns = 0
        for col in columns:
            if df[df[key] == k][col].isnull().sum() == len(df[df[key] == k]):
                # Found one more column which is full of missing values for identifier 'k'
                num_empty_columns += 1
        if num_empty_columns == len(columns):
            # Eliminate all rows corresponding to the analysed key if all the columns
            # are empty for the identifier 'k'
            df = df[~(df[key] == k)]
    return df


def apply_zscore_norm(value, df=None, mean=None, std=None, categories_means=None,
                      categories_stds=None, groupby_columns=None):
    '''Performs z-score normalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Original, unnormalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group normalization, i.e. when
        values are normalized according to their corresponding categories.
    mean : int or float, default None
        Average (mean) value to be used in the z-score normalization.
    std : int or float, default None
        Standard deviation value to be used in the z-score normalization.
    categories_means : dict, default None
        Dictionary containing the average values for each set of categories.
    categories_stds : dict, default None
        Dictionary containing the standard deviation values for each set of
        categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (mean and standard deviation) are retrieved.

    Returns
    -------
    value_norm : int or float
        Z-score normalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean is not None and std is not None:
        return (value - mean) / std
    elif (df is not None and categories_means is not None
          and categories_stds is not None and groupby_columns is not None):
        try:
            if isinstance(groupby_columns, list):
                return ((value - categories_means[tuple(df[groupby_columns])])
                        / categories_stds[tuple(df[groupby_columns])])
            else:
                return ((value - categories_means[df[groupby_columns]])
                        / categories_stds[df[groupby_columns]])
        except Exception:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `mean` and `std` or the `df`, `categories_means`, `categories_stds` and `groupby_columns` must be set.')


def apply_minmax_norm(value, df=None, min=None, max=None, categories_mins=None,
                      categories_maxs=None, groupby_columns=None):
    '''Performs minmax normalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Original, unnormalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group normalization, i.e. when
        values are normalized according to their corresponding categories.
    min : int or float, default None
        Minimum value to be used in the minmax normalization.
    max : int or float, default None
        Maximum value to be used in the minmax normalization.
    categories_mins : dict, default None
        Dictionary containing the minimum values for each set of categories.
    categories_maxs : dict, default None
        Dictionary containing the maximum values for each set of categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (minimum and maximum) are retrieved.

    Returns
    -------
    value_norm : int or float
        Minmax normalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if min and max:
        return (value - min) / (max - min)
    elif df and categories_mins and categories_maxs and groupby_columns:
        try:
            if isinstance(groupby_columns, list):
                return ((value - categories_mins[tuple(df[groupby_columns])])
                        / (categories_maxs[tuple(df[groupby_columns])] - categories_mins[tuple(df[groupby_columns])]))
            else:
                return ((value - categories_mins[df[groupby_columns]])
                        / (categories_maxs[df[groupby_columns]] - categories_mins[df[groupby_columns]]))
        except Exception:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `min` and `max` or the `df`, `categories_mins`, `categories_maxs` and `groupby_columns` must be set.')


def apply_zscore_denorm(value, df=None, mean=None, std=None, categories_means=None,
                      categories_stds=None, groupby_columns=None):
    '''Performs z-score denormalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Input normalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group denormalization, i.e. when
        values are denormalized according to their corresponding categories.
    mean : int or float, default None
        Average (mean) value to be used in the z-score denormalization.
    std : int or float, default None
        Standard deviation value to be used in the z-score denormalization.
    categories_means : dict, default None
        Dictionary containing the average values for each set of categories.
    categories_stds : dict, default None
        Dictionary containing the standard deviation values for each set of
        categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (mean and standard deviation) are retrieved.

    Returns
    -------
    value_denorm : int or float
        Z-score denormalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if mean is not None and std is not None:
        return value * std + mean
    elif (df is not None and categories_means is not None
          and categories_stds is not None and groupby_columns is not None):
        try:
            if isinstance(groupby_columns, list):
                return (value * categories_stds[tuple(df[groupby_columns])]
                        + categories_means[tuple(df[groupby_columns])])
            else:
                return (value * categories_stds[df[groupby_columns]]
                        + categories_means[df[groupby_columns]])
        except Exception:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `mean` and `std` or the `df`, `categories_means`, `categories_stds` and `groupby_columns` must be set.')


def apply_minmax_denorm(value, df=None, min=None, max=None, categories_mins=None,
                      categories_maxs=None, groupby_columns=None):
    '''Performs minmax denormalization when used inside a Pandas or Dask
    apply function.

    Parameters
    ----------
    value : int or float
        Input normalized value.
    df : pandas.DataFrame or dask.DataFrame, default None
        Original pandas dataframe which is used to retrieve the
        necessary statistical values used in group denormalization, i.e. when
        values are denormalized according to their corresponding categories.
    min : int or float, default None
        Minimum value to be used in the minmax denormalization.
    max : int or float, default None
        Maximum value to be used in the minmax denormalization.
    categories_mins : dict, default None
        Dictionary containing the minimum values for each set of categories.
    categories_maxs : dict, default None
        Dictionary containing the maximum values for each set of categories.
    groupby_columns : string or list of strings, default None
        Name(s) of the column(s) that contains the categories from which
        statistical values (minimum and maximum) are retrieved.

    Returns
    -------
    value_denorm : int or float
        Minmax denormalized value.
    '''
    if not isinstance(value, numbers.Number):
        raise Exception(f'ERROR: Input value should be a number, not an object of type {type(value)}.')
    if min is not None and max is not None:
        return value * (max - min) + min
    elif (df is not None and categories_mins is not None
          and categories_maxs is not None and groupby_columns is not None):
        try:
            if isinstance(groupby_columns, list):
                return (value * (categories_maxs[tuple(df[groupby_columns])]
                        - categories_mins[tuple(df[groupby_columns])])
                        + categories_mins[tuple(df[groupby_columns])])
            else:
                return (value * (categories_maxs[df[groupby_columns]]
                        - categories_mins[df[groupby_columns]])
                        + categories_mins[df[groupby_columns]])
        except Exception:
            warnings.warn(f'Couldn\'t manage to find the mean and standard deviation values for the groupby columns {groupby_columns} with values {tuple(df[groupby_columns])}.')
            return np.nan
    else:
        raise Exception('ERROR: Invalid parameters. Either the `min` and `max` or the `df`, `categories_mins`, `categories_maxs` and `groupby_columns` must be set.')


def normalize_data(df, data=None, id_columns=['patientunitstayid', 'ts'],
                   normalization_method='z-score', columns_to_normalize=None,
                   columns_to_normalize_categ=None, categ_columns=None,
                   see_progress=True, inplace=False):
    '''Performs data normalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original Pandas or Dask dataframe which is used to correctly calculate the
        necessary statistical values used in the normalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the normalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be normalized
        by the specified normalization method. If the data tensor isn't
        specified, the normalization is applied directly on the dataframe.
    id_columns : string or list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be normalized.
    normalization_method : string, default 'z-score'
        Specifies the normalization method used. It can be a z-score
        normalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max normalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_normalize : string or list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be normalized. If set to False, no column will normalized directly,
        although columns can still be normalized in groups of categories, if
        specified in the `columns_to_normalize_categ` parameter. Otherwise, all
        continuous columns will be normalized.
    columns_to_normalize_categ : string or list of tuples of strings, default None
        If specified, the columns provided in the list are going to be
        normalized on their categories. That is, the values (column 2 in the
        tuple) are normalized with stats of their respective categories (column
        1 of the tuple). Otherwise, no column will be normalized on their
        categories.
    categ_columns : string or list of strings, default None
        If specified, the columns in the list, which represent categorical 
        features, which either are a label or will be embedded, aren't 
        going to be normalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the normalization calculations.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data : pandas.DataFrame or dask.DataFrame or torch.Tensor
        Normalized Pandas or Dask dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for normalization
    if columns_to_normalize is None:
        # List of all columns in the dataframe
        feature_columns = list(df.columns)
        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_normalize = feature_columns
        if id_columns is not None:
            # Make sure that the id_columns is a list
            if isinstance(id_columns, str):
                id_columns = [id_columns]
            if not isinstance(id_columns, list):
                raise Exception(f'ERROR: The `id_columns` argument must be specified as either a single string or a list of strings. Received input with type {type(id_columns)}.')
            # List of all columns in the dataframe, except the ID columns
            [columns_to_normalize.remove(col) for col in id_columns]
        if categ_columns is not None:
            # Make sure that the categ_columns is a list
            if isinstance(categ_columns, str):
                categ_columns = [categ_columns]
            if not isinstance(categ_columns, list):
                raise Exception(f'ERROR: The `categ_columns` argument must be specified as either a single string or a list of strings. Received input with type {type(categ_columns)}.')
            # Prevent all features that will be embedded from being normalized
            [columns_to_normalize.remove(col) for col in categ_columns]
        # List of binary or one hot encoded columns
        binary_cols = search_explore.list_one_hot_encoded_columns(df[columns_to_normalize])
        if binary_cols is not None:
            # Prevent binary features from being normalized
            [columns_to_normalize.remove(col) for col in binary_cols]
        # Remove all non numeric columns that could be left
        columns_to_normalize = [col for col in columns_to_normalize 
                                if df[col].dtype == np.number]
        if columns_to_normalize is None:
            print('No columns to normalize, returning the original dataframe.')
            return df

    # Make sure that the columns_to_normalize is a list
    if isinstance(columns_to_normalize, str):
        columns_to_normalize = [columns_to_normalize]
    if not isinstance(columns_to_normalize, list):
                raise Exception(f'ERROR: The `columns_to_normalize` argument must be specified as either a single string or a list of strings. Received input with type {type(columns_to_normalize)}.')

    if type(normalization_method) is not str:
        raise ValueError('Argument normalization_method should be a string. Available options are "z-score" and "min-max".')

    if normalization_method.lower() == 'z-score':
        if columns_to_normalize is not False:
            # Calculate the means and standard deviations
            means = df[columns_to_normalize].mean()
            stds = df[columns_to_normalize].std()

            if isinstance(df, dd.DataFrame):
                # Make sure that the values are computed, in case we're using Dask
                means = means.compute()
                stds = stds.compute()

        # Check if the data being normalized is directly the dataframe
        if data is None:
            if not inplace:
                # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
                data = df.copy()
            else:
                # Use the original dataframe
                data = df

            # Normalize the right columns
            if columns_to_normalize is not False:
                print(f'z-score normalizing columns {columns_to_normalize}...')
                data[columns_to_normalize] = (data[columns_to_normalize] - means) / stds

            if columns_to_normalize_categ is not None:
                # Make sure that the columns_to_normalize_categ is a list
                if isinstance(columns_to_normalize_categ, str):
                    columns_to_normalize_categ = [columns_to_normalize_categ]
                if not isinstance(columns_to_normalize_categ, list):
                    raise Exception(f'ERROR: The `columns_to_normalize_categ` argument must be specified as either a single string or a list of strings. Received input with type {type(columns_to_normalize_categ)}.')
                print(f'z-score normalizing columns {columns_to_normalize_categ} by their associated categories...')
                for col_tuple in utils.iterations_loop(columns_to_normalize_categ, see_progress=see_progress):
                    # Calculate the means and standard deviations
                    means = df.groupby(col_tuple[0])[col_tuple[1]].mean()
                    stds = df.groupby(col_tuple[0])[col_tuple[1]].std()

                    if isinstance(df, dd.DataFrame):
                        # Make sure that the values are computed, in case we're using Dask
                        means = means.compute()
                        stds = stds.compute()

                    categories_means = dict(means)
                    categories_stds = dict(stds)

                    # Normalize the right categories
                    if isinstance(df, dd.DataFrame):
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                     categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_norm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                     categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is normalized
        else:
            if columns_to_normalize is not False:
                # Dictionaries to retrieve the mean and standard deviation values
                column_means = dict(means)
                column_stds = dict(stds)
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))
                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])
                # List of indeces of the tensor's columns which are needing normalization
                tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]
                # Normalize the right columns
                print(f'z-score normalizing columns {columns_to_normalize}...')
                for col in utils.iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                    data[:, :, col] = ((data[:, :, col] - column_means[idx_to_name[col]])
                                       / column_stds[idx_to_name[col]])

    elif normalization_method.lower() == 'min-max':
        if columns_to_normalize is not False:
            mins = df[columns_to_normalize].min()
            maxs = df[columns_to_normalize].max()

            if isinstance(df, dd.DataFrame):
                # Make sure that the values are computed, in case we're using Dask
                mins = means.compute()
                maxs = maxs.compute()

        # Check if the data being normalized is directly the dataframe
        if data is None:
            if not inplace:
                # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
                data = df.copy()
            else:
                # Use the original dataframe
                data = df

            if columns_to_normalize is not False:
                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_normalize}...')
                data[columns_to_normalize] = (data[columns_to_normalize] - mins) / (maxs - mins)

            if columns_to_normalize_categ is not None:
                print(f'min-max normalizing columns {columns_to_normalize_categ} by their associated categories...')
                for col_tuple in columns_to_normalize_categ:
                    # Calculate the means and standard deviations
                    mins = df.groupby(col_tuple[0])[col_tuple[1]].min()
                    maxs = df.groupby(col_tuple[0])[col_tuple[1]].max()

                    if isinstance(df, dd.DataFrame):
                        # Make sure that the values are computed, in case we're using Dask
                        mins = mins.compute()
                        maxs = maxs.compute()

                    categories_mins = dict(mins)
                    categories_maxs = dict(maxs)
                    # Normalize the right categories
                    if isinstance(df, dd.DataFrame):
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                     categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_norm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                     categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1)
        # Otherwise, the tensor is normalized
        else:
            if columns_to_normalize is not False:
                # Dictionaries to retrieve the min and max values
                column_mins = dict(mins)
                column_maxs = dict(maxs)
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))
                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])
                # List of indeces of the tensor's columns which are needing normalization
                tensor_columns_to_normalize = [name_to_idx[name] for name in columns_to_normalize]
                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_normalize}...')
                for col in utils.iterations_loop(tensor_columns_to_normalize, see_progress=see_progress):
                    data[:, :, col] = ((data[:, :, col] - column_mins[idx_to_name[col]])
                                       / (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]]))
    else:
        raise ValueError(f'{normalization_method} isn\'t a valid normalization method. Available options \
                         are "z-score" and "min-max".')
    return data


def denormalize_data(df, data=None, id_columns=['patientunitstayid', 'ts'],
                     denormalization_method='z-score', columns_to_denormalize=None,
                     columns_to_denormalize_cat=None, categ_columns=None,
                     see_progress=True, inplace=False):
    '''Performs data denormalization to a continuous valued tensor or dataframe,
       changing the scale of the data.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Original Pandas or Dask dataframe which is used to correctly calculate the
        necessary statistical values used in the denormalization. These values
        can't be calculated from the tensor as it might have been padded. If
        the data tensor isn't specified, the denormalization is applied directly
        on the dataframe.
    data : torch.Tensor, default None
        PyTorch tensor corresponding to the data which will be denormalized
        by the specified denormalization method. If the data tensor isn't
        specified, the denormalization is applied directly on the dataframe.
    id_columns : list of strings, default ['subject_id', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be denormalized.
    denormalization_method : string, default 'z-score'
        Specifies the denormalization method used. It can be a z-score
        denormalization, where the data is subtracted of it's mean and divided
        by the standard deviation, which makes it have zero average and unit
        variance, much like a standard normal distribution; it can be a
        min-max denormalization, where the data is subtracted by its minimum
        value and then divided by the difference between the minimum and the
        maximum value, getting to a fixed range from 0 to 1.
    columns_to_denormalize : list of strings, default None
        If specified, the columns provided in the list are the only ones that
        will be denormalized. If set to False, no column will denormalized directly,
        although columns can still be denormalized in groups of categories, if
        specified in the `columns_to_denormalize_cat` parameter. Otherwise, all
        continuous columns will be denormalized.
    columns_to_denormalize_cat : list of tuples of strings, default None
        If specified, the columns provided in the list are going to be
        denormalized on their categories. That is, the values (column 2 in the
        tuple) are denormalized with stats of their respective categories (column
        1 of the tuple). Otherwise, no column will be denormalized on their
        categories.
    categ_columns : list of strings, default None
        If specified, the columns in the list, which represent features that
        will be embedded, aren't going to be denormalized.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of the denormalization calculations.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data : pandas.DataFrame or dask.DataFrame or torch.Tensor
        Normalized Pandas or Dask dataframe or PyTorch tensor.
    '''
    # Check if specific columns have been specified for denormalization
    if columns_to_denormalize is None:
        # List of all columns in the dataframe
        feature_columns = list(df.columns)
        # Normalize all non identifier continuous columns, ignore one hot encoded ones
        columns_to_denormalize = feature_columns

        # List of all columns in the dataframe, except the ID columns
        [columns_to_denormalize.remove(col) for col in id_columns]

        if categ_columns is not None:
            # Prevent all features that will be embedded from being denormalized
            [columns_to_denormalize.remove(col) for col in categ_columns]

        # List of binary or one hot encoded columns
        binary_cols = search_explore.list_one_hot_encoded_columns(df[columns_to_denormalize])

        if binary_cols is not None:
            # Prevent binary features from being denormalized
            [columns_to_denormalize.remove(col) for col in binary_cols]

        if columns_to_denormalize is None:
            print('No columns to denormalize, returning the original dataframe.')
            return df

    if type(denormalization_method) is not str:
        raise ValueError('Argument denormalization_method should be a string. Available options \
                         are "z-score" and "min-max".')

    if denormalization_method.lower() == 'z-score':
        if columns_to_denormalize is not False:
            # Calculate the means and standard deviations
            means = df[columns_to_denormalize].mean()
            stds = df[columns_to_denormalize].std()

            if isinstance(df, dd.DataFrame):
                # Make sure that the values are computed, in case we're using Dask
                means = means.compute()
                stds = stds.compute()

        # Check if the data being denormalized is directly the dataframe
        if data is None:
            if not inplace:
                # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
                data = df.copy()
            else:
                # Use the original dataframe
                data = df

            # Normalize the right columns
            if columns_to_denormalize is not False:
                print(f'z-score normalizing columns {columns_to_denormalize}...')
                for col in utils.iterations_loop(columns_to_denormalize, see_progress=see_progress):
                    data[col] = data[col] * column_stds[col] + column_means[col]

            if columns_to_denormalize_cat is not None:
                print(f'z-score normalizing columns {columns_to_denormalize_cat} by their associated categories...')
                for col_tuple in utils.iterations_loop(columns_to_denormalize_cat, see_progress=see_progress):
                    # Calculate the means and standard deviations
                    means = df.groupby(col_tuple[0])[col_tuple[1]].mean()
                    stds = df.groupby(col_tuple[0])[col_tuple[1]].std()

                    if isinstance(df, dd.DataFrame):
                        # Make sure that the values are computed, in case we're using Dask
                        means = means.compute()
                        stds = stds.compute()

                    categories_means = dict(means)
                    categories_stds = dict(stds)

                    # Normalize the right categories
                    if isinstance(df, dd.DataFrame):
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                       categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_zscore_denorm(value=df[col_tuple[1]], df=df, categories_means=categories_means,
                                                                                       categories_stds=categories_stds, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is denormalized
        else:
            if columns_to_denormalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing denormalization
                tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_denormalize]

                # Normalize the right columns
                print(f'z-score normalizing columns {columns_to_denormalize}...')
                for col in utils.iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                    data[:, :, col] = (data[:, :, col] * column_stds[idx_to_name[col]]
                                       + column_means[idx_to_name[col]])

    elif denormalization_method.lower() == 'min-max':
        if columns_to_denormalize is not False:
            mins = df[columns_to_denormalize].min()
            maxs = df[columns_to_denormalize].max()

            if isinstance(df, dd.DataFrame):
                # Make sure that the values are computed, in case we're using Dask
                mins = means.compute()
                maxs = maxs.compute()

        # Check if the data being denormalized is directly the dataframe
        if data is None:
            if not inplace:
                # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
                data = df.copy()
            else:
                # Use the original dataframe
                data = df

            if columns_to_denormalize is not False:
                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_denormalize}...')
                for col in utils.iterations_loop(columns_to_denormalize, see_progress=see_progress):
                    data[col] = (data[col] * (column_maxs[col] - column_mins[col])
                                 + column_mins[col])

            if columns_to_denormalize_cat is not None:
                print(f'min-max normalizing columns {columns_to_denormalize_cat} by their associated categories...')
                for col_tuple in columns_to_denormalize_cat:
                    # Calculate the means and standard deviations
                    mins = df.groupby(col_tuple[0])[col_tuple[1]].min()
                    maxs = df.groupby(col_tuple[0])[col_tuple[1]].max()

                    if isinstance(df, dd.DataFrame):
                        # Make sure that the values are computed, in case we're using Dask
                        mins = mins.compute()
                        maxs = maxs.compute()

                    categories_mins = dict(mins)
                    categories_maxs = dict(maxs)

                    # Normalize the right categories
                    if isinstance(df, dd.DataFrame):
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                       categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1, meta=('df', float))
                    else:
                        data[col_tuple[1]] = data.apply(lambda df: apply_minmax_denorm(value=df[col_tuple[1]], df=df, categories_mins=categories_mins,
                                                                                       categories_maxs=categories_maxs, groupby_columns=col_tuple[0]),
                                                        axis=1)

        # Otherwise, the tensor is denormalized
        else:
            if columns_to_denormalize is not False:
                # Dictionary to convert the the tensor's column indeces into the dataframe's column names
                idx_to_name = dict(enumerate(df.columns))

                # Dictionary to convert the dataframe's column names into the tensor's column indeces
                name_to_idx = dict([(t[1], t[0]) for t in enumerate(df.columns)])

                # List of indeces of the tensor's columns which are needing denormalization
                tensor_columns_to_denormalize = [name_to_idx[name] for name in columns_to_denormalize]

                # Normalize the right columns
                print(f'min-max normalizing columns {columns_to_denormalize}...')
                for col in utils.iterations_loop(tensor_columns_to_denormalize, see_progress=see_progress):
                    data[:, :, col] = (data[:, :, col] * (column_maxs[idx_to_name[col]] - column_mins[idx_to_name[col]])
                                       + column_mins[idx_to_name[col]])

    else:
        raise ValueError(f'{denormalization_method} isn\'t a valid denormalization method. Available options \
                         are "z-score" and "min-max".')

    return data


def transpose_dataframe(df, column_to_transpose=None, inplace=False):
    '''Transpose a dataframe, either by its original index or through a specific
    column, which will be converted to the new column names (i.e. the header).

    Parameters
    ----------
    data : pandas.DataFrame or dask.DataFrame
        Dataframe that will be transposed.
    column_to_transpose : string, default None
        If specified, the given column will be used as the new column names, with
        its unique values forming the new dataframe's header. Otherwise, the
        dataframe will be transposed on its original index.
    inplace : bool, default False
        If set to True, the original tensor or dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original tensor or dataframe.

    Returns
    -------
    data : pandas.DataFrame or dask.DataFrame
        Transposed dataframe.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    if column_to_transpose is not None:
        # Set as index the column that has the desired column names as values
        data_df = data_df.set_index(column_to_transpose)
    if isinstance(data_df, pd.DataFrame):
        data_df = data_df.transpose()
    elif isinstance(data_df, dd.DataFrame):
        data_df = (dd.from_pandas(data_df.compute().transpose(), 
                                  npartitions=data_df.npartitions))
    else:
        raise Exception(f'ERROR: The input data must either be a Pandas dataframe or a Dask dataframe, not {type(df)}.')
    return data_df


def missing_values_imputation(data, method='zero', id_column=None, inplace=False):
    '''Performs missing values imputation to a tensor or dataframe corresponding to
    a single column.

    Parameters
    ----------
    data : torch.Tensor or pandas.DataFrame or dask.DataFrame
        PyTorch tensor corresponding to a single column or a dataframe which will 
        be imputed.
    method : string, default 'zero'
        Imputation method to be used. If user inputs 'zero', it will just fill all
        missing values with zero. If the user chooses 'zigzag', it will do a 
        forward fill, a backward fill and then replace all remaining missing values
        with zero (this option is only available for dataframes, not tensors).
        If the user selects 'interpolation', missing data will be interpolated based 
        on known neighboring values and then all possible remaining ones are
        replaced with zero (this option is only available for dataframes, not
        tensors).
    id_column : string, default None
        Name of the column which corresponds to the sequence or subject identifier
        in the dataframe. If not specified, the imputation will not differenciate
        different IDs nor sequences. Only used if the chosen imputation method is
        'zigzag' or 'interpolation'.
    inplace : bool, default False
        If set to True, the original tensor or dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original tensor or dataframe.

    Returns
    -------
    tensor : torch.Tensor
        Imputed PyTorch tensor.
    '''
    if ((not isinstance(data, pd.DataFrame))
         and (not isinstance(data, dd.DataFrame))
         and (not isinstance(data, torch.Tensor))):
        raise Exception(f'ERROR: The input data must either be a PyTorch tensor, a Pandas dataframe or a Dask dataframe, not {type(data)}.')
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original data
        if isinstance(data, torch.Tensor):
            data_copy = data.clone()
        else:
            data_copy = data.copy()
    else:
        # Use the original data object
        data_copy = data
    if method.lower() == 'zero':
        # Replace NaN's with zeros
        if isinstance(data, pd.DataFrame) or isinstance(data, dd.DataFrame):
            data_copy = data_copy.fillna(value=0)
        elif isinstance(data, torch.Tensor):
            data_copy = torch.where(data_copy != data_copy, torch.zeros_like(data_copy), data_copy)
    elif method.lower() == 'zigzag':
        if isinstance(data, pd.DataFrame) or isinstance(data, dd.DataFrame):
            if id_column is not None:
                # Perform imputation on each ID separately
                # Forward fill
                data_copy = (data_copy.set_index(id_column, append=True).groupby(id_column)
                            .fillna(method='ffill').reset_index(level=1))
                # Backward fill
                data_copy = (data_copy.set_index(id_column, append=True).groupby(id_column)
                            .fillna(method='bfill').reset_index(level=1))
                # Replace remaining missing values with zero
                data_copy = data_copy.fillna(value=0)
            else:
                # Apply imputation on all the data as one single sequence
                # Forward fill
                data_copy = data_copy.fillna(method='ffill')
                # Backward fill
                data_copy = data_copy.fillna(method='bfill')
                # Replace remaining missing values with zero
                data_copy = data_copy.fillna(value=0)
        elif isinstance(data, torch.Tensor):
            raise Exception('ERROR: PyTorch tensors aren\'t supported in the zigzag imputation method. Please use a dataframe instead.')
    elif method.lower() == 'interpolation':
        if isinstance(data, pd.DataFrame) or isinstance(data, dd.DataFrame):
            # Linear interpolation, placing a linear scale between known points and doing simple
            # backward and forward fill, when the missing value doesn't have known data points
            # before or after, respectively
            if id_column is not None:
                # Perform imputation on each ID separately
                data_copy = (data_copy.set_index(id_column, append=True).groupby(id_column)
                                      .apply(lambda group: group.interpolate(limit_direction='both'))
                                      .reset_index(level=1))
                # Replace remaining missing values with zero
                data_copy = data_copy.fillna(value=0)
            else:
                # Apply imputation on all the data as one single sequence
                data_copy = data_copy.interpolate(limit_direction='both')
                # Replace remaining missing values with zero
                data_copy = data_copy.fillna(value=0)
        elif isinstance(data, torch.Tensor):
            raise Exception('ERROR: PyTorch tensors aren\'t supported in the interpolation imputation method. Please use a dataframe instead.')
    else:
        raise Exception(f'ERROR: Unsupported {method} imputation method. Currently available options are `zero` and `zigzag`.')
    # [TODO] Add other, more complex imputation methods, like a denoising autoencoder
    return data_copy


def set_dosage_and_units(df, orig_column='dosage'):
    '''Separate medication dosage string column into numeric dosage and units
    features.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe containing the medication dosage information.
    orig_column : string, default 'dosage'
        Name of the original column, which will be split in two.

    Returns
    -------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe after adding the numeric dosage and units columns.
    '''
    # Start by assuming that dosage and unit are unknown
    dosage = np.nan
    unit = np.nan
    try:
        x = df[orig_column].split(' ')
        if len(x) == 2:
            try:
                x[0] = float(x[0])
            except Exception:
                return dosage, unit
            if utils.is_definitely_string(x[1]):
                # Add correctly formated dosage and unit values
                dosage = x[0]
                unit = x[1]
                return dosage, unit
            else:
                return dosage, unit
        else:
            return dosage, unit
    except Exception:
        return dosage, unit


def signal_idx_derivative(s, time_scale='seconds', periods=1):
    '''Creates a series that contains the signal's index derivative, with the
    same divisions (if needed) as the original data and on the desired time
    scale.

    Parameters
    ----------
    s : pandas.Series or dask.Series
        Series which will be analyzed for outlier detection.
    time_scale : bool, default 'seconds'
        How to calculate derivatives, either with respect to the index values,
        on the time scale of 'seconds', 'minutes', 'hours', 'days', 'months' or
        'years', or just sequentially, just getting the difference between
        consecutive values, 'False'. Only used if parameter 'signal' isn't set
        to 'value'.
    periods : int, default 1
        Defines the steps to take when calculating the derivative. When set to 1,
        it performs a normal backwards derivative. When set to 1, it performs a
        normal forwards derivative.

    Returns
    -------
    s_idx : pandas.Series or dask.Series
        Index derivative signal, on the desired time scale.
    '''
    # Calculate the signal index's derivative
    s_idx = s.index.to_series().diff()
    if isinstance(s_idx, dd.DataFrame):
        # Make the new derivative have the same divisions as the original signal
        s_idx = (s_idx.to_frame().rename(columns={s.index.name:'tmp_val'})
                      .reset_index()
                      .set_index(s.index.name, sorted=True, divisions=s.divisions)
                      .tmp_val)
    # Convert derivative to the desired time scale
    if time_scale == 'seconds':
        s_idx = s_idx.dt.seconds
    elif time_scale == 'minutes':
        s_idx = s_idx.dt.seconds / 60
    elif time_scale == 'hours':
        s_idx = s_idx.dt.seconds / 3600
    elif time_scale == 'days':
        s_idx = s_idx.dt.seconds / 86400
    elif time_scale == 'months':
        s_idx = s_idx.dt.seconds / 2592000
    return s_idx


def threshold_outlier_detect(s, max_thrs=None, min_thrs=None, threshold_type='absolute',
                             signal_type='value', time_scale='seconds',
                             derivate_direction='backwards'):
    '''Detects outliers based on predetermined thresholds.

    Parameters
    ----------
    s : pandas.Series or dask.Series
        Series which will be analyzed for outlier detection.
    max_thrs : int or float, default None
        Maximum threshold, i.e. no normal value can be larger than this
        threshold, in the signal (or its n-order derivative) that we're
        analyzing.
    min_thrs : int or float, default None
        Minimum threshold, i.e. no normal value can be smaller than this
        threshold, in the signal (or its n-order derivative) that we're
        analyzing.
    threshold_type : string, default 'absolute'
        Determines if we're using threshold values with respect to the original
        scale of values, 'absolute', relative to the signal's mean, 'mean' or
        'average', to the median, 'median' or to the standard deviation, 'std'.
        As such, the possible settings are ['absolute', 'mean', 'average',
        'median', 'std'].
    signal_type : string, default 'value'
        Sets if we're analyzing the original signal value, 'value', its first
        derivative, 'derivative' or 'speed', or its second derivative, 'second
        derivative' or 'acceleration'. As such, the possible settings are
        ['value', 'derivative', 'speed', 'second derivative', 'acceleration'].
    time_scale : string or bool, default 'seconds'
        How to calculate derivatives, either with respect to the index values,
        on the time scale of 'seconds', 'minutes', 'hours', 'days', 'months' or
        'years', or just sequentially, just getting the difference between
        consecutive values, 'False'. Only used if parameter 'signal' isn't set
        to 'value'.
    derivate_direction : string, default 'backwards'
        The direction in which we calculate the derivative, either comparing to
        previous values, 'backwards', or to the next values, 'forwards'. As such,
        the possible settings are ['backwards', 'forwards']. Only used if
        parameter 'signal' isn't set to 'value'.

    Returns
    -------
    outlier_s : pandas.Series or dask.Series
        Boolean series indicating where the detected outliers are.
    '''
    if signal_type.lower() == 'value':
        signal = s
    elif signal_type.lower() == 'derivative' or signal_type.lower() == 'speed':
        if derivate_direction.lower() == 'backwards':
            periods = 1
        elif derivate_direction.lower() == 'forwards':
            periods = -1
        else:
            raise Exception(f'ERROR: Invalid derivative direction. It must either be "backwards" or "forwards", not {derivate_direction}.')
        # Calculate the difference between consecutive values
        signal = s.diff(periods)
        if time_scale is not None:
            # Derivate by the index values
            signal = signal / signal_idx_derivative(signal, time_scale, periods)
    elif (signal_type.lower() == 'second derivative'
          or signal_type.lower() == 'acceleration'):
        if derivate_direction.lower() == 'backwards':
            periods = 1
        elif derivate_direction.lower() == 'forwards':
            periods = -1
        else:
            raise Exception(f'ERROR: Invalid derivative direction. It must either be "backwards" or "forwards", not {derivate_direction}.')
        # Calculate the difference between consecutive values
        signal = s.diff(periods).diff(periods)
        if time_scale is not None:
            # Derivate by the index values
            signal = signal / signal_idx_derivative(signal, time_scale, periods)
    else:
        raise Exception('ERROR: Invalid signal type. It must be "value", "derivative", "speed", "second derivative" or "acceleration", not {signal}.')
    if threshold_type.lower() == 'absolute':
        signal = signal
    elif threshold_type.lower() == 'mean' or threshold_type.lower() == 'average':
        signal_mean = signal.mean()
        if isinstance(signal, dd.DataFrame):
            # Make sure that the value is computed, in case we're using Dask
            signal_mean = signal_mean.compute()
        # Normalize by the average value
        signal = signal / signal_mean
    elif threshold_type.lower() == 'median':
        if isinstance(signal, dd.DataFrame):
            # Make sure that the value is computed, in case we're using Dask
            signal_median = signal.compute().median()
        else:
            signal_median = signal.median()
        # Normalize by the median value
        signal = signal / signal_median
    elif threshold_type.lower() == 'std':
        signal_mean = signal.mean()
        signal_std = signal.std()
        if isinstance(signal, dd.DataFrame):
            # Make sure that the values are computed, in case we're using Dask
            signal_mean = signal_mean.compute()
            signal_std = signal_std.compute()
        # Normalize by the average and standard deviation values
        signal = (signal - signal_mean) / signal_std
    else:
        raise Exception(f'ERROR: Invalid value type. It must be "absolute", "mean", "average", "median" or "std", not {threshold_type}.')

    # Search for outliers based on the given thresholds
    if max_thrs is not None and min_thrs is not None:
        outlier_s = (signal > max_thrs) | (signal < min_thrs)
    elif max_thrs is not None:
        outlier_s = signal > max_thrs
    elif min_thrs is not None:
        outlier_s = signal < min_thrs
    else:
        raise Exception('ERROR: At least a maximum or a minimum threshold must be set. Otherwise, no outlier will ever be detected.')

    return outlier_s


def slopes_outlier_detect(s, max_thrs=4, bidir_sens=0.5, threshold_type='std',
                          time_scale='seconds', only_bir=False):
    '''Detects outliers based on large variations on the signal's derivatives,
    either in one direction or on both at the same time.

    Parameters
    ----------
    s : pandas.Series or dask.Series
        Series which will be analyzed for outlier detection.
    max_thrs : int or float
        Maximum threshold, i.e. no point can have a magnitude derivative value
        deviate more than this threshold, in the signal that we're analyzing.
    bidir_sens : float, default 0.5
        Dictates how much more sensitive the algorithm is when a deviation (i.e.
        large variation) is found on both sides of the data point / both
        directions of the derivative. In other words, it's a factor that will be
        multiplied by the usual one-directional threshold (`max_thrs`), from which
        the resulting value will be used as the bidirectional threshold.
    threshold_type : string, default 'std'
        Determines if we're using threshold values with respect to the original
        scale of derivative values, 'absolute', relative to the derivative's
        mean, 'mean' or 'average', to the median, 'median' or to the standard
        deviation, 'std'. As such, the possible settings are ['absolute', 'mean',
        'average', 'median', 'std'].
    time_scale : string or bool, default 'seconds'
        How to calculate derivatives, either with respect to the index values,
        on the time scale of 'seconds', 'minutes', 'hours', 'days', 'months' or
        'years', or just sequentially, just getting the difference between
        consecutive values, 'False'. Only used if parameter 'signal' isn't set
        to 'value'.
    only_bir : bool, default False
        If set to True, the algorithm will only check for data points that have
        large derivatives on both directions.

    Returns
    -------
    outlier_s : pandas.Series or dask.Series
        Boolean series indicating where the detected outliers are.
    '''
    # Calculate the difference between consecutive values
    bckwrds_deriv = s.diff()
    frwrds_deriv = s.diff(-1)
    if time_scale is not None:
        # Derivate by the index values
        bckwrds_deriv = bckwrds_deriv / signal_idx_derivative(bckwrds_deriv, time_scale, periods=1)
        frwrds_deriv = frwrds_deriv / signal_idx_derivative(frwrds_deriv, time_scale, periods=-1)
    if threshold_type.lower() == 'absolute':
        bckwrds_deriv = bckwrds_deriv
        frwrds_deriv = frwrds_deriv
    elif threshold_type.lower() == 'mean' or threshold_type.lower() == 'average':
        bckwrds_deriv_mean = bckwrds_deriv.mean()
        frwrds_deriv_mean = frwrds_deriv.mean()
        if isinstance(bckwrds_deriv, dd.DataFrame):
            # Make sure that the value is computed, in case we're using Dask
            bckwrds_deriv_mean = bckwrds_deriv_mean.compute()
            frwrds_deriv_mean = frwrds_deriv_mean.compute()
        # Normalize by the average value
        bckwrds_deriv = bckwrds_deriv / bckwrds_deriv_mean
        frwrds_deriv = frwrds_deriv / frwrds_deriv_mean
    elif threshold_type.lower() == 'median':
        bckwrds_deriv_median = bckwrds_deriv.median()
        frwrds_deriv_median = frwrds_deriv.median()
        if isinstance(bckwrds_deriv, dd.DataFrame):
            # Make sure that the value is computed, in case we're using Dask
            bckwrds_deriv_median = bckwrds_deriv_median.compute()
            frwrds_deriv_median = frwrds_deriv_median.compute()
        # Normalize by the median value
        bckwrds_deriv = bckwrds_deriv / bckwrds_deriv_median
        frwrds_deriv = frwrds_deriv / frwrds_deriv_median
    elif threshold_type.lower() == 'std':
        bckwrds_deriv_mean = bckwrds_deriv.mean()
        frwrds_deriv_mean = frwrds_deriv.mean()
        bckwrds_deriv_std = bckwrds_deriv.std()
        frwrds_deriv_std = frwrds_deriv.std()
        if isinstance(bckwrds_deriv, dd.DataFrame):
            # Make sure that the values are computed, in case we're using Dask
            bckwrds_deriv_mean = bckwrds_deriv_mean.compute()
            frwrds_deriv_mean = frwrds_deriv_mean.compute()
            bckwrds_deriv_std = bckwrds_deriv_std.compute()
            frwrds_deriv_std = frwrds_deriv_std.compute()
        # Normalize by the average and standard deviation values
        bckwrds_deriv = (bckwrds_deriv - bckwrds_deriv_mean) / bckwrds_deriv_std
        frwrds_deriv = (frwrds_deriv - frwrds_deriv_mean) / frwrds_deriv_std
    else:
        raise Exception('ERROR: Invalid value type. It must be "absolute", "mean", "average", "median" or "std", not {threshold_type}.')

    # Bidirectional threshold, to be used when observing both directions of the derivative
    bidir_max = bidir_sens * max_thrs
    if only_bir is True:
        # Search for outliers on both derivatives at the same time, always on their respective magnitudes
        outlier_s = (bckwrds_deriv.abs() > bidir_max) & (frwrds_deriv.abs() > bidir_max)
    else:
        # Search for outliers on each individual derivative, followed by both at the same time with a lower threshold, always on their respective magnitudes
        outlier_s = ((bckwrds_deriv.abs() > max_thrs) | (frwrds_deriv.abs() > max_thrs)
                     | ((bckwrds_deriv.abs() > bidir_max) & (frwrds_deriv.abs() > bidir_max)))
    return outlier_s
