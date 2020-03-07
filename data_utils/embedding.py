from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import dask.dataframe as dd                             # Dask to handle big data in dataframes
import numpy as np                                      # NumPy to handle numeric and NaN operations
import numbers                                          # numbers allows to check if data is numeric
from functools import reduce                            # Parallelize functions
import re                                               # Methods for string editing and searching, regular expression matching operations
import warnings                                         # Print warnings for bad practices
from . import utils                                     # Generic and useful methods
from . import data_processing                           # Data processing and dataframe operations
from . import deep_learning                                    # Common and generic deep learning related methods
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def remove_digit_from_dict(enum_dict, forbidden_digit=0, inplace=False):
    '''Convert an enumeration dictionary to a representation that doesn't
    include any value with a specific digit.

    Parameters
    ----------
    enum_dict : dict
        Dictionary containing the mapping between the original values and a
        numbering.
    forbidden_digit : int, default 0
        Digit that we want to prevent from appearing in any enumeration
        encoding.
    inplace : bool, default False
        If set to True, the original dictionary will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dictionary.

    Returns
    -------
    enum_dict : dict
        Dictionary containing the mapping between the original values and a
        numbering. Now without any occurence of the specified digit.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dictionary
        new_enum_dict = enum_dict.copy()
    else:
        # Use the original dictionary
        new_enum_dict = enum_dict
    # Create a sequence of enumeration encoding values
    enum_seq = []
    # Value that represents the current sequence number
    num = 1
    # Digit to be used when replacing the forbidden digit
    alt_digit = forbidden_digit + 1
    for i in range(len(enum_dict)):
        # Replace undesired digit with the alternative one
        num = str(num).replace(str(forbidden_digit), str(alt_digit))
        # Add to the enumeration sequence
        num = int(num)
        enum_seq.append(num)
        # Increment to the following number
        num += 1
    # Create a dictionary to convert regular enumeration into the newly created
    # sequence
    old_to_new_dict = dict(enumerate(enum_seq, start=1))
    # Convert the enumeration dictionary to the new encoding scheme
    for key, val in enum_dict.items():
        new_enum_dict[key] = old_to_new_dict[val]
    return new_enum_dict


def create_enum_dict(unique_values, nan_value=None, forbidden_digit=None):
    '''Enumerate all categories in a specified categorical feature, while also
    attributing a specific number to NaN and other unknown values.

    Parameters
    ----------
    unique_values : list of strings
        Specifies all the unique values to be enumerated.
    nan_value : int, default None
        Integer number that gets assigned to NaN and NaN-like values.
    forbidden_digit : int, default None
        Digit that we want to prevent from appearing in any enumeration
        encoding.

    Returns
    -------
    enum_dict : dict
        Dictionary containing the mapping between the original values and the
        numbering obtained.
    '''
    # Enumerate the unique values in the categorical feature and put them in a dictionary
    if nan_value is not None:
        if nan_value is 0:
            enum_dict = dict(enumerate(unique_values, start=1))
        else:
            enum_dict = dict(enumerate(unique_values, start=0))
    else:
        enum_dict = dict(enumerate(unique_values, start=0))
    # Invert the dictionary to have the unique categories as keys and the numbers as values
    enum_dict = utils.invert_dict(enum_dict)
    if forbidden_digit is not None:
        # Change the enumeration to prevent it from including undesired digits
        enum_dict = remove_digit_from_dict(enum_dict, forbidden_digit, inplace=True)
    # Move NaN to key 0
    enum_dict[np.nan] = nan_value
    # Search for NaN-like categories
    for key, val in enum_dict.items():
        if type(key) is str:
            if utils.is_string_nan(key):
                # Move NaN-like key to nan_value
                enum_dict[key] = nan_value
        elif isinstance(key, numbers.Number):
            if np.isnan(key) or str(key).lower() == 'nan':
                # Move NaN-like key to nan_value
                enum_dict[key] = nan_value
    return enum_dict


def enum_categorical_feature(df, feature, nan_value=None, clean_name=True,
                             forbidden_digit=None, separator='0', apply_on_df=True):
    '''Enumerate all categories in a specified categorical feature, while also
    attributing a specific number to NaN and other unknown values.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which the categorical feature belongs to.
    feature : string
        Name of the categorical feature which will be enumerated.
    nan_value : int, default None
        Integer number that gets assigned to NaN and NaN-like values.
    clean_name : bool, default True
        If set to True, the method assumes that the feature is of type string
        and it will make sure that all the feature's values are in lower case,
        to reduce duplicate information.
    forbidden_digit : int, default None
        Digit that we want to prevent from appearing in any enumeration
        encoding.
    separator : string, default '0'
        Symbol that concatenates each string's words. As such, it can't appear
        in a single category's string.
    apply_on_df : bool, default True
        If set to True, the original column of the dataframe will be converted
        to the new enumeration encoding.

    Returns
    -------
    enum_series : pandas.Series or dask.Series
        Series corresponding to the analyzed feature, after
        enumeration.
    enum_dict : dict
        Dictionary containing the mapping between the original categorical values
        and the numbering obtained.
    '''
    if clean_name is True:
        # Clean the column's string values to have the same, standard format
        df = data_processing.clean_categories_naming(df, feature, separator=separator)
    # Get the unique values of the cateforical feature
    unique_values = df[feature].unique()
    if isinstance(df, dd.DataFrame):
        # Make sure that the unique values are computed, in case we're using Dask
        unique_values = unique_values.compute()
    # Enumerate the unique values in the categorical feature and put them in a dictionary
    enum_dict = create_enum_dict(unique_values, nan_value, forbidden_digit)
    if apply_on_df is False:
        return enum_dict
    else:
        # Create a series from the enumerations of the original feature's categories
        if isinstance(df, dd.DataFrame):
            enum_series = df[feature].map(lambda x: utils.apply_dict_convertion(x, enum_dict, nan_value), meta=('x', int))
        else:
            enum_series = df[feature].map(lambda x: utils.apply_dict_convertion(x, enum_dict, nan_value))
        return enum_series, enum_dict


def enum_category_conversion(df, enum_column, enum_dict, enum_to_category=None,
                             separator='0'):
    '''Convert between enumerated encodings and their respective categories'
    names, in either direction.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which the categorical feature belongs to.
    enum_column : string
        Name of the categorical feature which is encoded/enumerated. The
        feature's values must be single integer numbers, with a separator symbol
        if more than one category applies to a given row.
    enum_dict : dict
        Dictionary containing the category names that correspond to each
        enumeration number.
    enum_to_category : bool, default None
        Indicator on which the user can specify if the conversion is from
        numerical encodings to string categories names (True) or vice-versa
        (False). By default, it's not defined (None) and the method infers the
        direction of the conversion based on the input dictionary's key type.
    separator : string, default '0'
        Symbol that concatenates each string's words.

    Returns
    -------
    categories : string
        String containing all the categories names of the current row. If more
        than one category is present, their names are separated by the '0'
        separator.
    '''
    if enum_to_category is None:
        # If all the keys are integers, then we're converting from enumerations to category names;
        # otherwise, it's the opposite direction
        enum_to_category = all([isinstance(item, int) for item in list(enum_dict.keys())])
    # Check what direction the conversion is being done
    if enum_to_category is False:
        # Separate the enumerations
        enums = str(df[enum_column]).split(separator)
        # Get the individual categories names
        categories = [str(enum_dict[str(n)]) for n in enums]
    else:
        if str(df[enum_column]).lower() == 'nan':
            return enum_dict['nan']
        else:
            enums_val = int(df[enum_column])
            if str(enums_val) == separator:
                # Return the current category, avoiding its removal with the following split operation
                return enum_dict[enums_val]
            # Separate the enumerations
            enums = str(int(df[enum_column])).split(separator)
            # Get the individual categories names
            categories = [enum_dict[int(float(n))] for n in enums]
    # Join the categories by the designated separator symbol
    categories = separator.join(categories)
    return categories


def converge_enum(df1, cat_feat_name, df2=None, dict1=None, dict2=None,
                  nan_value=None, forbidden_digit=None, sort=True, separator='0',
                  inplace=False):
    '''Converge the categorical encoding (enumerations) on the same feature of
    two dataframes.

    Parameters
    ----------
    df1 : pandas.DataFrame or dask.DataFrame
        A dataframe that has the enumerated categorical feature, which
        encoding needs to be converged with the other.
    cat_feat_name : string or list of strings
        Name of the categorical feature whose encodings need to be converged. If
        the user wants to converge two columns from the same dataframe, both
        names must be defined in a list format
        (e.g. cat_feat_name = ['var_x', 'var_y']), and variable `df2` has to be
        None.
    df2 : pandas.DataFrame or dask.DataFrame, default None
        A second dataframe that has the enumerated categorical feature, which
        encoding needs to be converged with the other. If not specified, only
        dataframe `df1` will be used, so the user must provide two feature names
        from this dataframe to converge (e.g. cat_feat_name = ['var_x', 'var_y']).
    dict1 : dict, default None
        Dictionary mapping between the category names and the first dataframe's
        encoding number. If not specified, the method will create the dictionary.
    dict2 : dict, default None
        Dictionary mapping between the category names and the second dataframe's
        encoding number. If not specified, the method will create the dictionary.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    forbidden_digit : int, default None
        Digit that we want to prevent from appearing in any enumeration
        encoding.
    sort : bool, default True
        If set to True, the final dictionary of mapping between categories names
        and enumeration numbers will be sorted alphabetically. In case sorting
        is used, the resulting dictionary and dataframes will always be the same.
    separator : string, default '0'
        Symbol that concatenates each string's words.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data1_df : pandas.DataFrame or dask.DataFrame
        The first input dataframe after having its categorical feature converted
        to the new, converged enumeration encoding.
    all_data_dict : dict, default None
        New dictionary that maps both dataframes' unique categories to the
        converged enumeration encoding. Remember to save this dictionary, as
        this converged dictionary creation process is stochastic, if sorting is
        not performed.

    If df2 is None:

    data2_df : pandas.DataFrame or dask.DataFrame
        The second input dataframe after having its categorical feature
        converted to the new, converged enumeration encoding.
    '''
    if isinstance(cat_feat_name, str):
        # Use just one feature name
        cat_feat_name1 = cat_feat_name
        cat_feat_name2 = cat_feat_name
    elif isinstance(cat_feat_name, list):
        # Use two feature names
        cat_feat_name1 = cat_feat_name[0]
        cat_feat_name2 = cat_feat_name[1]
    else:
        raise Exception(f'ERROR: The categorical feature name(s) must be given as either a string or a list of strings. Received input of type {type(cat_feat_name)} for the feature name(s).')
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data1_df = df1.copy()
        if df2 is not None:
            data2_df = df2.copy()
        else:
            # Just use dataframe `df1`
            data2_df = data1_df
    else:
        # Use the original dataframes
        data1_df = df1
        if df2 is not None:
            data2_df = df2
        else:
            data2_df = data1_df
    if dict1 is not None and dict2 is not None:
        data1_dict = dict1.copy()
        data2_dict = dict2.copy()
    else:
        # Determine each dataframe's dictionary of categories
        data1_df[cat_feat_name1], data1_dict = enum_categorical_feature(data1_df, cat_feat_name1, nan_value=nan_value)
        data2_df[cat_feat_name2], data2_dict = enum_categorical_feature(data2_df, cat_feat_name2, nan_value=nan_value)
    # Make sure that none of the categories' strings contain the separator symbol
    if separator != '_':
        in_category_symbol = '_'
    else:
        in_category_symbol = '-'
    data1_dict = utils.replace_dict_strings(data1_dict, str_to_replace=separator,
                                            new_str=in_category_symbol, replace_keys=True,
                                            replace_vals=False, inplace=True)
    data2_dict = utils.replace_dict_strings(data2_dict, str_to_replace=separator,
                                            new_str=in_category_symbol, replace_keys=True,
                                            replace_vals=False, inplace=True)
    # Invert the dictionaries of categories
    data1_dict_inv = utils.invert_dict(data1_dict)
    data2_dict_inv = utils.invert_dict(data2_dict)
    data1_dict_inv[nan_value] = 'nan'
    data2_dict_inv[nan_value] = 'nan'
    # Revert back to the original dictionaries, now without multiple NaN-like categories
    data1_dict = utils.invert_dict(data1_dict_inv)
    data2_dict = utils.invert_dict(data2_dict_inv)
    # Add the 'nan' key to the inverted dictionaries, as the dataframe migth
    # contain actual 'nan' values, besides the desired `nan_value` code
    data1_dict_inv['nan'] = 'nan'
    data2_dict_inv['nan'] = 'nan'
    # Get the unique categories of each dataframe
    data1_categories = list(data1_dict.keys())
    data2_categories = list(data2_dict.keys())
    # Combine all the unique categories into one single list
    all_categories = set(data1_categories + data2_categories)
    all_categories.remove('nan')
    if sort is True:
        all_categories = list(all_categories)
        all_categories.sort()
    # Create a new dictionary for the combined categories
    all_data_dict = create_enum_dict(all_categories, nan_value, forbidden_digit)
    all_data_dict['nan'] = nan_value
    # Revert the feature of each dataframe to its original categories strings
    print(f'Reverting column {cat_feat_name1} in dataframe 1 to the original string format...')
    data1_df[cat_feat_name1] = data1_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name1,
                                                                                  enum_dict=data1_dict_inv,
                                                                                  enum_to_category=True), axis=1)
    print(f'Reverting column {cat_feat_name2} in dataframe 2 to the original string format...')
    data2_df[cat_feat_name2] = data2_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name2,
                                                                                  enum_dict=data2_dict_inv,
                                                                                  enum_to_category=True), axis=1)
    # Convert the features' values into the new enumeration
    print(f'Converting column {cat_feat_name1} in dataframe 1 to the new encoding format...')
    data1_df[cat_feat_name1] = data1_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name1,
                                                                                  enum_dict=all_data_dict,
                                                                                  enum_to_category=False), axis=1)
    print(f'Converting column {cat_feat_name2} in dataframe 2 to the new encoding format...')
    data2_df[cat_feat_name2] = data2_df.apply(lambda df: enum_category_conversion(df, enum_column=cat_feat_name2,
                                                                                  enum_dict=all_data_dict,
                                                                                  enum_to_category=False), axis=1)
    print('Enumeration conversion done!')
    if df2 is None:
        return data1_df, all_data_dict
    else:
        return data1_df, data2_df, all_data_dict


def remove_nan_enum_from_string(x, nan_value='0'):
    '''Removes missing values (NaN) from enumeration encoded strings.

    Parameters
    ----------
    x : string
        Original string, with possible NaNs included.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.

    Returns
    -------
    x : string
        NaN removed string.
    '''
    # Make sure that the NaN value is represented as a string
    nan_value = str(nan_value)
    # Only remove NaN values if the string isn't just a single NaN value
    if x != nan_value:
        # Remove NaN value that might have a following encoded value
        if f'{nan_value};' in x:
            x = re.sub(f'{nan_value};', '', x)
        # Remove NaN value that might be at the end of the string
        if nan_value in x:
            x = re.sub(f';{nan_value}', '', x)
        # If the string got completly emptied, place a single NaN value on it
        if x == '':
            x = nan_value
    return x


def join_categ_list(df, separator='0', unique=True, nan_value=0, remove_listed_nan=True):
    '''Join categories encodings from a series (or groupby series) into a single
    value representation.
    e.g. [1, 2, 3, 4, 5, 6] -> 10203040506

    Parameters
    ----------
    df : pandas.Series or pandas.Series.GroupBy
        Pandas series which contains the categorical values to be joined.
    separator : string, default '0'
        Symbol that concatenates each string's words.
    unique : bool, default True
        If set to True, the method will only select unique categories, not
        accounting for repeated occurences.
    nan_value : int, default 0
        Integer number that represents NaN and NaN-like values.
    remove_listed_nan : bool, default True
        If set to True, joined rows where non-NaN values exist have the NaN
        values removed.

    Returns
    -------
    data_df : pandas.Series
        Resulting series from joining the categories encodings.
    '''
    if unique is True:
        # Get the set of unique categories in this group
        categ = set(df)
    else:
        categ = list(df)
    if len(categ) == 1:
        categ_val = categ[0]
        if str(categ_val).lower() == 'nan':
            return np.nan
        else:
            return int(categ_val)
    # Make sure that the categories values are numeric
    categ = [int(float(val)) if str(val).lower() != 'nan' else np.nan
             for val in categ]
    if remove_listed_nan is True and len(categ) > 1:
        # Remove the missing values encodings, if they exist
        try:
            categ.remove(nan_value)
        except (KeyError, ValueError):
            # There was no NaN represented value in the current group of values
            pass
        # Remove non encoded missing values, if they exist
        try:
            categ.remove(np.nan)
        except (KeyError, ValueError):
            # There was no NaN value in the current group of values
            pass
    # Get a single value representation of the categories list
    categ_val = separator.join([str(cat) for cat in categ])
    # if int(categ_val) < 2**64:
    #     # Return the value in a numeric format
    #     return int(categ_val)
    # else:
    #     # As the value can't be represented in a 64 bit numeric type, it must be in string format
    #     warnings.warn(f'Found a categories list value too big to be represented in a 64 bit numeric type: {categ_val}')
    #     return str(categ_val)
    if int(categ_val) >= 10**300:
        warnings.warn(f'Found a categories list value that goes beyond the maximum float representation!')
        return str(categ_val)
    elif int(categ_val) >= 2**64:
        warnings.warn(f'Found a categories list value that will be too big to be represented in a 64 bit numeric type: {categ_val}')
    elif int(categ_val) >= 10**14:
        warnings.warn(f'Found a categories list value that might be too big to be represented in a 64 bit numeric type: {categ_val}')
    # Return the value in a numeric format
    return int(categ_val)


def join_repeated_rows(df, cat_feat=[], id_columns=['patientunitstayid', 'ts'],
                       cont_join_method='mean', has_timestamp=None,
                       unique=True, nan_value=0, remove_listed_nan=True,
                       separator='0', inplace=False):
    '''Join rows that have the same identifier columns based on concatenating
    categorical encodings and on averaging continuous features.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which will be processed.
    cat_feat : string or list of strings, default []
        Name(s) of the categorical feature(s) which will have their values
        concatenated along the ID's.
    id_columns : list of strings, default ['patientunitstayid', 'ts']
        List of columns names which represent identifier columns. These are not
        supposed to be changed.
    cont_join_method : string, default 'mean'
        Defines which method to use when joining rows of continuous features.
        Can be either 'mean', 'min' or 'max'.
    has_timestamp : bool, default None
        If set to True, the resulting dataframe will be sorted and set as index
        by the timestamp column (`ts`). If not specified, the method will
        automatically look for a `ts` named column in the input dataframe.
    unique : bool, default True
        If set to True, the method will only select unique categories, not
        accounting for repeated occurences.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    remove_listed_nan : bool, default True
        If set to True, joined rows where non-NaN values exist have the NaN
        values removed.
    separator : string, default '0'
        Symbol that concatenates each string's words.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data_df : pandas.DataFrame or dask.DataFrame
        Resulting dataframe from merging all the concatenated or averaged
        features.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    # Define a list of dataframes
    df_list = []
    # See if there is a timestamp column on the dataframe (only considering as a
    # timestamp column those that are named 'ts')
    if has_timestamp is None:
        if 'ts' in data_df.columns:
            has_timestamp = True
        else:
            has_timestamp = False
    print('Concatenating categorical encodings...')
    if isinstance(cat_feat, str):
        # Make sure that the categorical feature names are in a list format,
        # even if it's just one feature name
        cat_feat = [cat_feat]
    for feature in utils.iterations_loop(cat_feat):
        # Convert to string format
        data_df[feature] = data_df[feature].astype(str)
        # Join with other categorical enumerations on the same ID's
        data_to_add = data_df.groupby(id_columns)[feature].apply(lambda x: join_categ_list(x, separator=separator,
                                                                                           unique=unique,
                                                                                           nan_value=nan_value,
                                                                                           remove_listed_nan=remove_listed_nan)).reset_index()
        if has_timestamp is True:
            # Sort by time `ts` and set it as index
            data_to_add = data_to_add.sort_values('ts')
        # Add to the list of dataframes that will be merged
        df_list.append(data_to_add)
    remaining_feat = list(set(data_df.columns) - set(cat_feat) - set(id_columns))
    if len(remaining_feat) > 0:
        print('Joining continuous features...')
        for feature in utils.iterations_loop(remaining_feat):
            if data_df[feature].dtype == 'object':
                raise Exception(f'ERROR: There is at least one non-numeric feature in the dataframe. This method requires all columns to be numeric, either integer or floats. In case there are categorical features still in string format, consider using the `string_encod_to_numeric` method first. The column {feature} is of type {df[feature].dtype}.')
        # Join remaining features through their average, min or max value
        # (just to be sure that there aren't missing or different values)
        if cont_join_method.lower() == 'mean':
            data_to_add = data_df.groupby(id_columns).mean().reset_index()
        elif cont_join_method.lower() == 'min':
            data_to_add = data_df.groupby(id_columns).min().reset_index()
        elif cont_join_method.lower() == 'max':
            data_to_add = data_df.groupby(id_columns).max().reset_index()
        if has_timestamp is True:
            # Sort by time `ts` and set it as index
            data_to_add = data_to_add.sort_values('ts')
        # Add to the list of dataframes that will be merged
        df_list.append(data_to_add)
    # Merge all dataframes
    print('Merging features\' dataframes...')
    if isinstance(df, dd.DataFrame):
        data_df = reduce(lambda x, y: dd.merge(x, y, on=id_columns), df_list)
    else:
        data_df = reduce(lambda x, y: pd.merge(x, y, on=id_columns), df_list)
    print('Done!')
    return data_df


def string_encod_to_numeric(df, cat_feat=None, separator='0', separator_num=0, inplace=False):
    '''Convert the string encoded columns that represent lists of categories,
    separated by semicolons, into numeric columns through the replacement of
    the semicolon character by a given number. This allows the dataframe to
    be adequately converted into a PyTorch or TensorFlow tensor.

    Parameters
    ----------
    df : pandas.DataFrame or dask.DataFrame
        Dataframe which will be processed.
    cat_feat : string or list of strings, default None
        Name(s) of the categorical encoded feature(s) which will have their
        semicolon separators converted into its binary ASCII code. If not
        specified, the method will look through all columns, processing
        the ones that might have semicolons.
    separator_num : int, default 0
        Number to use as a representation of the semicolon encoding
        separator.
    inplace : bool, default False
        If set to True, the original dataframe will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original dataframe.

    Returns
    -------
    data_df : pandas.DataFrame or dask.DataFrame
        Resulting dataframe from converting the string encoded columns into
        numeric ones, making it tensor ready.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_df = df.copy()
    else:
        # Use the original dataframe
        data_df = df
    if cat_feat is None:
        cat_feat = []
        # Go through all the features processing the ones that might have semicolons
        for feature in df.columns:
            # Only analyze the feature if it has string values
            if df[feature].dtype == 'object':
                cat_feat.append(feature)
    elif isinstance(cat_feat, str):
        # Make sure that the categorical feature names are in a list format,
        # even if it's just one feature name
        cat_feat = [cat_feat]
    if isinstance(cat_feat, list):
        for feature in cat_feat:
            # Make sure that all values are in string format
            data_df[feature] = data_df[feature].astype(str)
            # Replace semicolon characters by its binary ASCII code
            data_df[feature] = data_df[feature].str.replace(separator, str(separator_num))
            try:
                # Convert column to a numeric format
                data_df[feature] = data_df[feature].astype(float)
            except Exception as e:
                warnings.warn(f'There was a problem converting column {feature} to dtype float. Original exception message: "{str(e)}"')
    else:
        raise Exception(f'ERROR: When specified, the categorical features `cat_feat` must be in string or list of strings format, not {type(cat_feat)}.')
    return data_df


def prepare_embed_bag(data, feature=None, separator_num=0, padding_value=999999,
                      nan_value=0):
    '''Prepare a categorical feature for embedding bag, i.e. split category
    enumerations into separate numbers, combine them into a single list and set
    the appropriate offsets as to when each row's group of categories end.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor that contains the categorical feature that will be embedded.
    feature : int, default None
        Index of the categorical feature on which embedding bag will be
        applied. Can only be left undefined if the data is one dimensional.
    separator_num : int, default 0
        Number to use as a representation of the semicolon encoding
        separator.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.

    Returns
    -------
    enum_list : torch.Tensor
        List of all categorical enumerations, i.e. the numbers corresponding to
        each of the feature's categories, contained in the input series.
    offset_list : torch.Tensor
        List of when each row's categorical enumerations start, considering the
        enum_list list.
    '''
    enum_list = []
    count = 0
    offset_list = [count]
    if feature is None and len(data.shape) > 1:
        raise Exception('ERROR: If multidimensional data is passed in the input, the feature from which to get the full list of categorical encodings must be defined.')
    if len(data.shape) < 3:
        for i in range(data.shape[0]):
            # Get the full list of digits of the current value
            if len(data.shape) == 1:
                feature_val_i = utils.get_full_number_string(data[i].item())
            else:
                feature_val_i = utils.get_full_number_string(data[i, feature].item())
            if len(feature_val_i) > 1:
                # Separate digits in the same string
                digits_list = feature_val_i.split(str(separator_num))
            else:
                # Just use append the single encoding value
                digits_list = [feature_val_i]
            if int(digits_list[0]) == padding_value:
                # Add a missing value embedding
                enum_list.append([str(nan_value)])
            else:
                # Add the digits to the enumeration encodings list
                enum_list.append(digits_list)
            # Set the end of the current list
            count += len(digits_list)
            offset_list.append(count)
        # Flatten list
        enum_list = [int(value) for sublist in enum_list for value in sublist]
    elif len(data.shape) == 3:
        # Process each sequence separately, creating 2D categorical encodings and offset lists
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Get the full list of digits of the current value
                feature_val_i = utils.get_full_number_string(data[i, j, feature].item())
                if len(feature_val_i) > 1:
                    # Separate digits in the same string
                    digits_list = feature_val_i.split(str(separator_num))
                else:
                    # Just use append the single encoding value
                    digits_list = [feature_val_i]
                if int(digits_list[0]) == padding_value:
                    # Add a missing value embedding
                    enum_list.append([str(nan_value)])
                else:
                    # Add the digits to the enumeration encodings list
                    enum_list.append(digits_list)
                # Set the end of the current list
                count += len(digits_list)
                offset_list.append(count)
        # Flatten full list
        enum_list = [int(value) for sublist in enum_list for value in sublist]
    else:
        raise Exception(f'ERROR: Data with more than 3 dimensions is not supported. Input data has {len(data.shape)} dimensions.')
    # Convert to PyTorch tensor
    enum_list = torch.tensor(enum_list)
    offset_list = torch.tensor(offset_list)
    return enum_list, offset_list


def run_embed_bag(data, embedding_layer, enum_list, offset, inplace=False):
    '''Run an embedding bag layer on a list(s) of encoded categories, adding
    the new embedding columns to the data tensor.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor that contains the categorical feature that will be embedded.
    embedding_layer : torch.nn.EmbeddingBag
        PyTorch layer that applies the embedding bag, i.e. calculates the
        average embedding based on multiple encoded values.
    enum_list : torch.Tensor
        List of all categorical enumerations, i.e. the numbers corresponding to
        each of the feature's categories, contained in the input series.
    offset : torch.Tensor
        List of when each row's categorical enumerations start, considering the
        enum_list list.
    inplace : bool, default False
        If set to True, the original tensor will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original tensor.

    Returns
    -------
    data : torch.Tensor
        Data tensor with the new embedding features added.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original tensor
        data_tensor = data.clone()
    else:
        # Use the original dataframe
        data_tensor = data
    # Check if GPU is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu is True:
        # Move data and embedding model to GPU
        (data_tensor, enum_list,
         offset, embedding_layer) = (data_tensor.cuda(), enum_list.cuda(),
                                     offset.cuda(), embedding_layer.cuda())
    # Get a tensor with the embedding values retrieved from the embedding bag layer
    embed_data = embedding_layer(enum_list, offset)[:-1]
    if len(data_tensor.shape) == 1:
        # Add the new embedding columns to the data tensor
        data_tensor = torch.cat((data_tensor.double(), embed_data.double()))
    elif len(data_tensor.shape) == 2:
        # Add the new embedding columns to the data tensor
        data_tensor = torch.cat((data_tensor.double(), embed_data.double()), dim=1)
    elif len(data_tensor.shape) == 3:
        # Change shape of the embeddings tensor to match the original data
        embed_data = embed_data.view(data_tensor.shape[0], data_tensor.shape[1], embedding_layer.embedding_dim)
        # Add the new embedding columns to the data tensor
        data_tensor = torch.cat((data_tensor.double(), embed_data.double()), dim=2)
    else:
        raise Exception(f'ERROR: Data with more than 3 dimensions is not supported. Input data has {len(data_tensor.shape)} dimensions.')
    return data_tensor


def embedding_bag_pipeline(data, embedding_layer, features, model_forward=False,
                           n_id_cols=2, padding_value=999999, nan_value=0,
                           inplace=False):
    '''Run the complete pipeline that gets us from a data tensor with categorical
    features, i.e. columns with lists of encodings as values, into a data tensor
    with embedding columns.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor that contains the categorical feature(s) that will be embedded.
    embedding_layer : torch.nn.EmbeddingBag or torch.nn.ModuleList or torch.nn.ModuleDict
        PyTorch layer(s) that applies the embedding bag, i.e. calculates the
        average embedding based on multiple encoded values.
    features : int or list of int
        Index (or indeces) of the categorical column(s) that will be ran through
        its (or their) respective embedding layer(s). This feature(s) is (are)
        removed from the data tensor after the embedding columns are added.
    model_forward : bool, default False
        Indicates if the method is being executed inside a machine learning model's
        forward method. If so, it will account for a previous removal of sample
        identidying columns.
    n_id_cols : int, default 2
        Number of ID columns. 1 represents simple tabular data, while 2 represents
        multivariate sequential data (e.g. EHR time series data).
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    nan_value : int, default 0
        Integer number that gets assigned to NaN and NaN-like values.
    inplace : bool, default False
        If set to True, the original tensor will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original tensor.

    Returns
    -------
    data : torch.Tensor
        Data tensor with the new embedding features added and the old categorical
        features removed.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_tensor = data.clone()
    else:
        # Use the original dataframe
        data_tensor = data
    # Check if it's only a single categorical feature or more
    if isinstance(embedding_layer, torch.nn.EmbeddingBag) and isinstance(features, int):
        # Get the list of all the encodings and their offsets
        if model_forward:
            enum_list, offset_list = prepare_embed_bag(data_tensor, features-n_id_cols,
                                                       padding_value=padding_value, nan_value=nan_value)
        else:
            enum_list, offset_list = prepare_embed_bag(data_tensor, features,
                                                       padding_value=padding_value, nan_value=nan_value)
        # Run the embedding bag and add the embedding columns to the tensor
        data_tensor = run_embed_bag(data_tensor, embedding_layer, enum_list, offset_list, inplace)
    elif isinstance(embedding_layer, torch.nn.ModuleList) and isinstance(features, list):
        for i in range(len(features)):
            # Get the list of all the encodings and their offsets
            if model_forward:
                enum_list, offset_list = prepare_embed_bag(data_tensor, features[i]-n_id_cols,
                                                           padding_value=padding_value, nan_value=nan_value)
            else:
                enum_list, offset_list = prepare_embed_bag(data_tensor, features[i],
                                                           padding_value=padding_value, nan_value=nan_value)
            # Run the embedding bag and add the embedding columns to the tensor
            data_tensor = run_embed_bag(data_tensor, embedding_layer[f'embed_{i}'], enum_list, offset_list, inplace)
    elif isinstance(embedding_layer, torch.nn.ModuleDict) and isinstance(features, list):
        for feature in features:
            # Get the list of all the encodings and their offsets
            if model_forward:
                enum_list, offset_list = prepare_embed_bag(data_tensor, feature-n_id_cols,
                                                           padding_value=padding_value, nan_value=nan_value)
            else:
                enum_list, offset_list = prepare_embed_bag(data_tensor, feature,
                                                           padding_value=padding_value, nan_value=nan_value)
            # Run the embedding bag and add the embedding columns to the tensor
            data_tensor = run_embed_bag(data_tensor, embedding_layer[f'embed_{feature}'], enum_list, offset_list, inplace)
    else:
        raise Exception(f'ERROR: The user must either a single embedding bag and feature index or lists of embedding bag layers and feature indeces. The input `embedding_layer` has type {type(embedding_layer)} while `feature` has type {type(features)}.')
    # Remove the old categorical feature(s)
    data_tensor = deep_learning.remove_tensor_column(data_tensor, features, inplace=inplace)
    return data_tensor


# [TODO] Create a function that takes a set of embeddings (which will be used in
# an embedding bag) and reverts them back to the original text
# [TODO] Define an automatic method to discover which embedded category was more
# important by doing inference on individual embeddings of each category separately,
# seeing which one caused a bigger change in the output.
