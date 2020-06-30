import comet_ml                                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
import yaml                                             # Save and load YAML files
from . import search_explore                            # Methods to search and explore data
from . import deep_learning                             # Common and generic deep learning related methods
from . import padding                                   # Padding and variable sequence length related methods
from . import datasets                                  # PyTorch dataset classes

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def one_hot_label(labels, n_outputs=None, dataset=None):
    # Create an all zeroes tensor with the required shape (i.e. #samples x #outputs)
    if n_outputs is not None:
        ohe_labels = torch.zeros(labels.shape[0], n_outputs)
    elif dataset is not None:
        ohe_labels = torch.zeros(labels.shape[0], int(dataset.y.max())+1)
    else:
        raise Exception('ERROR: Either `n_outputs` or `dataset` must be provided. All of them were left as None.')
    # Place ones in the columns that represent each activated output
    for i in range(ohe_labels.shape[0]):
        ohe_labels[i, int(labels[i])] = 1
    return ohe_labels


def create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1, batch_size=32,
                      get_indices=True, shuffle_dataset=True, num_workers=0,
                      train_indices=None, val_indices=None, test_indices=None):
    '''Distributes the data into train, validation and test sets and returns the
    respective data loaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset object which will be used to train, validate and test the model.
    test_train_ratio : float, default 0.2
        Number from 0 to 1 which indicates the percentage of the data
        which will be used as a test set. The remaining percentage
        is used in the training and validation sets.
    validation_ratio : float, default 0.1
        Number from 0 to 1 which indicates the percentage of the data
        from the training set which is used for validation purposes.
        A value of 0.0 corresponds to not using validation.
    train_indices : list of integers, default None
        Indices of the data which will be used during training.
    val_indices : list of integers, default None
        Indices of the data which will be used to evaluate the
        model's performance on a validation set during training.
    test_indices : list of integers, default None
        Indices of the data which will be used to evaluate the
        model's performance on a test set, after finishing the
        training process.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indices : bool, default True
        If set to True, the function returns the dataloader objects of
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    shuffle_dataset : bool, default True
        If set to True, the data of each set is shuffled.
    num_workers : int, default 0
        How many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process. Therefore, data loading may block
        computing. On the other hand, with `num_workers` > 0 we can get multiple
        workers loading the data in the background while the GPU is busy training,
        which might hide the loading time.

    Returns
    -------
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for getting batches of data which will be used
        during training.
    val_dataloader : torch.utils.data.DataLoader
        Dataloader for getting batches of data which will be used to
        evaluate the model's performance on a validation set during
        training.
    test_dataloader : torch.utils.data.DataLoader
        Dataloader for getting batches of data which will be used to
        evaluate the model's performance on a test set, after
        finishing the training process.

    If get_indices is True:

    train_indices : list of integers
        Indices of the data which will be used during training.
    val_indices : list of integers
        Indices of the data which will be used to evaluate the
        model's performance on a validation set during training.
    test_indices : list of integers
        Indices of the data which will be used to evaluate the
        model's performance on a test set, after finishing the
        training process.
    '''
    if (train_indices is None
    or val_indices is None
    or test_indices is None):
        # Create data indices for training and test splits
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(test_train_ratio * dataset_size))
        if shuffle_dataset is True:
            np.random.shuffle(indices)
        train_indices, test_indices = indices[test_split:], indices[:test_split]
        # Create data indices for training and validation splits
        train_dataset_size = len(train_indices)
        val_split = int(np.floor(validation_ratio * train_dataset_size))
        if shuffle_dataset is True:
            np.random.shuffle(train_indices)
        train_indices, val_indices = train_indices[val_split:], train_indices[:val_split]
    # Create data samplers that randomly sample from the respective indices on each run
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=val_sampler,
                                                 num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  sampler=test_sampler,
                                                  num_workers=num_workers)
    if get_indices is True:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader, test_dataloader


# [TODO] Create a generic train method that can train any relevant machine learning model on the input data
def train(model, train_dataloader, val_dataloader, test_dataloader=None,
          cols_to_remove=[0, 1], model_type='multivariate_rnn',
          seq_len_dict=None, batch_size=32, n_epochs=50, lr=0.001,
          models_path='models/', ModelClass=None, padding_value=999999,
          do_test=True, log_comet_ml=False, comet_ml_api_key=None,
          comet_ml_project_name=None, comet_ml_workspace=None,
          comet_ml_save_model=False, experiment=None, features_list=None,
          get_val_loss_min=False, **kwargs):
    model = deep_learning.train(model, train_dataloader, val_dataloader, test_dataloader=test_dataloader,
                                cols_to_remove=cols_to_remove, model_type=model_type,
                                seq_len_dict=seq_len_dict, batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                models_path=models_path, ModelClass=ModelClass, padding_value=padding_value,
                                do_test=do_test, log_comet_ml=log_comet_ml, comet_ml_api_key=comet_ml_api_key,
                                comet_ml_project_name=comet_ml_project_name, comet_ml_workspace=comet_ml_workspace,
                                comet_ml_save_model=comet_ml_save_model, experiment=experiment,
                                features_list=features_list, get_val_loss_min=get_val_loss_min, **kwargs)
    if get_val_loss_min is True:
        # Also return the minimum validation loss alongside the corresponding model
        return model[0], model[1]
    else:
        return model


def optimize_hyperparameters(Model, config_name, comet_ml_api_key,
                             comet_ml_project_name, comet_ml_workspace, df=None,
                             dataset=None, train_dataloader=None,
                             val_dataloader=None, test_dataloader=None,
                             n_inputs=None, id_column=None, label_column=None,
                             inst_column=None, id_columns_idx=None, n_outputs=1,
                             Dataset=None, model_type='multivariate_rnn',
                             is_custom=False, models_path='models/',
                             model_name='checkpoint', array_param=None,
                             metrics=['loss', 'accuracy', 'AUC'],
                             config_path='', var_seq=True, clip_value=0.5,
                             padding_value=999999, batch_size=32,
                             n_epochs=10, lr=0.001, test_train_ratio=0.2,
                             validation_ratio=0.1, comet_ml_save_model=True,
                             already_embedded=False, verbose=False,
                             see_progress=True, **kwargs):
    '''Optimize a machine learning model's hyperparameters, by training it
    several times while exploring different hyperparameters values, returning
    the best performing ones.

    Parameters
    ----------
    Model : torch.nn.Module or sklearn.* (any machine learning model)
        Class constructor for the desired machine learning model.
    config_name : str
        Name of the configuration file, containing information about the
        parameters to optimize. This data is organized in a YAML format, akin to
        a dictionary object, where the optimization algorithm is set, each
        hyperparameter gets a key with its name, followed by a list of values in
        the order of (minimum value to explore in the optimization, maximum
        value to explore in the optimization, initial value to use), and the
        metric to be optimized.
    comet_ml_api_key : string
        Comet.ml API key used when logging data to the platform.
    comet_ml_project_name : string
        Name of the comet.ml project used when logging data to the platform.
    comet_ml_workspace : string
        Name of the comet.ml workspace used when logging data to the platform.
    df : pandas.DataFrame or dask.DataFrame, default None
        Dataframe containing all the data that will be used in the
        optimization's training processes.
    train_dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during training. If
        not specified, the method will create one automatically.
    val_dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches when evaluating
        the model's performance on a validation set during training. If not
        specified, the method will create one automatically.
    test_dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches whe evaluating
        the model's performance on a test set, after finishing the
        training process If not specified, the method will create one
        automatically.
    dataset : torch.utils.data.Dataset, default None
        Dataset object that contains the data used to train, validate and test
        the machine learning models. Having the dataloaders set, this argument
        is only needed if the data has variable sequence length and its dataset
        object loads files in each batch, instead of data from a single file.
        In essence, it's needed to give us the current batch's sequence length
        information, when we couldn't have known this for the whole data
        beforehand. If not specified, the method will create one automatically.
    n_inputs : int, default None
        Total number of input features present in the dataframe.
    id_column : str, default None
        Name of the column which corresponds to the subject identifier.
    label_column : str, default None
        Name of the column which corresponds to the label.
    inst_column : str, default None
        Name of the column which corresponds to the instance or timestamp
        identifier.
    id_columns_idx : int or list of ints, default None
        Index or list of indices of columns to remove from the features before
        feeding to the model. This tend to be the identifier columns, such as
        `subject_id` and `ts` (timestamp).
    n_outputs : int, default 1
        Total number of outputs givenm by the machine learning model.
    Dataset : torch.torch.utils.data.Dataset, default None
        Class constructor for the dataset, which will be used for iterating
        through batches of data. It must be able to receive as inputs a PyTorch
        tensor and a Pandas or Dask dataframe.
    model_type : string, default 'multivariate_rnn'
        Sets the type of model to train. Important to know what type of
        inference to do. Currently available options are ['multivariate_rnn',
        'mlp'].
    is_custom : bool, default False
        If set to True, the method will assume that the model being used is a
        custom built one, which won't require sequence length information during
        the feedforward process.
    models_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    model_name : string, default 'checkpoint'
        Name that will be given to the saved models. Validation loss and
        timestamp info will then be appended to the name.
    array_param : list of strings, default None
        List of feature names that might have multiple values associated to
        them. For example, in a neural network with multiple layers, there
        could be multiple `n_hidden` values, each one indicating the number
        of units in each hidden layer.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (`loss`), accuracy (`accuracy`),
        AUC (`AUC`), weighted AUC (`AUC_weighted`), precision (`precision`),
        recall (`recall`) and F1 (`F1`).
    config_path : str, default ''
        Path to the directory where the configuration file is stored.
    var_seq : bool, default True
        Specifies if the data has variable sequence length. Valuable information
        if the data must be adjusted by padding.
    clip_value : int or float, default 0.5
        Gradient clipping value, which limit the maximum change in the
        model parameters, so as to avoid exploiding gradients.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : int, default 10
        Number of epochs, i.e. the number of times the training loop
        iterates through all of the training data.
    lr : float, default 0.001
        Learning rate used in the optimization algorithm.
    test_train_ratio : float, default 0.2
        Percentage of data to use for the test set.
    validation_ratio : float, default 0.1
        Percentage of training data to use for the validation set.
    comet_ml_save_model : bool, default True
        If set to True, uploads the model with the lowest validation loss
        to comet.ml when logging data to the platform.
    already_embedded : bool, default False
        If set to True, it means that the categorical features are already
        embedded when fetching a batch, i.e. there's no need to run the embedding
        layer(s) during the model's feedforward.
    verbose : bool, default False
        If set to True, a set of metrics and status indicators will be printed
        throughout training.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of each loop.
    kwargs : dict
        Optional additional parameters, specific to the machine learning model
        being used.

    Returns
    -------
    val_loss_min : float
        Minimum validation loss over all the optimization process.
    exp_name_min : str
        Name of the comet ml experiment with the overall minimum validation
        loss.

    [TODO] Write a small tutorial on how to write the YAML configuration file,
    based on this: https://www.comet.ml/docs/python-sdk/introduction-optimizer/
    '''
    # Only log training info to Comet.ml if the required parameters are specified
    if not (comet_ml_api_key is not None
            and comet_ml_project_name is not None
            and comet_ml_workspace is not None):
        raise Exception('ERROR: All necessary Comet.ml parameters (comet_ml_api_key, comet_ml_project_name, comet_ml_workspace) must be correctly specified. Otherwise, the parameter optimization won\'t work.')
    # Load the hyperparameter optimization configuration file into a dictionary
    config_file = open(f'{config_path}{config_name}', 'r')
    config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    # Get all the names of the hyperparameters that will be optimized
    params_names = list(config_dict['parameters'].keys())
    if array_param is not None:
        if isinstance(array_param, str):
            # Make sure that the array parameter names are in a list format
            array_param = [array_param]
        # Create a dictionary of lists, attributing all subparameter
        # names that belong to each array parameter
        array_subparam = dict()
        for param in array_param:
            # Add all the names of subparameters that start with the same parameter name
            array_subparam[param] = [subparam for subparam in params_names
                                     if subparam.startswith(param)]
    # Create a Comet.ml parameter optimizer
    param_optimizer = comet_ml.Optimizer(config_dict,
                                         api_key=comet_ml_api_key,
                                         project_name=comet_ml_project_name,
                                         workspace=comet_ml_workspace)

    seq_len_dict = None
    if df is not None:
        if inst_column is not None and var_seq is True:
            print('Building a dictionary containing the sequence length of each patient\'s time series...')
            # Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
            seq_len_dict = padding.get_sequence_length_dict(df, id_column=id_column, ts_column=inst_column)
            print('Creating a padded tensor version of the dataframe...')
            # Pad data (to have fixed sequence length) and convert into a PyTorch tensor
            data = padding.dataframe_to_padded_tensor(df, seq_len_dict=seq_len_dict,
                                                      id_column=id_column,
                                                      ts_column=inst_column,
                                                      padding_value=padding_value,
                                                      inplace=True)
        else:
            # Just convert the data into a PyTorch tensor
            data = torch.from_numpy(df.to_numpy())
        if id_columns_idx is None:
            # Find the column indices for the ID columns
            id_columns_idx = [search_explore.find_col_idx(df, col) for col in [id_column, inst_column]]

    if dataset is None:
        print('Creating a dataset object...')
        # Create a Dataset object from the data tensor
        if Dataset is not None:
            dataset = Dataset(data, df)
        else:
            if model_type.lower() == 'multivariate_rnn':
                dataset = datasets.Time_Series_Dataset(df, data, id_column=id_column,
                                                       ts_column=inst_column, seq_len_dict=seq_len_dict)
            elif model_type.lower() == 'mlp':
                dataset = datasets.Tabular_Dataset(data, df)
            else:
                raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {model_type}.')
    if train_dataloader is None and val_dataloader is None and test_dataloader is None:
        print('Distributing the data to train, validation and test sets and getting their data loaders...')
        # Get the train, validation and test sets data loaders, which will allow loading batches
        train_dataloader, val_dataloader, test_dataloader = create_train_sets(dataset, test_train_ratio=test_train_ratio,
                                                                              validation_ratio=validation_ratio,
                                                                              batch_size=batch_size, get_indices=False)
    # Start off with a minimum validation score of infinity
    val_loss_min = np.inf

    for experiment in param_optimizer.get_experiments():
        print('Starting a new parameter optimization iteration...')
        # Get the current optimized values of the hyperparameters
        params_values = dict(zip(params_names, [experiment.get_parameter(param)
                                                for param in params_names]))
        if array_param is not None:
            for param in array_param:
                # Join the values of the subparameters
                subparam_names = array_subparam[param]
                params_values[param] = [params_values[subparam] for subparam in subparam_names]
                # Remove the now redundant subparameters
                for subparam in subparam_names:
                    del params_values[subparam]
        # Instantiate the model (removing the two identifier columns and the labels from the input size)
        model = Model(n_inputs=n_inputs, n_outputs=n_outputs, **params_values, **kwargs)
        # Check if GPU (CUDA) is available
        on_gpu = torch.cuda.is_available()
        if on_gpu:
            # Move the model to the GPU
            model = model.cuda()
        print('Training the model...')
        # Train the model and get the minimum validation loss
        model, val_loss = deep_learning.train(model, train_dataloader, val_dataloader,
                                              test_dataloader=test_dataloader,
                                              dataset=dataset,
                                              cols_to_remove=id_columns_idx,
                                              model_type=model_type,
                                              is_custom=is_custom,
                                              seq_len_dict=seq_len_dict,
                                              batch_size=batch_size, n_epochs=n_epochs,
                                              lr=lr, clip_value=clip_value,
                                              models_path=models_path,
                                              model_name=model_name,
                                              ModelClass=Model,
                                              padding_value=padding_value,
                                              do_test=True, metrics=metrics,
                                              log_comet_ml=True,
                                              comet_ml_api_key=comet_ml_api_key,
                                              comet_ml_project_name=comet_ml_project_name,
                                              comet_ml_workspace=comet_ml_workspace,
                                              comet_ml_save_model=comet_ml_save_model,
                                              experiment=experiment, features_list=None,
                                              get_val_loss_min=True,
                                              already_embedded=already_embedded,
                                              verbose=verbose,
                                              see_progress=see_progress)
        if val_loss < val_loss_min:
            # Update optimization minimum validation loss and the corresponding
            # experiment name
            val_loss_min = val_loss
            exp_name_min = experiment.get_key()
            if verbose is True:
                print(f'Achieved a new minimum validation loss of {val_loss_min} on experiment {exp_name_min}')
        # Log optimization parameters
        experiment.log_parameter('n_inputs', n_inputs)
        experiment.log_parameter('n_outputs', n_outputs)
        experiment.log_parameter('clip_value', clip_value)
        experiment.log_parameter('padding_value', padding_value)
        experiment.log_parameter('batch_size', batch_size)
        experiment.log_parameter('n_epochs', n_epochs)
        experiment.log_parameter('lr', lr)
        experiment.log_parameter('test_train_ratio', test_train_ratio)
        experiment.log_parameter('validation_ratio', validation_ratio)
        experiment.log_asset(f'{config_path}{config_name}', config_name)
        experiment.log_other('param_optimizer_status', param_optimizer.status())
    if verbose is True:
        print(f'Finished the hyperparameter optimization! The best performing experiment was {exp_name_min}, with a minimum validation loss of {val_loss_min}')
    return val_loss_min, exp_name_min


# [TODO] Create a generic inference method that can run inference with any relevant machine learning model on the input data
