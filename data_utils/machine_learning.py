import comet_ml                                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
import yaml                                             # Save and load YAML files
from . import deep_learning                             # Common and generic deep learning related methods
from . import padding                                   # Padding and variable sequence length related methods

# Ignore Dask's 'meta' warning
warnings.filterwarnings("ignore", message="`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.")

# Methods

def create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1, batch_size=32,
                      get_indeces=True, shuffle_dataset=True):
    '''Distributes the data into train, validation and test sets and returns the respective data loaders.

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
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    get_indeces : bool, default True
        If set to True, the function returns the dataloader objects of
        the train, validation and test sets and also the indices of the
        sets' data. Otherwise, it only returns the data loaders.
    shuffle_dataset : bool, default True
        If set to True, the data of each set is shuffled.

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

    If get_indeces is True:

    train_indices : torch.utils.data.DataLoader
        Indices of the data which will be used during training.
    val_indices : torch.utils.data.DataLoader
        Indices of the data which will be used to evaluate the 
        model's performance on a validation set during training.
    test_indices : torch.utils.data.DataLoader
        Indices of the data which will be used to evaluate the 
        model's performance on a test set, after finishing the 
        training process.
    '''
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

    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create dataloaders for each set, which will allow loading batches
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    if get_indeces is True:
        # Return the data loaders and the indices of the sets
        return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices
    else:
        # Just return the data loaders of each set
        return train_dataloader, val_dataloader, test_dataloader


# [TODO] Create a generic train method that can train any relevant machine learning model on the input data
def train(model, train_dataloader, val_dataloader, seq_len_dict, test_dataloader=None,
          batch_size=32, n_epochs=50, lr=0.001, model_path='models/',
          do_test=True, log_comet_ml=False, comet_ml_api_key=None,
          comet_ml_project_name=None, comet_ml_workspace=None,
          comet_ml_save_model=False, experiment=None, features_list=None,
          get_val_loss_min=False, **kwargs):
    model = deep_learning.train(model, train_dataloader, val_dataloader, seq_len_dict=seq_len_dict, 
                                test_dataloader=test_dataloader, batch_size=batch_size, n_epochs=n_epochs, 
                                lr=lr, model_path=model_path, do_test=do_test, log_comet_ml=log_comet_ml, 
                                comet_ml_api_key=comet_ml_api_key, comet_ml_project_name=comet_ml_project_name,
                                comet_ml_workspace=comet_ml_workspace, comet_ml_save_model=comet_ml_save_model,
                                experiment=experiment, features_list=features_list, 
                                get_val_loss_min=get_val_loss_min, **kwargs)
    return model


def optimize_hyperparameters(Model, Dataset, df, config_name, comet_ml_api_key,
                             comet_ml_project_name, comet_ml_workspace, n_inputs,
                             id_column, label_column, inst_column=None,
                             n_outputs=1, config_path='', var_seq=True,
                             clip_value=10, padding_value=999999, batch_size=32,
                             n_epochs=20, lr=0.001, test_train_ratio=0.2,
                             validation_ratio=0.1, comet_ml_save_model=True,
                             **kwargs):
    '''Optimize a machine learning model's hyperparameters, by training it
    several times while exploring different hyperparameters values, returning
    the best performing ones.

    Parameters
    ----------
    Model : torch.nn.Module or sklearn.* (any machine learning model)
        Class constructor for the desired machine learning model.
    Dataset : torch.torch.utils.data.Dataset
        Class constructor for the dataset, which will be used for iterating
        through batches of data. It must be able to receive as inputs a PyTorch
        tensor and a Pandas or Dask dataframe.
    df : pandas.DataFrame or dask.DataFrame
        Dataframe containing all the data that will be used in the
        optimization's training processes.
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
    n_inputs : int
        Total number of input features present in the dataframe.
    id_column : str
        Name of the column which corresponds to the subject identifier.
    label_column : str
        Name of the column which corresponds to the label.
    inst_column : str, default None
        Name of the column which corresponds to the instance or timestamp
        identifier.
    n_outputs : int, default 1
        Total number of outputs givenm by the machine learning model.
    config_path : str, default ''
        Path to the directory where the configuration file is stored.
    var_seq : bool, default True
        Specifies if the data has variable sequence length. Valuable information
        if the data must be adjusted by padding.
    clip_value : numeric, default 10
        Gradient clipping threshold to avoid exploding gradients.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : int, default 50
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
        raise Exception('ERROR: All necessary Comet.ml parameters \
                         (comet_ml_api_key, comet_ml_project_name, \
                         comet_ml_workspace) must be correctly specified. \
                         Otherwise, the parameter optimization won\'t work.')
    # Load the hyperparameter optimization configuration file into a dictionary
    config_file = open(f'{config_path}config_name', 'r')
    config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    # Get all the names of the hyperparameters that will be optimized
    params_names = list(config_dict['parameters'].keys())
    # Create a Comet.ml parameter optimizer:
    param_optimizer = comet_ml.Optimizer(config_file,
                                         api_key=comet_ml_api_key,
                                         project_name=comet_ml_project_name,
                                         workspace=comet_ml_workspace)

    if inst_column is not None and var_seq is True:
        print('Building a dictionary containing the sequence length of each patient\'s time series...')
        # Dictionary containing the sequence length (number of temporal events) of each sequence (patient)
        seq_len_df = df.groupby(id_column)[inst_column].count().to_frame().sort_values(by=inst_column, ascending=False)
        seq_len_dict = dict([(idx, val[0]) for idx, val in list(zip(seq_len_df.index, seq_len_df.values))])
        print('Creating a padded tensor version of the dataframe...')
        # Pad data (to have fixed sequence length) and convert into a PyTorch tensor
        data = padding.dataframe_to_padded_tensor(df, seq_len_dict, n_patients, n_inputs, padding_value=padding_value)
    else:
        # Just convert the data into a PyTorch tensor
        data = torch.from_numpy(df.to_numpy())

    print('Creating a dataset object...')
    # Create a Dataset object from the data tensor
    dataset = Dataset(data, df)
    print('Distributing the data to train, validation and test sets and getting their data loaders...')
    # Get the train, validation and test sets data loaders, which will allow loading batches
    train_dataloader, val_dataloader, test_dataloader = create_train_sets(dataset, test_train_ratio=0.2, validation_ratio=0.1,
                                                                          batch_size=batch_size, get_indeces=False)
    # Start off with a minimum validation score of infinity
    val_loss_min = np.inf

    for experiment in param_optimizer.get_experiments():
        print('Starting a new parameter optimization iteration...')
        # Get the current optimized values of the hyperparameters
        params_values = dict(zip(params_names, [param_optimizer.get_parameter(param)
                                                for param in params_names]))
        # Instantiate the model (removing the two identifier columns and the labels from the input size)
        model = Model(n_inputs, n_outputs, **params_values, **kwargs)
        # Check if GPU (CUDA) is available
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            # Move the model to the GPU
            model = model.cuda()
        # Set gradient clipping to avoid exploding gradients
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        print('Training the model...')
        # Train the model and get the minimum validation loss
        model, val_loss = train(model, train_dataloader, val_dataloader,
                                test_dataloader, seq_len_dict, batch_size,
                                n_epochs, lr, model_path='models/',
                                padding_value=padding_value, do_test=True,
                                log_comet_ml=True,
                                comet_ml_save_model=comet_ml_save_model,
                                experiment=experiment,
                                features_list=list(df.columns).remove(label_column),
                                get_val_loss_min=True)
        if val_loss < val_loss_min:
            # Update optimization minimum validation loss and the corresponding
            # experiment name 
            val_loss_min = val_loss
            exp_name_min = experiment.get_key()
        # Log optimization parameters
        experiment.log_metric('n_inputs', n_inputs)
        experiment.log_metric('n_outputs', n_outputs)
        experiment.log_metric('clip_value', clip_value)
        experiment.log_metric('padding_value', padding_value)
        experiment.log_metric('batch_size', batch_size)
        experiment.log_metric('n_epochs', n_epochs)
        experiment.log_metric('lr', lr)
        experiment.log_metric('test_train_ratio', test_train_ratio)
        experiment.log_metric('validation_ratio', validation_ratio)
        experiment.log_asset(f'{config_path}config_name', config_name)
        experiment.log_asset(param_optimizer.status(), 'param_optimizer_status')
    return val_loss_min, exp_name_min


# [TODO] Create a generic inference method that can run inference with any relevant machine learning model on the input data
