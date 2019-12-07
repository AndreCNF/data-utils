from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
from torch.nn.functional import softmax                 # Softmax activation function to normalize scores
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
from datetime import datetime                           # datetime to use proper date and time formats
import sys                                              # Identify types of exceptions
import inspect                                          # Inspect methods and their arguments
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
from . import utils                                     # Generic and useful methods
from . import padding                                   # Padding and variable sequence length related methods
from . import machine_learning
import data_utils as du

# Ignore Dask's 'meta' warning
warnings.filterwarnings('ignore', message='`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.')

# Methods

def remove_tensor_column(data, col_idx, inplace=False):
    '''Remove a column(s) from a PyTorch tensor.

    Parameters
    ----------
    data : torch.Tensor
        Data tensor that contains the column(s) that will be removed.
    col_idx : int or list of int
        Index (or indices) or the column(s) to remove.
    inplace : bool, default False
        If set to True, the original tensor will be used and modified
        directly. Otherwise, a copy will be created and returned, without
        changing the original tensor.

    Returns
    -------
    data : torch.Tensor
        Data tensor with the undesired column(s) removed.
    '''
    if not inplace:
        # Make a copy of the data to avoid potentially unwanted changes to the original dataframe
        data_tensor = data.clone()
    else:
        # Use the original dataframe
        data_tensor = data
    if isinstance(col_idx, int):
        # Turn the column index into a list, for ease of coding
        col_idx = [col_idx]
    if not isinstance(col_idx, list):
        raise Exception(f'ERROR: The `col_idx` parameter must either specify a single int of a column to remove or a list of ints in the case of multiple columns to remove. Received input `col_idx` of type {type(col_idx)}.')
    for col in col_idx:
        # Make a list of the indices of the columns that we want to keep, 
        # without the unwanted one
        columns_to_keep = list(range(col)) + list(range(col + 1, data_tensor.shape[-1]))
        # Remove the current column
        if len(data_tensor.shape) == 2:
            data_tensor = data_tensor[:, columns_to_keep]
        elif len(data_tensor.shape) == 3:
            data_tensor = data_tensor[:, :, columns_to_keep]
        else:
            raise Exception(f'ERROR: Currently only supporting either 2D or 3D data. Received data tensor with {len(data_tensor.shape)} dimensions.')
    return data_tensor


def load_checkpoint(filepath, Model, *args):
    '''Load a model from a specified path and name.

    Parameters
    ----------
    filepath : str
        Path to the model being loaded, including it's own file name.
    Model : torch.nn.Module (any deep learning model)
        Class constructor for the desired deep learning model.
    args : list of str
        Names of the neural network's parameters that need to be
        loaded.

    Returns
    -------
    model : nn.Module
        The loaded model with saved weight values.
    '''
    # Load the saved data
    checkpoint = torch.load(filepath)
    # Retrieve the parameters' values and integrate them in
    # a dictionary
    params = dict(zip(args, [checkpoint[param] for param in args]))
    # Create a model with the saved parameters
    model = Model(params)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def change_grad(grad, data, min=0, max=1):
    '''Restrict the gradients to only have valid values.

    Parameters
    ----------
    grad : torch.Tensor
        PyTorch tensor containing the gradients of the data being optimized.
    data : torch.Tensor
        PyTorch tensor containing the data being optimized.
    min : int, default 0
        Minimum valid data value.
    max : int, default 0
        Maximum valid data value.

    Returns
    -------
    grad : torch.Tensor
        PyTorch tensor containing the corrected gradients of the data being
        optimized.
    '''
    # Minimum accepted gradient value to be considered
    min_grad_val = 0.001

    for i in range(data.shape[0]):
        if (data[i] == min and grad[i] < 0) or (data[i] == max and grad[i] > 0):
            # Stop the gradient from excedding the limit
            grad[i] = 0
        elif data[i] == min and grad[i] > min_grad_val:
            # Make the gradient have a integer value
            grad[i] = 1
        elif data[i] == max and grad[i] < -min_grad_val:
            # Make the gradient have a integer value
            grad[i] = -1
        else:
            # Avoid any insignificant gradient
            grad[i] = 0

    return grad


def ts_tensor_to_np_matrix(data, feat_num=None, padding_value=999999):
    '''Convert a 3D PyTorch tensor, such as one representing multiple time series
    data, into a 2D NumPy matrix. Can be useful for applying the SHAP Kernel
    Explainer.

    Parameters
    ----------
    data : torch.Tensor
        PyTorch tensor containing the three dimensional data being converted.
    feat_num : list of int, default None
        List of the column numbers that represent the features. If not specified,
        all columns will be used.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.

    Returns
    -------
    data_matrix : numpy.ndarray
        NumPy two dimensional matrix obtained from the data after conversion.
    '''
    # View as a single sequence, i.e. like a dataframe without grouping by id
    data_matrix = data.contiguous().view(-1, data.shape[2]).detach().numpy()
    # Remove rows that are filled with padding values
    if feat_num is not None:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix[:, feat_num]]]
    else:
        data_matrix = data_matrix[[not all(row == padding_value) for row in data_matrix]]
    return data_matrix


# [TODO] Create methods that contain the essential code inside a training iteration,
# for each model type (e.g. RNN, MLP, etc)
def inference_iter_multi_var_rnn(model, features, labels, cols_to_remove=[0, 1],
                                 is_train=False, optimizer=None):
    '''Run a single inference or training iteration on a Recurrent Neural Network (RNN),
    applied to multivariate data, such as EHR. Performance metrics still need to be
    calculated after executing this method.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a
        classification task.
    features : torch.Tensor
        Data tensor that contains the features on which to run the model.
    labels : torch.Tensor
        Tensor that contains the labels for each row.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp).
    is_train : bool, default True
        Indicates if the method is being called in a training loop. If
        set to True, the network's weights will be updated by the
        given optimizer.
    optimizer : torch.optim.Optimizer
        Optimization algorthim, responsible for updating the model's
        weights in order to minimize (or maximize) the intended goal.

    Returns
    -------
    correct_pred : torch.Tensor
        Binary data tensor with the prediction results, indicating 1
        if the prediction is correct and 0 otherwise.
    unpadded_scores : torch.Tensor
        Data tensor containing the output scores resulting from the
        inference. Without paddings.
    unpadded_labels : torch.Tensor
        Tensor containing the labels for each row. Without paddings.
    loss : torch.nn.modules.loss
        Obtained loss value. Although the optimization step can be
        done inside this method, the loss value could be useful as
        a metric.
    '''
    # Make the data have type float instead of double, as it would cause problems
    features, labels = features.float(), labels.float()
    # Sort the data by sequence length
    features, labels, x_lengths = padding.sort_by_seq_len(features, seq_len_dict, labels)
    # Remove unwanted columns from the data
    features = remove_tensor_column(features, cols_to_remove, inplace=True)
    # Feedforward the data through the model
    scores = model.forward(features, x_lengths)
    # Adjust the labels so that it gets the exact same shape as the predictions
    # (i.e. sequence length = max sequence length of the current batch, not the max of all the data)
    labels = torch.nn.utils.rnn.pack_padded_sequence(labels, x_lengths, batch_first=True)
    labels, _ = torch.nn.utils.rnn.pad_packed_sequence(labels, batch_first=True, padding_value=padding_value)
    # Calculate the cross entropy loss
    loss = model.loss(scores, labels, x_lengths)
    if is_train is True:
        # Backpropagate the loss and update the model's weights
        loss.backward()
        optimizer.step()
    # Create a mask by filtering out all labels that are not a padding value
    mask = (labels <= 1).view_as(scores)
    # Completely remove the padded values from the labels and the scores using the mask
    unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask)
    unpadded_scores = torch.masked_select(scores, mask)
    # Get the top class (highest output probability) and find the samples where they are correct
    if model.n_outputs == 1:
        pred = torch.round(unpadded_scores)
    else:
        top_prob, top_class = unpadded_scores.topk(1)
    correct_pred = pred == unpadded_labels
    return correct_pred, unpadded_scores, unpadded_labels, loss


def inference_iter_mlp(model, features, labels, cols_to_remove=0,
                       is_train=False, optimizer=None):
    '''Run a single inference or training iteration on a Multilayer Perceptron (MLP),
    applied to two dimensional / tabular data. Performance metrics still need to be
    calculated after executing this method.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a
        classification task.
    features : torch.Tensor
        Data tensor that contains the features on which to run the model.
    labels : torch.Tensor
        Tensor that contains the labels for each row.
    cols_to_remove : int or list of ints, default 0
        Index or list of indeces of columns to remove from the features before
        feeding to the model. This tend to be the identifier columns, such as 
        `subject_id` and `ts` (timestamp).
    is_train : bool, default True
        Indicates if the method is being called in a training loop. If
        set to True, the network's weights will be updated by the
        given optimizer.
    optimizer : torch.optim.Optimizer
        Optimization algorthim, responsible for updating the model's
        weights in order to minimize (or maximize) the intended goal.

    Returns
    -------
    correct_pred : torch.Tensor
        Binary data tensor with the prediction results, indicating 1
        if the prediction is correct and 0 otherwise.
    scores : torch.Tensor
        Data tensor containing the output scores resulting from the
        inference.
    loss : torch.nn.modules.loss
        Obtained loss value. Although the optimization step can be
        done inside this method, the loss value could be useful as
        a metric.
    '''
    # Make the data have type float instead of double, as it would cause problems
    features, labels = features.float(), labels.float()
    # Remove unwanted columns from the data
    features = remove_tensor_column(features, cols_to_remove, inplace=True)
    # Feedforward the data through the model
    scores = model.forward(features)
    # Calculate the cross entropy loss
    loss = model.loss(scores, labels)
    if is_train is True:
        # Backpropagate the loss and update the model's weights
        loss.backward()
        optimizer.step()
    # Get the top class (highest output probability) and find the samples where they are correct
    if model.n_outputs == 1:
        pred = torch.round(scores)
    else:
        top_prob, top_class = scores.topk(1)
    correct_pred = top_class.view_as(labels) == labels
    return correct_pred, scores, loss


def model_inference(model, seq_len_dict, dataloader=None, data=None, metrics=['loss', 'accuracy', 'AUC'],
                    model_type='multivariate_rnn', padding_value=999999, output_rounded=False,
                    experiment=None, set_name='test', seq_final_outputs=False, cols_to_remove=[0, 1]):
    '''Do inference on specified data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which does the inference on the data.
    seq_len_dict : dict
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during inference.
    data : tuple of torch.Tensor, default None
        If a data loader isn't specified, the user can input directly a
        tuple of PyTorch tensor on which inference will be done. The first
        tensor must correspond to the features tensor whe second one
        should be the labels tensor.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (loss), accuracy, AUC
        (Receiver Operating Curve Area Under the Curve), precision, recall
        and F1.
    model_type : string, default 'multivariate_rnn'
        Sets the type of model to train. Important to know what type of
        inference to do. Currently available options are ['multivariate_rnn',
        'mlp'].
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    output_rounded : bool, default False
        If True, the output is rounded, to represent the class assigned by
        the model, instead of just probabilities (>= 0.5 rounded to 1,
        otherwise it's 0)
    experiment : comet_ml.Experiment, default None
        Represents a connection to a Comet.ml experiment to which the
        metrics performance is uploaded, if specified.
    set_name : str
        Defines what name to give to the set when uploading the metrics
        values to the specified Comet.ml experiment.
    seq_final_outputs : bool, default False
        If set to true, the function only returns the ouputs given at each
        sequence's end.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp).

    Returns
    -------
    output : torch.Tensor
        Contains the output scores (or classes, if output_rounded is set to
        True) for all of the input data.
    metrics_vals : dict of floats
        Dictionary containing the calculated performance on each of the
        specified metrics.
    '''
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()

    # Create an empty dictionary with all the possible metrics
    metrics_vals = {'loss': None,
                    'accuracy': None,
                    'AUC': None,
                    'precision': None,
                    'recall': None,
                    'F1': None}

    # Initialize the metrics
    if 'loss' in metrics:
        loss = 0
    if 'accuracy' in metrics:
        acc = 0
    if 'AUC' in metrics:
        auc = 0
    if 'AUC_weighted' in metrics:
        auc_wgt = 0
    if 'precision' in metrics:
        prec = 0
    if 'recall' in metrics:
        rcl = 0
    if 'F1' in metrics:
        f1_score = 0

    # Check if the user wants to do inference directly on a PyTorch tensor
    if dataloader is None and data is not None:
        features, labels = data[0], data[1]
        # Do inference on the data
        if model_type.lower() == 'multivariate_rnn':
            correct_pred, scores,
            labels, loss = inference_iter_multi_var_rnn(model, features, labels,
                                                        cols_to_remove, is_train=False,
                                                        optimizer=optimizer)
        elif model_type.lower() == 'mlp':
            correct_pred, scores, loss = inference_iter_mlp(model, features, labels,
                                                            cols_to_remove, is_train=False,
                                                            optimizer=optimizer)
        else:
            raise Exception('ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {threshold_type}.')
        if output_rounded is True:
            # Get the predicted classes
            output = pred.int()
        else:
            # Get the model scores (class probabilities)
            output = unpadded_scores
        if seq_final_outputs is True:
            # Only get the outputs retrieved at the sequences' end
            # Cumulative sequence lengths
            final_seq_idx = np.cumsum(x_lengths) - 1
            # Get the outputs of the last instances of each sequence
            output = output[final_seq_idx]
        if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
            # Calculate the number of true positives, false negatives, true negatives and false positives
            true_pos = int(sum(torch.masked_select(pred, labels.bool())))
            false_neg = int(sum(torch.masked_select(pred == 0, labels.bool())))
            true_neg = int(sum(torch.masked_select(pred == 0, (labels == 0).bool())))
            false_pos = int(sum(torch.masked_select(pred, (labels == 0).bool())))

        if 'loss' in metrics:
            # Add the loss of the current batch
            metrics_vals['loss'] = model.loss(scores, labels, x_lengths).item()
        if 'accuracy' in metrics:
            # Add the accuracy of the current batch, ignoring all padding values
            metrics_vals['accuracy'] = torch.mean(correct_pred.type(torch.FloatTensor)).item()
        if 'AUC' in metrics:
            # Add the ROC AUC of the current batch
            if model.n_outputs == 1:
                metrics_vals['AUC'] = roc_auc_score(labels.numpy(), scores.detach().numpy())
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                metrics_vals['AUC'] = roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                    multi_class='ovr', average='macro', labels=labels_in_batch.numpy())
        if 'AUC_weighted' in metrics:
            # Calculate a weighted version of the AUC; important for imbalanced datasets
            if model.n_outputs == 1:
                raise Exception('ERROR: The performance metric `AUC_weighted` is only available for multiclass tasks. Consider using the metric `AUC` for your single output model.')
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                metrics_vals['AUC_weighted'] = roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                             multi_class='ovr', average='weighted', labels=labels_in_batch.numpy())
        if 'precision' in metrics:
            # Add the precision of the current batch
            curr_prec = true_pos / (true_pos + false_pos)
            metrics_vals['precision'] = curr_prec
        if 'recall' in metrics:
            # Add the recall of the current batch
            curr_rcl = true_pos / (true_pos + false_neg)
            metrics_vals['recall'] = curr_rcl
        if 'F1' in metrics:
            # Check if precision has not yet been calculated
            if 'curr_prec' not in locals():
                curr_prec = true_pos / (true_pos + false_pos)
            # Check if recall has not yet been calculated
            if 'curr_rcl' not in locals():
                curr_rcl = true_pos / (true_pos + false_neg)
            # Add the F1 score of the current batch
            metrics_vals['F1'] = 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl)

        return output, metrics_vals

    # Initialize the output
    output = torch.tensor([]).int()

    # Evaluate the model on the set
    for features, labels in dataloader:
        # Turn off gradients, saves memory and computations
        with torch.no_grad():
            # Do inference on the data
            if model_type.lower() == 'multivariate_rnn':
                correct_pred, scores,
                labels, cur_loss = inference_iter_multi_var_rnn(model, features, labels,
                                                                cols_to_remove, is_train=False,
                                                                optimizer=optimizer)
            elif model_type.lower() == 'mlp':
                correct_pred, scores, cur_loss = inference_iter_mlp(model, features, labels,
                                                                    cols_to_remove, is_train=False,
                                                                    optimizer=optimizer)
            else:
                raise Exception('ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {threshold_type}.')
            if output_rounded is True:
                # Get the predicted classes
                output = torch.cat([output, torch.round(scores).int()])
            else:
                # Get the model scores (class probabilities)
                output = torch.cat([output.float(), scores])

            if seq_final_outputs is True:
                # Indeces at the end of each sequence
                final_seq_idx = [n_subject*features.shape[1]+x_lengths[n_subject]-1 for n_subject in range(features.shape[0])]

                # Get the outputs of the last instances of each sequence
                output = output[final_seq_idx]

            if any(mtrc in metrics for mtrc in ['precision', 'recall', 'F1']):
                # Calculate the number of true positives, false negatives, true negatives and false positives
                true_pos = int(sum(torch.masked_select(pred, labels.bool())))
                false_neg = int(sum(torch.masked_select(pred == 0, labels.bool())))
                true_neg = int(sum(torch.masked_select(pred == 0, (labels == 0).bool())))
                false_pos = int(sum(torch.masked_select(pred, (labels == 0).bool())))

            if 'loss' in metrics:
                # Add the loss of the current batch
                loss += cur_loss
            if 'accuracy' in metrics:
                # Get the correct predictions
                correct_pred = pred == labels
                # Add the accuracy of the current batch, ignoring all padding values
                acc += torch.mean(correct_pred.type(torch.FloatTensor))
            if 'AUC' in metrics:
            # Add the ROC AUC of the current batch
            if model.n_outputs == 1:
                auc += roc_auc_score(labels.numpy(), scores.detach().numpy())
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                auc += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                     multi_class='ovr', average='macro', labels=labels_in_batch.numpy())
            if 'AUC_weighted' in metrics:
                # Calculate a weighted version of the AUC; important for imbalanced datasets
                if model.n_outputs == 1:
                    raise Exception('ERROR: The performance metric `AUC_weighted` is only available for multiclass tasks. Consider using the metric `AUC` for your single output model.')
                else:
                    # It might happen that not all labels are present in the current batch;
                    # as such, we must focus on the ones that appear in the batch
                    labels_in_batch = labels.unique().long()
                    auc_wgt += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                             multi_class='ovr', average='weighted', labels=labels_in_batch.numpy())
            if 'precision' in metrics:
                # Add the precision of the current batch
                curr_prec = true_pos / (true_pos + false_pos)
                prec += curr_prec
            if 'recall' in metrics:
                # Add the recall of the current batch
                curr_rcl = true_pos / (true_pos + false_neg)
                rcl += curr_rcl
            if 'F1' in metrics:
                # Check if precision has not yet been calculated
                if 'curr_prec' not in locals():
                    curr_prec = true_pos / (true_pos + false_pos)
                # Check if recall has not yet been calculated
                if 'curr_rcl' not in locals():
                    curr_rcl = true_pos / (true_pos + false_neg)
                # Add the F1 score of the current batch
                f1_score += 2 * curr_prec * curr_rcl / (curr_prec + curr_rcl)

    # Calculate the average of the metrics over the batches
    if 'loss' in metrics:
        metrics_vals['loss'] = loss / len(dataloader)
        # Get just the value, not a tensor
        metrics_vals['loss'] = metrics_vals['loss'].item()
    if 'accuracy' in metrics:
        metrics_vals['accuracy'] = acc / len(dataloader)
        # Get just the value, not a tensor
        metrics_vals['accuracy'] = metrics_vals['accuracy'].item()
    if 'AUC' in metrics:
        metrics_vals['AUC'] = auc / len(dataloader)
    if 'precision' in metrics:
        metrics_vals['precision'] = prec / len(dataloader)
    if 'recall' in metrics:
        metrics_vals['recall'] = rcl / len(dataloader)
    if 'F1' in metrics:
        metrics_vals['F1'] = f1_score / len(dataloader)

    if experiment is not None:
        # Log metrics to Comet.ml
        if 'loss' in metrics:
            experiment.log_metric(f'{set_name}_loss', metrics_vals['loss'])
        if 'accuracy' in metrics:
            experiment.log_metric(f'{set_name}_acc', metrics_vals['accuracy'])
        if 'AUC' in metrics:
            experiment.log_metric(f'{set_name}_auc', metrics_vals['AUC'])
        if 'precision' in metrics:
            experiment.log_metric(f'{set_name}_prec', metrics_vals['precision'])
        if 'recall' in metrics:
            experiment.log_metric(f'{set_name}_rcl', metrics_vals['recall'])
        if 'F1' in metrics:
            experiment.log_metric(f'{set_name}_f1_score', metrics_vals['F1'])

    return output, metrics_vals


def train(model, train_dataloader, val_dataloader, test_dataloader=None,
          cols_to_remove=[0, 1], model_type='multivariate_rnn',
          seq_len_dict=None, batch_size=32, n_epochs=50, lr=0.001, 
          model_path='models/', ModelClass=None, padding_value=999999, 
          do_test=True, log_comet_ml=False, comet_ml_api_key=None,
          comet_ml_project_name=None, comet_ml_workspace=None,
          comet_ml_save_model=False, experiment=None, features_list=None,
          get_val_loss_min=False):
    '''Trains a given model on the provided data.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which is trained on the data to perform a
        classification task.
    train_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches during training.
    val_dataloader : torch.utils.data.DataLoader
        Data loader which will be used to get data batches when evaluating
        the model's performance on a validation set during training.
    test_dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches whe evaluating
        the model's performance on a test set, after finishing the
        training process.
    cols_to_remove : list of ints, default [0, 1]
        List of indeces of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as subject_id
        and ts (timestamp
    model_type : string, default 'multivariate_rnn'
        Sets the type of model to train. Important to know what type of
        inference to do. Currently available options are ['multivariate_rnn',
        'mlp'].
    seq_len_dict : dict, default None
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    batch_size : int, default 32
        Defines the batch size, i.e. the number of samples used in each
        training iteration to update the model's weights.
    n_epochs : int, default 50
        Number of epochs, i.e. the number of times the training loop
        iterates through all of the training data.
    lr : float, default 0.001
        Learning rate used in the optimization algorithm.
    model_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    ModelClass : object, default None
        Sets the class which corresponds to the machine learning 
        model type. It will be needed if test inference is 
        performed (do_test set to True), as we need to know
        the model type so as to load the best scored model.
    padding_value : numeric, default 999999
        Value to use in the padding, to fill the sequences.
    do_test : bool, default True
        If true, evaluates the model on the test set, after completing
        the training.
    log_comet_ml : bool, default False
        If true, makes the code upload a training report and metrics
        to comet.ml, a online platform which allows for a detailed
        version control for machine learning models.
    comet_ml_api_key : string, default None
        Comet.ml API key used when logging data to the platform.
    comet_ml_project_name : string, default None
        Name of the comet.ml project used when logging data to the
        platform.
    comet_ml_workspace : string, default None
        Name of the comet.ml workspace used when logging data to the
        platform.
    comet_ml_save_model : bool, default False
        If set to true, uploads the model with the lowest validation loss
        to comet.ml when logging data to the platform.
    experiment : comet_ml.Experiment, default None
        Defines an already existing Comet.ml experiment object to be used in the
        training. If not defined (None), a new experiment is created inside the
        method. In any case, a Comet.ml experiment is only used if log_comet_ml
        is set to True and the remaining necessary Comet.ml related parameters
        (comet_ml_api_key, comet_ml_project_name, comet_ml_workspace) are
        properly set up.
    features_list : list of strings, default None
        Names of the features being used in the current pipeline. This
        will be logged to comet.ml, if activated, in order to have a
        more detailed version control.
    get_val_loss_min : bool, default False
        If set to True, besides returning the trained model, the method also
        returns the minimum validation loss found during training.

    Returns
    -------
    model : nn.Module
        The same input model but with optimized weight values.
    val_loss_min : float
        If get_val_loss_min is set to True, the method also returns the minimum
        validation loss found during training.
    '''
    # Register all the hyperparameters
    model_args = inspect.getfullargspec(model.__init__).args[1:]
    hyper_params = dict([(param, getattr(model, param))
                         for param in model_args])
    hyper_params.update({'batch_size': batch_size,
                         'n_epochs': n_epochs,
                         'learning_rate': lr})
    
    if log_comet_ml is True:
        if experiment is None:
            # Create a new Comet.ml experiment
            experiment = Experiment(api_key=comet_ml_api_key, project_name=comet_ml_project_name, workspace=comet_ml_workspace)
        experiment.log_other("completed", False)
        experiment.log_other("random_seed", du.random_seed)
        # Report hyperparameters to Comet.ml
        experiment.log_parameters(hyper_params)
        if features_list is not None:
            # Log the names of the features being used
            experiment.log_other("features_list", features_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)                 # Adam optimization algorithm
    step = 0                                                                # Number of iteration steps done so far
    print_every = 10                                                        # Steps interval where the metrics are printed
    train_on_gpu = torch.cuda.is_available()                                # Check if GPU is available
    val_loss_min = np.inf                                                   # Start with an infinitely big minimum validation loss

    for epoch in range(1, n_epochs+1):
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0
        train_auc = 0
        if model.n_outputs > 1:
            train_auc_wgt = 0

        try:
            # Loop through the training data
            for features, labels in train_dataloader:
                # Activate dropout to train the model
                model.train()
                # Clear the gradients of all optimized variables
                optimizer.zero_grad()

                if train_on_gpu is True:
                    # Move data to GPU
                    features, labels = features.cuda(), labels.cuda()

                # Do inference on the data
                if model_type.lower() == 'multivariate_rnn':
                    correct_pred, scores,
                    labels, loss = inference_iter_multi_var_rnn(model, features, labels,
                                                                cols_to_remove, is_train=True,
                                                                optimizer=optimizer)
                elif model_type.lower() == 'mlp':
                    correct_pred, scores, loss = inference_iter_mlp(model, features, labels,
                                                                    cols_to_remove, is_train=True,
                                                                    optimizer=optimizer)
                else:
                    raise Exception('ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {threshold_type}.')
                train_loss += loss                                              # Add the training loss of the current batch
                train_acc += torch.mean(correct_pred.type(torch.FloatTensor))   # Add the training accuracy of the current batch, ignoring all padding values
                # Add the training ROC AUC of the current batch
                if model.n_outputs == 1:
                    train_auc += roc_auc_score(labels.numpy(), scores.detach().numpy())
                else:
                    # It might happen that not all labels are present in the current batch;
                    # as such, we must focus on the ones that appear in the batch
                    labels_in_batch = labels.unique().long()                    
                    train_auc += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                               multi_class='ovr', average='macro', labels=labels_in_batch.numpy())
                    # Also calculate a weighted version of the AUC; important for imbalanced dataset
                    train_auc_wgt += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                   multi_class='ovr', average='weighted', labels=labels_in_batch.numpy())
                step += 1                                                       # Count one more iteration step
                model.eval()                                                    # Deactivate dropout to test the model

                # Initialize the validation metrics
                val_loss = 0
                val_acc = 0
                val_auc = 0
                if model.n_outputs > 1:
                    val_auc_wgt = 0

                # Loop through the validation data
                for features, labels in val_dataloader:
                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        # Do inference on the data
                        if model_type.lower() == 'multivariate_rnn':
                            correct_pred, scores,
                            labels, loss = inference_iter_multi_var_rnn(model, features, labels,
                                                                        cols_to_remove, is_train=False,
                                                                        optimizer=optimizer)
                        elif model_type.lower() == 'mlp':
                            correct_pred, scores, loss = inference_iter_mlp(model, features, labels,
                                                                            cols_to_remove, is_train=False,
                                                                            optimizer=optimizer)
                        else:
                            raise Exception('ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {threshold_type}.')
                        val_loss += loss                                                # Add the validation loss of the current batch
                        val_acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the validation accuracy of the current batch, ignoring all padding values
                        # Add the training ROC AUC of the current batch
                        if model.n_outputs == 1:
                            val_auc += roc_auc_score(labels.numpy(), scores.detach().numpy())
                        else:
                            # It might happen that not all labels are present in the current batch;
                            # as such, we must focus on the ones that appear in the batch
                            
                            labels_in_batch = labels.unique().long()
                            val_auc += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                    multi_class='ovr', average='macro', labels=labels_in_batch.numpy())
                            # Also calculate a weighted version of the AUC; important for imbalanced dataset
                            val_auc_wgt += roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                        multi_class='ovr', average='weighted', labels=labels_in_batch.numpy())

                # Calculate the average of the metrics over the batches
                val_loss = val_loss / len(val_dataloader)
                val_acc = val_acc / len(val_dataloader)
                val_auc = val_auc / len(val_dataloader)
                if model.n_outputs > 1:
                    val_auc_wgt = val_auc_wgt / len(val_dataloader)

                # Display validation loss
                if step%print_every == 0:
                    print(f'Epoch {epoch} step {step}: Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
                # Check if the performance obtained in the validation set is the best so far (lowest loss value)
                if val_loss < val_loss_min:
                    print(f'New minimum validation loss: {val_loss_min} -> {val_loss}.')
                    # Update the minimum validation loss
                    val_loss_min = val_loss
                    # Get the current day and time to attach to the saved model's name
                    current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
                    # Filename and path where the model will be saved
                    model_filename = f'{model_path}checkpoint_{current_datetime}.pth'
                    print(f'Saving model in {model_filename}')
                    # Save the best performing model so far, along with additional information to implement it
                    checkpoint = hyper_params
                    checkpoint['state_dict'] = model.state_dict()
                    torch.save(checkpoint, model_filename)

                    if log_comet_ml is True and comet_ml_save_model is True:
                        # Upload the model to Comet.ml
                        experiment.log_asset(file_data=model_filename, overwrite=True)

            # Calculate the average of the metrics over the epoch
            train_loss = train_loss / len(train_dataloader)
            train_acc = train_acc / len(train_dataloader)
            train_auc = train_auc / len(train_dataloader)
            if model.n_outputs > 1:
                train_auc_wgt = train_auc_wgt / len(train_dataloader)

            if log_comet_ml is True:
                # Log metrics to Comet.ml
                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("train_acc", train_acc, step=epoch)
                experiment.log_metric("train_auc", train_auc, step=epoch)
                experiment.log_metric("val_loss", val_loss, step=epoch)
                experiment.log_metric("val_acc", val_acc, step=epoch)
                experiment.log_metric("val_auc", val_auc, step=epoch)
                experiment.log_metric("epoch", epoch)
                if model.n_outputs > 1:
                    experiment.log_metric("train_auc_wgt", train_auc_wgt, step=epoch)
                    experiment.log_metric("val_auc_wgt", val_auc_wgt, step=epoch)

            # Print a report of the epoch
            print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; Training AUC: {train_auc}; \
                    Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
            print('----------------------')
        except Exception:
            warnings.warn(f'There was a problem doing training epoch {epoch}. Ending training.')

    try:
        if do_test is True and model_filename is not None:
            # Load the model with the best validation performance
            model = load_checkpoint(model_filename, ModelClass)

            # Run inference on the test data
            model_inference(model, seq_len_dict, dataloader=test_dataloader , experiment=experiment)
    except UnboundLocalError:
        warnings.warn('Inference failed due to non existent saved models. Skipping evaluation on test set.')
    except Exception:
        warnings.warn(f'Inference failed due to {sys.exc_info()[0]}. Skipping evaluation on test set.')

    if log_comet_ml is True:
        # Only report that the experiment completed successfully if it finished the training without errors
        experiment.log_other("completed", True)

    if get_val_loss_min is True:
        # Also return the minimum validation loss alongside the corresponding model
        return model, val_loss_min.item()
    else:
        return model
