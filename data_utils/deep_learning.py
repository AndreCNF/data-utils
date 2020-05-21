from comet_ml import Experiment                         # Comet.ml can log training metrics, parameters, do version control and parameter optimization
import torch                                            # PyTorch to create and apply deep learning models
import torch.nn as nn                                   # Pytorch activation functions, to introduce non-linearoity and get output probabilities
import numpy as np                                      # NumPy to handle numeric and NaN operations
import warnings                                         # Print warnings for bad practices
from datetime import datetime                           # datetime to use proper date and time formats
import sys                                              # Identify types of exceptions
import inspect                                          # Inspect methods and their arguments
from sklearn.metrics import roc_auc_score               # ROC AUC model performance metric
from . import utils                                     # Generic and useful methods
from . import search_explore                            # Methods to search and explore data
from . import padding                                   # Padding and variable sequence length related methods
from . import machine_learning                          # Machine learning focused pipeline methods
import data_utils as du

# Ignore Dask's 'meta' warning
warnings.filterwarnings('ignore', message='`meta` is not specified, inferred from partial data. Please provide `meta` if the result is unexpected.')

# Get the activation functions that might be used
sigmoid = nn.Sigmoid()
softmax = nn.Softmax()

# Methods

def remove_tensor_column(data, col_idx, inplace=False):
    '''Remove a column(s) from a PyTorch tensor.

    Parameters
    ----------
    data : torch.Tensor or numpy.Array
        Data tensor or array that contains the column(s) that will be removed.
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
    # Sort the list of columns in descending order, so as to avoid removing the
    # wrong columns
    col_idx.sort(reverse=True)
    for col in col_idx:
        if col is None:
            continue
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


def load_checkpoint(filepath, ModelClass, *args):
    '''Load a model from a specified path and name.

    Parameters
    ----------
    filepath : str
        Path to the model being loaded, including it's own file name.
    ModelClass : torch.nn.Module (any deep learning model)
        Class constructor for the desired deep learning model.
    args : list of str
        Names of the neural network's parameters that need to be
        loaded.

    Returns
    -------
    model : nn.Module
        The loaded model with saved weight values.
    '''
    # Check if GPU is available
    on_gpu = torch.cuda.is_available()
    # Load the model
    if on_gpu is False:
        # Make sure that the model is loaded onto the CPU, if no GPU is available
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(filepath)
    if len(args) == 0:
        # Fetch all the model parameters' names
        args = inspect.getfullargspec(ModelClass.__init__).args[1:]
    # Retrieve the parameters' values and integrate them in
    # a dictionary
    params = dict(zip(args, [checkpoint[param] for param in args]))
    # Create a model with the saved parameters
    model = ModelClass(**params)
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


def inference_iter_multi_var_rnn(model, features, labels,
                                 padding_value=999999, cols_to_remove=[0, 1],
                                 is_train=False, prob_output=True, optimizer=None,
                                 is_custom=False, already_embedded=False,
                                 seq_lengths=None, distributed_train=False):
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
    seq_len_dict : dict, default None
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
    padding_value : numeric
        Value to use in the padding, to fill the sequences.
    output_rounded : bool, default False
    cols_to_remove : list of ints, default [0, 1]
        List of indices of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as `subject_id`
        and `ts` (timestamp).
    is_train : bool, default True
        Indicates if the method is being called in a training loop. If
        set to True, the network's weights will be updated by the
        given optimizer.
    prob_output : bool, default True
        If set to True, the model's output will be given in class probabilities
        format. Otherwise, the output comes as the original logits.
    optimizer : torch.optim.Optimizer, default None
        Optimization algorthim, responsible for updating the model's
        weights in order to minimize (or maximize) the intended goal.
    is_custom : bool, default False
        If set to True, the method will assume that the model being used is a
        custom built one, which won't require sequence length information during
        the feedforward process.
    already_embedded : bool, default False
        If set to True, it means that the categorical features are already
        embedded when fetching a batch, i.e. there's no need to run the embedding
        layer(s) during the model's feedforward.
    seq_lengths : list or numpy.ndarray or torch.Tensor
        List of sequence lengths, relative to the input data.
    distributed_train : bool, default False
        Indicates whether the model is wrapped in a DistributedDataParallel 
        wrapper (i.e. if the model is being trained in a distributed training
        context).

    Returns
    -------
    pred : torch.Tensor
        One hot encoded tensor with 1's in the columns of the predicted
        classes and 0's in the rest.
    correct_pred : torch.Tensor
        Boolean data tensor with the prediction results, indicating True
        if the prediction is correct and False otherwise.
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
    global sigmoid, softmax
    # Make the data have type float instead of double, as it would cause problems
    features, labels = features.float(), labels.float()
    # Remove unwanted columns from the data
    features = remove_tensor_column(features, cols_to_remove, inplace=True)
    # Feedforward the data through the model
    if is_custom is False:
        scores = model.forward(features, get_hidden_state=False, seq_lengths=seq_lengths,
                               prob_output=False, already_embedded=already_embedded)
    else:
        scores = model.forward(features, get_hidden_state=False, prob_output=False,
                               already_embedded=already_embedded)
    if distributed_train is True:
        # Get the original model's custom attributes and methods
        model_loss = model.module.loss
        model_n_outputs = model.module.n_outputs
    else:
        model_loss = model.loss
        model_n_outputs = model.n_outputs
    # Calculate the negative log likelihood loss
    loss = model_loss(scores, labels)
    if is_train is True:
        # Backpropagate the loss and update the model's weights
        loss.backward()
        optimizer.step()
    # Create a mask by filtering out all labels that are not a padding value
    mask = (labels != padding_value).view_as(scores)
    # Completely remove the padded values from the labels and the scores using the mask
    unpadded_labels = torch.masked_select(labels.contiguous().view_as(scores), mask)
    unpadded_scores = torch.masked_select(scores, mask)
    if prob_output is True:
        # Get the outputs in the form of probabilities
        if model_n_outputs == 1:
            unpadded_scores = sigmoid(unpadded_scores)
        else:
            # Normalize outputs on their last dimension
            unpadded_scores = softmax(unpadded_scores, dim=len(unpadded_scores.shape)-1)
    # Get the top class (highest output probability) and find the samples where they are correct
    if model_n_outputs == 1:
        if prob_output is True:
            pred = torch.round(unpadded_scores)
        else:
            if model_n_outputs == 1:
                pred = torch.round(sigmoid(unpadded_scores))
            else:
                pred = torch.round(softmax(unpadded_scores))
    else:
        top_prob, pred = unpadded_scores.topk(1)
    correct_pred = pred == unpadded_labels
    return pred, correct_pred, unpadded_scores, unpadded_labels, loss


def inference_iter_mlp(model, features, labels, cols_to_remove=0,
                       is_train=False, prob_output=True, optimizer=None):
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
        Index or list of indices of columns to remove from the features before
        feeding to the model. This tend to be the identifier columns, such as
        `subject_id` and `ts` (timestamp).
    is_train : bool, default True
        Indicates if the method is being called in a training loop. If
        set to True, the network's weights will be updated by the
        given optimizer.
    prob_output : bool, default True
        If set to True, the model's output will be given in class probabilities
        format. Otherwise, the output comes as the original logits.
    optimizer : torch.optim.Optimizer
        Optimization algorthim, responsible for updating the model's
        weights in order to minimize (or maximize) the intended goal.

    Returns
    -------
    pred : torch.Tensor
        One hot encoded tensor with 1's in the columns of the predicted
        classes and 0's in the rest.
    correct_pred : torch.Tensor
        Boolean data tensor with the prediction results, indicating True
        if the prediction is correct and False otherwise.
    scores : torch.Tensor
        Data tensor containing the output scores resulting from the
        inference.
    loss : torch.nn.modules.loss
        Obtained loss value. Although the optimization step can be
        done inside this method, the loss value could be useful as
        a metric.
    '''
    global sigmoid, softmax
    # Make the data have type float instead of double, as it would cause problems
    features, labels = features.float(), labels.float()
    # Remove unwanted columns from the data
    features = remove_tensor_column(features, cols_to_remove, inplace=True)
    # Feedforward the data through the model
    scores = model.forward(features, prob_output=False)
    # Calculate the negative log likelihood loss
    loss = model.loss(scores, labels)
    if is_train is True:
        # Backpropagate the loss and update the model's weights
        loss.backward()
        optimizer.step()
    if prob_output is True:
        # Get the outputs in the form of probabilities
        if model.n_outputs == 1:
            scores = sigmoid(scores)
        else:
            # Normalize outputs on their last dimension
            scores = softmax(scores, dim=len(scores.shape)-1)
    # Get the top class (highest output probability) and find the samples where they are correct
    if model.n_outputs == 1:
        if prob_output is True:
            pred = torch.round(scores)
        else:
            if model.n_outputs == 1:
                pred = torch.round(sigmoid(scores))
            else:
                pred = torch.round(softmax(scores))
    else:
        top_prob, pred = scores.topk(1)
    # Certify that labels are of type long, like the predictions `pred`
    labels = labels.long()
    correct_pred = pred.view_as(labels) == labels
    return pred, correct_pred, scores, loss


def model_inference(model, dataloader=None, data=None, dataset=None,
                    metrics=['loss', 'accuracy', 'AUC'], model_type='multivariate_rnn',
                    is_custom=False, seq_len_dict=None, padding_value=999999,
                    output_rounded=False, experiment=None, set_name='test',
                    seq_final_outputs=False, cols_to_remove=[0, 1],
                    already_embedded=False, see_progress=True):
    '''Do inference on specified data using a given model.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model which does the inference on the data.
    dataloader : torch.utils.data.DataLoader, default None
        Data loader which will be used to get data batches during inference.
    data : tuple of torch.Tensor, default None
        If a data loader isn't specified, the user can input directly a
        tuple of PyTorch tensor on which inference will be done. The first
        tensor must correspond to the features tensor whe second one
        should be the labels tensor.
    dataset : torch.utils.data.Dataset
        Dataset object that contains the data used to train, validate and test
        the machine learning models. Having the dataloaders set, this argument
        is only needed if the data has variable sequence length and its dataset
        object loads files in each batch, instead of data from a single file.
        In essence, it's needed to give us the current batch's sequence length
        information, when we couldn't have known this for the whole data
        beforehand.
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (`loss`), accuracy (`accuracy`),
        AUC (`AUC`), weighted AUC (`AUC_weighted`), precision (`precision`),
        recall (`recall`) and F1 (`F1`).
    model_type : string, default 'multivariate_rnn'
        Sets the type of model to train. Important to know what type of
        inference to do. Currently available options are ['multivariate_rnn',
        'mlp'].
    is_custom : bool, default False
        If set to True, the method will assume that the model being used is a
        custom built one, which won't require sequence length information during
        the feedforward process.
    seq_len_dict : dict, default None
        Dictionary containing the sequence lengths for each index of the
        original dataframe. This allows to ignore the padding done in
        the fixed sequence length tensor.
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
        List of indices of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as `subject_id`
        and `ts` (timestamp).
    already_embedded : bool, default False
        If set to True, it means that the categorical features are already
        embedded when fetching a batch, i.e. there's no need to run the embedding
        layer(s) during the model's feedforward.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of each loop.

    Returns
    -------
    output : torch.Tensor
        Contains the output scores (or classes, if output_rounded is set to
        True) for all of the input data.
    metrics_vals : dict of floats
        Dictionary containing the calculated performance on each of the
        specified metrics.
    '''
    global sigmoid, softmax
    # Guarantee that the model is in evaluation mode, so as to deactivate dropout
    model.eval()
    # Check if GPU is available
    on_gpu = torch.cuda.is_available()
    if on_gpu is True:
        # Move the model to GPU
        model = model.cuda()
    # Create an empty dictionary with all the possible metrics
    metrics_vals = {'loss': None,
                    'accuracy': None,
                    'AUC': None,
                    'AUC_weighted': None,
                    'precision': None,
                    'recall': None,
                    'F1': None}
    # Initialize the metrics
    if 'loss' in metrics:
        loss = 0
    if 'accuracy' in metrics:
        acc = 0
    if 'AUC' in metrics:
        auc = list()
    if 'AUC_weighted' in metrics:
        auc_wgt = list()
    if 'precision' in metrics:
        prec = 0
    if 'recall' in metrics:
        rcl = 0
    if 'F1' in metrics:
        f1_score = 0

    # Check if the user wants to do inference directly on a PyTorch tensor
    if dataloader is None and data is not None:
        features, labels = data[0], data[1]
        if is_custom is False or seq_final_outputs is True:
            # Find the original sequence lengths
            seq_lengths = search_explore.find_seq_len(labels, padding_value=padding_value)
            # [TODO] Dynamically calculate and pad according to the current batch's maximum sequence length
            # total_length = max(seq_lengths)
        else:
            # No need to find the sequence lengths now
            seq_lengths = None
            # total_length = None
        if on_gpu is True:
            # Move data to GPU
            features, labels = features.cuda(), labels.cuda()
        # Do inference on the data
        if model_type.lower() == 'multivariate_rnn':
            (pred, correct_pred,
             scores, labels, loss) = (inference_iter_multi_var_rnn(model, features, labels,
                                                                   padding_value=padding_value,
                                                                   cols_to_remove=cols_to_remove, is_train=False,
                                                                   prob_output=True, is_custom=is_custom,
                                                                   already_embedded=already_embedded,
                                                                   seq_lengths=seq_lengths))
        elif model_type.lower() == 'mlp':
            pred, correct_pred, scores, loss = (inference_iter_mlp(model, features, labels,
                                                                   cols_to_remove, is_train=False,
                                                                   prob_output=True))
        else:
            raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {model_type}.')
        if on_gpu is True:
            # Move data to CPU for performance computations
            correct_pred, scores, labels = correct_pred.cpu(), scores.cpu(), labels.cpu()
        if output_rounded is True:
            # Get the predicted classes
            output = pred
        else:
            # Get the model scores (class probabilities)
            output = scores
        if seq_final_outputs is True:
            # Only get the outputs retrieved at the sequences' end
            # Cumulative sequence lengths
            final_seq_idx = np.cumsum(seq_lengths) - 1
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
            metrics_vals['loss'] = loss
        if 'accuracy' in metrics:
            # Add the accuracy of the current batch, ignoring all padding values
            metrics_vals['accuracy'] = torch.mean(correct_pred.type(torch.FloatTensor)).item()
        if 'AUC' in metrics:
            # Add the ROC AUC of the current batch
            if model.n_outputs == 1:
                try:
                    metrics_vals['AUC'] = roc_auc_score(labels.numpy(), scores.detach().numpy())
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the AUC metric. Received exception "{str(e)}".')
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                try:
                    metrics_vals['AUC'] = roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                        multi_class='ovr', average='macro', labels=labels_in_batch.numpy())
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the AUC metric. Received exception "{str(e)}".')
        if 'AUC_weighted' in metrics:
            # Calculate a weighted version of the AUC; important for imbalanced datasets
            if model.n_outputs == 1:
                raise Exception('ERROR: The performance metric `AUC_weighted` is only available for multiclass tasks. Consider using the metric `AUC` for your single output model.')
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                try:
                    metrics_vals['AUC_weighted'] = roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                                 multi_class='ovr', average='weighted', labels=labels_in_batch.numpy())
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the weighted AUC metric. Received exception "{str(e)}".')
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
        # Remove the current features and labels from memory
        del features
        del labels

        return output, metrics_vals

    # Initialize the output
    output = torch.tensor([]).int()

    # Evaluate the model on the set
    for features, labels in utils.iterations_loop(dataloader,
                                                  see_progress=see_progress,
                                                  desc='Test batches'):
        if dataset is not None:
            # Make sure that the data has the right amount of dimensions
            features, labels = features.squeeze(), labels.squeeze()
        # Turn off gradients, saves memory and computations
        with torch.no_grad():
            if is_custom is False or seq_final_outputs is True:
                # Find the original sequence lengths
                seq_lengths = search_explore.find_seq_len(labels, padding_value=padding_value)
                # [TODO] Dynamically calculate and pad according to the current batch's maximum sequence length
                # total_length = max(seq_lengths)
            else:
                # No need to find the sequence lengths now
                seq_lengths = None
                # total_length = None
            if on_gpu is True:
                # Move data to GPU
                features, labels = features.cuda(), labels.cuda()
            # Do inference on the data
            if model_type.lower() == 'multivariate_rnn':
                (pred, correct_pred,
                 scores, labels, cur_loss) = (inference_iter_multi_var_rnn(model, features, labels,
                                                                           padding_value=padding_value,
                                                                           cols_to_remove=cols_to_remove, is_train=False,
                                                                           prob_output=True, is_custom=is_custom,
                                                                           already_embedded=already_embedded,
                                                                           seq_lengths=seq_lengths))
            elif model_type.lower() == 'mlp':
                pred, correct_pred, scores, cur_loss = (inference_iter_mlp(model, features, labels,
                                                                           cols_to_remove, is_train=False,
                                                                           prob_output=True))
            else:
                raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {model_type}.')
            if on_gpu is True:
                # Move data to CPU for performance computations
                correct_pred, scores, labels = correct_pred.cpu(), scores.cpu(), labels.cpu()
            if output_rounded is True:
                # Get the predicted classes
                output = torch.cat([output, torch.round(scores).int()])
            else:
                # Get the model scores (class probabilities)
                output = torch.cat([output.float(), scores])

            if seq_final_outputs is True:
                # Only get the outputs retrieved at the sequences' end
                # Cumulative sequence lengths
                final_seq_idx = np.cumsum(seq_lengths) - 1
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
                # Add the accuracy of the current batch, ignoring all padding values
                acc += torch.mean(correct_pred.type(torch.FloatTensor))
            # [TODO] Confirm if the AUC calculation works on the logit scores, without using the output probabilities
            if 'AUC' in metrics:
                # Add the ROC AUC of the current batch
                if model.n_outputs == 1:
                    try:
                        auc.append(roc_auc_score(labels.numpy(), scores.detach().numpy()))
                    except Exception as e:
                        warnings.warn(f'Couldn\'t calculate the AUC metric. Received exception "{str(e)}".')
                else:
                    # It might happen that not all labels are present in the current batch;
                    # as such, we must focus on the ones that appear in the batch
                    labels_in_batch = labels.unique().long()
                    try:
                        auc.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                 multi_class='ovr', average='macro', labels=labels_in_batch.numpy()))
                    except Exception as e:
                        warnings.warn(f'Couldn\'t calculate the AUC metric. Received exception "{str(e)}".')
            if 'AUC_weighted' in metrics:
                # Calculate a weighted version of the AUC; important for imbalanced datasets
                if model.n_outputs == 1:
                    raise Exception('ERROR: The performance metric `AUC_weighted` is only available for multiclass tasks. Consider using the metric `AUC` for your single output model.')
                else:
                    # It might happen that not all labels are present in the current batch;
                    # as such, we must focus on the ones that appear in the batch
                    labels_in_batch = labels.unique().long()
                    try:
                        auc_wgt.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                     multi_class='ovr', average='weighted', labels=labels_in_batch.numpy()))
                    except Exception as e:
                        warnings.warn(f'Couldn\'t calculate the AUC metric. Received exception "{str(e)}".')
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
            # Remove the current features and labels from memory
            del features
            del labels

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
        metrics_vals['AUC'] = np.mean(auc)
    if 'AUC_weighted' in metrics:
        metrics_vals['AUC_weighted'] = np.mean(auc_wgt)
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
        if 'AUC_weighted' in metrics:
            experiment.log_metric(f'{set_name}_auc_wgt', metrics_vals['AUC_weighted'])
        if 'precision' in metrics:
            experiment.log_metric(f'{set_name}_prec', metrics_vals['precision'])
        if 'recall' in metrics:
            experiment.log_metric(f'{set_name}_rcl', metrics_vals['recall'])
        if 'F1' in metrics:
            experiment.log_metric(f'{set_name}_f1_score', metrics_vals['F1'])

    return output, metrics_vals


def train(model, train_dataloader, val_dataloader, test_dataloader=None,
          dataset=None, cols_to_remove=[0, 1], model_type='multivariate_rnn',
          is_custom=False, seq_len_dict=None, batch_size=32, n_epochs=50,
          lr=0.001, clip_value=0.5, models_path='models/', model_name='checkpoint',
          ModelClass=None, padding_value=999999, do_test=True,
          metrics=['loss', 'accuracy', 'AUC'], log_comet_ml=False,
          comet_ml_api_key=None, comet_ml_project_name=None,
          comet_ml_workspace=None, comet_ml_save_model=False,
          experiment=None, features_list=None, get_val_loss_min=False,
          already_embedded=False, see_progress=True):
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
    dataset : torch.utils.data.Dataset
        Dataset object that contains the data used to train, validate and test
        the machine learning models. Having the dataloaders set, this argument
        is only needed if the data has variable sequence length and its dataset
        object loads files in each batch, instead of data from a single file.
        In essence, it's needed to give us the current batch's sequence length
        information, when we couldn't have known this for the whole data
        beforehand.
    cols_to_remove : list of ints, default [0, 1]
        List of indices of columns to remove from the features before feeding to
        the model. This tend to be the identifier columns, such as `subject_id`
        and `ts` (timestamp).
    model_type : string, default 'multivariate_rnn'
        Sets the type of model to train. Important to know what type of
        inference to do. Currently available options are ['multivariate_rnn',
        'mlp'].
    is_custom : bool, default False
        If set to True, the method will assume that the model being used is a
        custom built one, which won't require sequence length information during
        the feedforward process.
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
    clip_value : int or float, default 0.5
        Gradient clipping value, which limit the maximum change in the
        model parameters, so as to avoid exploiding gradients.
    models_path : string, default 'models/'
        Path where the model will be saved. By default, it saves in
        the directory named "models".
    model_name : string, default 'checkpoint'
        Name that will be given to the saved models. Validation loss and
        timestamp info will then be appended to the name.
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
    metrics : list of strings, default ['loss', 'accuracy', 'AUC'],
        List of metrics to be used to evaluate the model on the infered data.
        Available metrics are cross entropy loss (`loss`), accuracy (`accuracy`),
        AUC (`AUC`), weighted AUC (`AUC_weighted`), precision (`precision`),
        recall (`recall`) and F1 (`F1`).
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
    already_embedded : bool, default False
        If set to True, it means that the categorical features are already
        embedded when fetching a batch, i.e. there's no need to run the embedding
        layer(s) during the model's feedforward.
    see_progress : bool, default True
        If set to True, a progress bar will show up indicating the execution
        of each loop.

    Returns
    -------
    model : nn.Module
        The same input model but with optimized weight values.
    val_loss_min : float
        If get_val_loss_min is set to True, the method also returns the minimum
        validation loss found during training.
    '''
    global sigmoid, softmax
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
        experiment.log_other('completed', False)
        experiment.log_other('random_seed', du.random_seed)
        # Report hyperparameters to Comet.ml
        experiment.log_parameters(hyper_params)
        if features_list is not None:
            # Log the names of the features being used
            experiment.log_other('features_list', features_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)                 # Adam optimization algorithm
    step = 0                                                                # Number of iteration steps done so far
    print_every = 10                                                        # Steps interval where the metrics are printed
    on_gpu = torch.cuda.is_available()                                      # Check if GPU is available
    val_loss_min = np.inf                                                   # Start with an infinitely big minimum validation loss

    if on_gpu is True:
        # Move the model to GPU
        model = model.cuda()

    if clip_value is not None:
        # Set gradient clipping to avoid exploding gradients
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    for epoch in utils.iterations_loop(range(1, n_epochs+1),
                                       see_progress=see_progress,
                                       desc='Epochs'):
        # Initialize the training metrics
        train_loss = 0
        train_acc = 0
        train_auc = list()
        if model.n_outputs > 1:
            train_auc_wgt = list()

        # try:
        # Loop through the training data
        for features, labels in utils.iterations_loop(train_dataloader,
                                                      see_progress=see_progress,
                                                      desc='Steps',
                                                      leave=False):
            if dataset is not None:
                # Make sure that the data has the right amount of dimensions
                features, labels = features.squeeze(), labels.squeeze()
            if is_custom is False:
                # Find the original sequence lengths
                seq_lengths = search_explore.find_seq_len(labels, padding_value=padding_value)
                # [TODO] Dynamically calculate and pad according to the current batch's maximum sequence length
                # total_length = max(seq_lengths)
            else:
                # No need to find the sequence lengths now
                seq_lengths = None
                # total_length = None
            # Activate dropout to train the model
            model.train()
            # Clear the gradients of all optimized variables
            optimizer.zero_grad()
            if on_gpu is True:
                # Move data to GPU
                features, labels = features.cuda(), labels.cuda()
            # Do inference on the data
            if model_type.lower() == 'multivariate_rnn':
                (pred, correct_pred,
                 scores, labels, loss) = (inference_iter_multi_var_rnn(model, features, labels,
                                                                       padding_value=padding_value,
                                                                       cols_to_remove=cols_to_remove, is_train=True,
                                                                       prob_output=True, optimizer=optimizer,
                                                                       is_custom=is_custom,
                                                                       already_embedded=already_embedded,
                                                                       seq_lengths=seq_lengths))
            elif model_type.lower() == 'mlp':
                pred, correct_pred, scores, loss = (inference_iter_mlp(model, features, labels,
                                                                       cols_to_remove, is_train=True,
                                                                       prob_output=True, optimizer=optimizer))
            else:
                raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {model_type}.')
            train_loss += loss                                              # Add the training loss of the current batch
            train_acc += torch.mean(correct_pred.type(torch.FloatTensor))   # Add the training accuracy of the current batch, ignoring all padding values
            if on_gpu is True:
                # Move data to CPU for performance computations
                scores, labels = scores.cpu(), labels.cpu()
            # Add the training ROC AUC of the current batch
            if model.n_outputs == 1:
                try:
                    train_auc.append(roc_auc_score(labels.numpy(), scores.detach().numpy()))
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the training AUC on step {step}. Received exception "{str(e)}".')
            else:
                # It might happen that not all labels are present in the current batch;
                # as such, we must focus on the ones that appear in the batch
                labels_in_batch = labels.unique().long()
                try:
                    train_auc.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                   multi_class='ovr', average='macro', labels=labels_in_batch.numpy()))
                    # Also calculate a weighted version of the AUC; important for imbalanced dataset
                    train_auc_wgt.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                       multi_class='ovr', average='weighted', labels=labels_in_batch.numpy()))
                except Exception as e:
                    warnings.warn(f'Couldn\'t calculate the training AUC on step {step}. Received exception "{str(e)}".')
            step += 1                                                       # Count one more iteration step
            model.eval()                                                    # Deactivate dropout to test the model
            # Remove the current features and labels from memory
            del features
            del labels

            # Initialize the validation metrics
            val_loss = 0
            val_acc = 0
            val_auc = list()
            if model.n_outputs > 1:
                val_auc_wgt = list()
            # Loop through the validation data
            for features, labels in utils.iterations_loop(val_dataloader,
                                                          see_progress=see_progress,
                                                          desc='Val batches',
                                                          leave=False):
                if dataset is not None:
                    # Make sure that the data has the right amount of dimensions
                    features, labels = features.squeeze(), labels.squeeze()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    if is_custom is False:
                        # Find the original sequence lengths
                        seq_lengths = search_explore.find_seq_len(labels, padding_value=padding_value)
                        # [TODO] Dynamically calculate and pad according to the current batch's maximum sequence length
                        # total_length = max(seq_lengths)
                    else:
                        # No need to find the sequence lengths now
                        seq_lengths = None
                        # total_length = None
                    if on_gpu is True:
                        # Move data to GPU
                        features, labels = features.cuda(), labels.cuda()
                    # Do inference on the data
                    if model_type.lower() == 'multivariate_rnn':
                        (pred, correct_pred,
                         scores, labels, loss) = (inference_iter_multi_var_rnn(model, features, labels,
                                                                               padding_value=padding_value,
                                                                               cols_to_remove=cols_to_remove, is_train=False,
                                                                               prob_output=True, is_custom=is_custom,
                                                                               already_embedded=already_embedded,
                                                                               seq_lengths=seq_lengths))
                    elif model_type.lower() == 'mlp':
                        pred, correct_pred, scores, loss = (inference_iter_mlp(model, features, labels,
                                                                               cols_to_remove, is_train=False,
                                                                               prob_output=True))
                    else:
                        raise Exception(f'ERROR: Invalid model type. It must be "multivariate_rnn" or "mlp", not {model_type}.')
                    val_loss += loss                                                # Add the validation loss of the current batch
                    val_acc += torch.mean(correct_pred.type(torch.FloatTensor))     # Add the validation accuracy of the current batch, ignoring all padding values
                    if on_gpu is True:
                        # Move data to CPU for performance computations
                        scores, labels = scores.cpu(), labels.cpu()
                    # Add the training ROC AUC of the current batch
                    if model.n_outputs == 1:
                        try:
                            val_auc.append(roc_auc_score(labels.numpy(), scores.detach().numpy()))
                        except Exception as e:
                            warnings.warn(f'Couldn\'t calculate the validation AUC on step {step}. Received exception "{str(e)}".')
                    else:
                        # It might happen that not all labels are present in the current batch;
                        # as such, we must focus on the ones that appear in the batch
                        labels_in_batch = labels.unique().long()
                        try:
                            val_auc.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                         multi_class='ovr', average='macro', labels=labels_in_batch.numpy()))
                            # Also calculate a weighted version of the AUC; important for imbalanced dataset
                            val_auc_wgt.append(roc_auc_score(labels.numpy(), softmax(scores[:, labels_in_batch], dim=1).detach().numpy(),
                                                             multi_class='ovr', average='weighted', labels=labels_in_batch.numpy()))
                        except Exception as e:
                            warnings.warn(f'Couldn\'t calculate the validation AUC on step {step}. Received exception "{str(e)}".')
                    # Remove the current features and labels from memory
                    del features
                    del labels

            # Calculate the average of the metrics over the batches
            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            val_auc = np.mean(val_auc)
            if model.n_outputs > 1:
                val_auc_wgt = np.mean(val_auc_wgt)

            # Display validation loss
            if step % print_every == 0:
                print(f'Epoch {epoch} step {step}: Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
            # Check if the performance obtained in the validation set is the best so far (lowest loss value)
            if val_loss < val_loss_min:
                print(f'New minimum validation loss: {val_loss_min} -> {val_loss}.')
                # Update the minimum validation loss
                val_loss_min = val_loss
                # Get the current day and time to attach to the saved model's name
                current_datetime = datetime.now().strftime('%d_%m_%Y_%H_%M')
                # Filename and path where the model will be saved
                model_filename = f'{model_name}_{val_loss:.4f}valloss_{current_datetime}.pth'
                print(f'Saving model as {model_filename}')
                # Save the best performing model so far, along with additional information to implement it
                checkpoint = hyper_params
                checkpoint['state_dict'] = model.state_dict()
                torch.save(checkpoint, f'{models_path}{model_filename}')
                if log_comet_ml is True and comet_ml_save_model is True:
                    # Upload the model to Comet.ml
                    experiment.log_asset(file_data=model_filename, overwrite=True)
        # except Exception as e:
        #     warnings.warn(f'There was a problem doing training epoch {epoch}. Ending current epoch. Original exception message: "{str(e)}"')
        # try:
        # Calculate the average of the metrics over the epoch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        train_auc = np.mean(train_auc)
        if model.n_outputs > 1:
            train_auc_wgt = np.mean(train_auc_wgt)
        # Remove attached gradients so as to be able to print the values
        train_loss, val_loss = train_loss.detach(), val_loss.detach()
        if on_gpu is True:
            # Move metrics data to CPU
            train_loss, val_loss = train_loss.cpu(), val_loss.cpu()

        if log_comet_ml is True:
            # Log metrics to Comet.ml
            experiment.log_metric('train_loss', train_loss, step=epoch)
            experiment.log_metric('train_acc', train_acc, step=epoch)
            experiment.log_metric('train_auc', train_auc, step=epoch)
            experiment.log_metric('val_loss', val_loss, step=epoch)
            experiment.log_metric('val_acc', val_acc, step=epoch)
            experiment.log_metric('val_auc', val_auc, step=epoch)
            experiment.log_metric('epoch', epoch)
            experiment.log_epoch_end(epoch, step=step)
            if model.n_outputs > 1:
                experiment.log_metric('train_auc_wgt', train_auc_wgt, step=epoch)
                experiment.log_metric('val_auc_wgt', val_auc_wgt, step=epoch)
        # Print a report of the epoch
        print(f'Epoch {epoch}: Training loss: {train_loss}; Training Accuracy: {train_acc}; Training AUC: {train_auc}; \
                Validation loss: {val_loss}; Validation Accuracy: {val_acc}; Validation AUC: {val_auc}')
        print('----------------------')
        # except Exception as e:
        #     warnings.warn(f'There was a problem printing metrics from epoch {epoch}. Original exception message: "{str(e)}"')

    try:
        if model_filename is not None:
            # Load the model with the best validation performance
            model = load_checkpoint(f'{models_path}{model_filename}', ModelClass)
        if do_test is True:
            # Run inference on the test data
            model_inference(model, dataloader=test_dataloader, dataset=dataset,
                            model_type=model_type, is_custom=is_custom,
                            seq_len_dict=seq_len_dict, padding_value=padding_value,
                            experiment=experiment, cols_to_remove=cols_to_remove,
                            already_embedded=already_embedded,
                            see_progress=see_progress)
    except UnboundLocalError:
        warnings.warn('Inference failed due to non existent saved models. Skipping evaluation on test set.')
    except Exception as e:
        warnings.warn(f'Inference failed due to "{str(e)}". Skipping evaluation on test set.')

    if log_comet_ml is True:
        # Only report that the experiment completed successfully if it finished the training without errors
        experiment.log_other("completed", True)

    if get_val_loss_min is True:
        # Also return the minimum validation loss alongside the corresponding model
        return model, val_loss_min.item()
    else:
        return model
