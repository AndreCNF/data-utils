import colorlover as cl                                 # Get colors from colorscales
import plotly.graph_objs as go                          # Plotly for interactive and pretty plots
import dash_core_components as dcc                      # Dash components to be used in a dashboard
import numpy as np                                      # NumPy to handle numeric and NaN operations
import data_utils as du

# Pandas to handle the data in dataframes
if du.use_modin is True:
    import modin.pandas as pd
else:
    import pandas as pd

# Colors to use in performance or quality plots
perf_colors = cl.scales['8']['div']['RdYlGn']

# Methods

def set_bar_color(values, ids, seq_len, threshold=0,
                  neg_color='rgba(30,136,229,1)', pos_color='rgba(255,13,87,1)'):
    '''Determine each bar's color in a bar chart, according to the values being
    plotted and the predefined threshold.

    Parameters
    ----------
    values : numpy.Array
        Array containing the values to be plotted.
    ids : int or list of ints
        ID or list of ID's that select which time series / sequences to use in
        the color selection.
    seq_len : int or list of ints
        Single or multiple sequence lengths, which represent the true, unpadded
        size of the input sequences.
    threshold : int or float, default 0
        Value to use as a threshold in the plot's color selection. In other
        words, values that exceed this threshold will have one color while the
        remaining have a different one, as specified in the parameters.
    pos_color : string
        Color to use in the bars corresponding to threshold exceeding values.
    neg_color : string
        Color to use in the bars corresponding to values bellow the threshold.

    Returns
    -------
    colors : list of strings
        Resulting bar colors list.
    '''
    if type(ids) is list:
        # Create a list of lists, with the colors for each sequences' instances
        return [[pos_color if val > 0 else neg_color for val in values[id, :seq_len]]
                for id in ids]
    else:
        # Create a single list, with the colors for the sequence's instances
        return [pos_color if val > 0 else neg_color for val in values[ids, :seq_len]]


def bullet_indicator(value, min_val=0, max_val=100, higher_is_better=True,
                     background_color='white', output_type='plotly', dash_id='some_indicator',
                     dash_height='70%', show_number=True, show_delta=False,
                     ref_value=None, font_family='Roboto', font_size=14,
                     font_color='black', prefix='', suffix=''):
    '''Generate a bullet indicator plot, which can help visualize.

    Parameters
    ----------
    value : int
        Value which will be plotted on the graph.
    min_val : int, default 0
        Minimum value in the range of numbers that the input can assume.
    max_val : int, default 100
        Maximum value in the range of numbers that the input can assume.
    higher_is_better : bool, default True
        If set to True, values closer to the maximum will be represented in green,
        while those closer to the minimum will be in red. Vice versa for False.
    background_color : str, default 'white'
        The plot's background color. Can be set in color name (e.g. 'white'),
        hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    output_type : str, default 'plotly'
        The format on which the output is presented. Available options are
        'dash', `plotly` and 'figure'.
    dash_id : str, default 'some_indicator'
        ID to be used in Dash.
    dash_height : str, default '70%'
        Height value to be used in the Dash graph.
    show_number : bool, default True
        If set to True, the number will be shown next to the plot.
    show_delta : bool, default False
        If set to True, the value's variation, based on a reference value, will
        be plotted.
    ref_value : int, default None
        Reference value to use in the delta visualization.
    font_family : str, default 'Roboto'
        Text font family to be used in the numbers shown next to the graph.
    font_size : int, default 14
        Text font size to be used in the numbers shown next to the graph.
    font_color : str, default 'black'
        Text font color to be used in the numbers shown next to the graph. Can
        be set in color name (e.g. 'white'), hexadecimal code (e.g. '#555') or
        GB (e.g. 'rgb(0,0,255)').
    prefix : str, default ''
        Text to be appended to the beginning of the number, shown next to the
        indicator graph. e.g. '%', '€', 'km', 'g'
    suffix : str, default ''
        Text to be appended to the end of the number, shown next to the
        indicator graph. e.g. '%', '€', 'km', 'g'

    Returns
    -------
    If output_type == 'figure':

    figure : dict
        Figure dictionary which can be used in Plotly.

    Else if output_type == 'plotly':

    figure : plotly.graph_objs._figure.Figure
        Figure in a plotly figure object, which can be displayed in a notebook.

    Else if output_type == 'dash':

    figure : dcc.Graph
        Figure in a Dash graph format, ready to be used in a dashboard.
    '''
    global perf_colors
    # Define the bar color
    if higher_is_better:
        color = perf_colors[int(max((value/max_val)*len(perf_colors)-1, 0))]
    else:
        perf_colors[len(perf_colors)-1-int(max((value/max_val)*len(perf_colors)-1, 0))]
    # Define if the value and the delta is shown next to the plot
    if show_number and show_delta:
        mode='number+gauge+delta'
    elif show_number and not show_delta:
        mode='number+gauge'
    elif not show_number and show_delta:
        mode='gauge+delta'
    else:
        mode='gauge'
    # Create the figure
    figure={
        'data': [dict(
                type='indicator',
                mode=mode,
                value=value,
                number=dict(
                    font=dict(
                        family=font_family,
                        size=font_size,
                        color=font_color
                    ),
                    prefix=prefix,
                    suffix=suffix
                ),
                gauge=dict(
                    shape='bullet',
                    bar=dict(
                        thickness=1,
                        color=color
                    ),
                    axis=dict(range=[min_val, max_val])
                ),
                delta=dict(reference=ref_value)
        )],
        'layout': dict(
            paper_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            )
        )
    }
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=dict(height=dash_height),
            config=dict(displayModeBar=False)
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')


def shap_summary_plot(shap_values, feature_names, max_display=10,
                      background_color='white', output_type='plotly',
                      dash_id='some_shap_summary_plot', dash_height='70%',
                      font_family='Roboto', font_size=14, font_color='black'):
    '''Plot the overall feature importance, based on SHAP values, through an
    horizontal bar plot.

    Parameters
    ----------
    shap_values : numpy.ndarray or list
        Array or list containing all the SHAP values which we want to plot.
    feature_names : list
        List with the names of the features that the SHAP values refer to.
    max_display : str
        The maximum number of features to plot.
    background_color : str, default 'white'
        The plot's background color. Can be set in color name (e.g. 'white'),
        hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    output_type : str, default 'plotly'
        The format on which the output is presented. Available options are
        'dash', `plotly` and 'figure'.
    dash_id : str, default 'some_shap_summary_plot'
        ID to be used in Dash.
    dash_height : str, default '70%'
        Height value to be used in the Dash graph.
    font_family : str, default 'Roboto'
        Text font family to be used in the numbers shown next to the graph.
    font_size : int, default 14
        Text font size to be used in the numbers shown next to the graph.
    font_color : str, default 'black'
        Text font color to be used in the numbers shown next to the graph. Can
        be set in color name (e.g. 'white'), hexadecimal code (e.g. '#555') or
        GB (e.g. 'rgb(0,0,255)').

    Returns
    -------
    If output_type == 'figure':

    figure : dict
        Figure dictionary which can be used in Plotly.

    Else if output_type == 'plotly':

    figure : plotly.graph_objs._figure.Figure
        Figure in a plotly figure object, which can be displayed in a notebook.

    Else if output_type == 'dash':

    figure : dcc.Graph
        Figure in a Dash graph format, ready to be used in a dashboard.
    '''
    # Calculate the mean absolute value of each feature's SHAP values
    mean_abs_shap = np.mean(np.abs(shap_values).reshape(-1, shap_values.shape[-1]), axis=0)
    # Sort the SHAP values and the feature names
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_mean_abs_shap = mean_abs_shap[sorted_idx]
    sorted_feature_names = [feature_names[idx] for idx in sorted_idx]
    # Create the figure
    # [TODO] Implement max_display
    figure={
        'data': [dict(
            type='bar',
            x=sorted_mean_abs_shap,
            y=sorted_feature_names,
            orientation='h'
        )],
        'layout': dict(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)',
            xaxis=dict(showgrid=False),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            )
        )
    }
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=dict(height=dash_height)
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')


def shap_waterfall_plot(expected_value, shap_values, features, feature_names,
                        max_display=10, background_color='white',
                        output_type='plotly', dash_id='some_shap_summary_plot',
                        dash_height='70%', font_family='Roboto', font_size=14,
                        font_color='black'):
    '''Do a waterfall plot on a single sample, based on SHAP values, showing
    each feature's contribution to the corresponding output.

    Parameters
    ----------
    expected_value : float
        This is the reference value that the feature contributions start from.
        For SHAP values it should be the value of explainer.expected_value.
    shap_values : numpy.array
        One dimensional array of SHAP values.
    features : numpy.array
        One dimensional array of feature values. This provides the values of all
        the features, and should be the same shape as the shap_values argument.
    feature_names : list
        List of feature names (# features).
    max_display : str
        The maximum number of features to plot.
    background_color : str, default 'white'
        The plot's background color. Can be set in color name (e.g. 'white'),
        hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    output_type : str, default 'plotly'
        The format on which the output is presented. Available options are
        'dash', `plotly` and 'figure'.
    dash_id : str, default 'some_shap_summary_plot'
        ID to be used in Dash.
    dash_height : str, default '70%'
        Height value to be used in the Dash graph.
    font_family : str, default 'Roboto'
        Text font family to be used in the numbers shown next to the graph.
    font_size : int, default 14
        Text font size to be used in the numbers shown next to the graph.
    font_color : str, default 'black'
        Text font color to be used in the numbers shown next to the graph. Can
        be set in color name (e.g. 'white'), hexadecimal code (e.g. '#555') or
        GB (e.g. 'rgb(0,0,255)').

    Returns
    -------
    If output_type == 'figure':

    figure : dict
        Figure dictionary which can be used in Plotly.

    Else if output_type == 'plotly':

    figure : plotly.graph_objs._figure.Figure
        Figure in a plotly figure object, which can be displayed in a notebook.

    Else if output_type == 'dash':

    figure : dcc.Graph
        Figure in a Dash graph format, ready to be used in a dashboard.
    '''
    if len(shap_values.shape) > 1:
        raise Exception(f'ERROR: Received multiple samples, with input shape {shap_values.shape}. The waterfall plot only handles individual samples, i.e. one-dimensional inputs.')
    # Sort the SHAP values and the feature names
    sorted_idx = np.argsort(np.abs(shap_values))
    sorted_shap_values = shap_values[sorted_idx]
    sorted_features = features[sorted_idx]
    sorted_feature_names = [feature_names[idx] for idx in sorted_idx]
    # Create the figure
    # [TODO] Fix the xaxis positioning to center on the expected value
    # [TODO] Add markers for the expected value and the output value
    # [TODO] Use SHAP's colors
    # [TODO] Implement max_display
    figure={
        'data': [dict(
            type='waterfall',
#             offset=expected_value,
#             measure='relative',
            x=sorted_shap_values,
            y=sorted_feature_names,
            base=expected_value,
            hovertext=[f'{feature}={val:.2e}' for (feature, val) in zip(sorted_feature_names, features)],
            orientation='h'
        )],
        'layout': dict(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            xaxis_title='Output value',
            xaxis=dict(showgrid=False, autorange=True, rangemode='normal'),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            )
        )
    }
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=dict(height=dash_height)
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')
