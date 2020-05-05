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


def indicator_plot(value, min_val=0, max_val=100, type='bullet', higher_is_better=True,
                   background_color='white', output_type='plotly', dash_id='some_indicator',
                   dash_height=None, dash_width=None, show_number=True, show_delta=False,
                   ref_value=None, font_family='Roboto', font_size=14,
                   font_color='black', prefix='', suffix='', showticklabels=False):
    '''Generate an indicator plot, which can help visualize performance. Can
    either be of type bullet or gauge.

    Parameters
    ----------
    value : int
        Value which will be plotted on the graph.
    type : str, default 'bullet'
        Type of indicator plot. Can either be 'bullet' or 'gauge'.
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
    dash_height : str, default None
        Height value to be used in the Dash graph.
    dash_width : str, default None
        Width value to be used in the Dash graph.
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
    showticklabels : bool, default False
        Determines whether or not the tick labels are drawn.

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
    if type.lower() == 'bullet':
        shape = 'bullet'
    elif type.lower() == 'gauge':
        shape = 'angular'
    else:
        raise Exception(f'ERROR: Invalid indicator plot type inserted. Expected "bullet" or "gauge", received "{type}".')
    # Create the figure
    figure=dict(
        data=[dict(
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
                shape=shape,
                bar=dict(
                    thickness=1,
                    color=color
                ),
                axis=dict(
                    range=[min_val, max_val],
                    showticklabels=showticklabels
                )
            ),
            delta=dict(reference=ref_value)
        )],
        layout=dict(
            paper_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            )
        )
    )
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        style = dict()
        if dash_height is not None:
            style['height'] = dash_height
        if dash_width is not None:
            style['width'] = dash_width
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=style,
            config=dict(displayModeBar=False)
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')


def shap_summary_plot(shap_values, feature_names, max_display=10,
                      background_color='white', marker_color='blue',
                      output_type='plotly', dash_id='some_shap_summary_plot',
                      dash_height=None, dash_width=None, font_family='Roboto',
                      font_size=14, font_color='black',
                      xaxis_title='mean(|SHAP value|) (average impact on model output magnitude)'):
    '''Plot the overall feature importance, based on SHAP values, through an
    horizontal bar plot.

    Parameters
    ----------
    shap_values : numpy.ndarray or list
        Array or list containing all the SHAP values which we want to plot.
    feature_names : list
        List with the names of the features that the SHAP values refer to.
    max_display : int
        The maximum number of features to plot.
    background_color : str, default 'white'
        The plot's background color. Can be set in color name (e.g. 'white'),
        hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    marker_color : str, default 'blue'
        The color of the bars in the plot. Can be set in color name (e.g. 'white'),
        hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    output_type : str, default 'plotly'
        The format on which the output is presented. Available options are
        'dash', `plotly` and 'figure'.
    dash_id : str, default 'some_shap_summary_plot'
        ID to be used in Dash.
    dash_height : str, default None
        Height value to be used in the Dash graph.
    dash_width : str, default None
        Width value to be used in the Dash graph.
    font_family : str, default 'Roboto'
        Text font family to be used in the numbers shown next to the graph.
    font_size : int, default 14
        Text font size to be used in the numbers shown next to the graph.
    font_color : str, default 'black'
        Text font color to be used in the numbers shown next to the graph. Can
        be set in color name (e.g. 'white'), hexadecimal code (e.g. '#555') or
        GB (e.g. 'rgb(0,0,255)').
    xaxis_title : str, default 'mean(|SHAP value|) (average impact on model output magnitude)'
        Phrase that appears bellow the X axis.

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
    if max_display is not None:
        # Only show the `max_display` most impactful features
        sorted_idx = sorted_idx[-max_display:]
    sorted_mean_abs_shap = mean_abs_shap[sorted_idx]
    sorted_feature_names = [feature_names[idx] for idx in sorted_idx]
    # Create the figure
    figure=dict(
        data=[dict(
            type='bar',
            x=sorted_mean_abs_shap,
            y=sorted_feature_names,
            orientation='h',
            marker=dict(color=marker_color)
        )],
        layout=dict(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            xaxis_title=xaxis_title,
            xaxis=dict(showgrid=False),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            )
        )
    )
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        style = dict()
        if dash_height is not None:
            style['height'] = dash_height
        if dash_width is not None:
            style['width'] = dash_width
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=style
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')


def shap_waterfall_plot(expected_value, shap_values, features, feature_names,
                        max_display=10, background_color='white',
                        line_color='gray', increasing_color='red',
                        decreasing_color='blue', output_type='plotly',
                        dash_id='some_shap_summary_plot', dash_height=None,
                        dash_width=None, font_family='Roboto', font_size=14,
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
    line_color : str, default 'gray'
        The waterfall plot's connector color. Can be set in color name
        (e.g. 'white'), hexadecimal code (e.g. '#555') or RGB (e.g. 'rgb(0,0,255)').
    increasing_color : str, default 'red'
        Color of the waterfall bars that indicate an increasing value.
    decreasing_color : str, default 'blue'
        Color of the waterfall bars that indicate a decreasing value.
    output_type : str, default 'plotly'
        The format on which the output is presented. Available options are
        'dash', `plotly` and 'figure'.
    dash_id : str, default 'some_shap_summary_plot'
        ID to be used in Dash.
    dash_height : str, default None
        Height value to be used in the Dash graph.
    dash_width : str, default None
        Width value to be used in the Dash graph.
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
    # Get the model's output on the current sample
    model_output = shap_values.sum() + expected_value
    # Sort the SHAP values and the feature names
    sorted_idx = np.argsort(np.abs(shap_values))
    sorted_shap_values = shap_values[sorted_idx]
    sorted_features = features[sorted_idx]
    sorted_feature_names = [feature_names[idx] for idx in sorted_idx]
    if max_display is None:
        max_display = len(sorted_feature_names)
    if max_display < len(sorted_feature_names):
        # Isolate the data regarding the less relevant features that will be aggregated
        other_features_shap_values = sorted_shap_values[:-(max_display-1)]
        other_features_names = sorted_feature_names[:-(max_display-1)]
        n_other_features = len(other_features_names)
        # Get the sum of the less relevant features' SHAP values
        other_features_shap_values = other_features_shap_values.sum()
        # Remove the detailed info of the less relevant features
        sorted_shap_values = sorted_shap_values[-(max_display-1):]
        sorted_feature_names = sorted_feature_names[-(max_display-1):]
        sorted_features = sorted_features[-(max_display-1):]
        # Add the less relevant features aggregate data into the plot
        hovertext = [f'{feature}={val}' if du.utils.is_integer(val) else f'{feature}={val:.2e}'
                     for (feature, val) in zip(sorted_feature_names, sorted_features)]
        hovertext = ['Multiple features'] + hovertext
        sorted_shap_values = np.insert(sorted_shap_values, 0, other_features_shap_values)
        sorted_feature_names = [f'{n_other_features} other features'] + sorted_feature_names
    else:
        hovertext = [f'{feature}={val}' if du.utils.is_integer(val) else f'{feature}={val:.2e}'
                     for (feature, val) in zip(sorted_feature_names, sorted_features)]
    # Create the figure
    # [TODO] Fix the xaxis positioning to center on the expected value
    figure=dict(
        data=[dict(
            type='waterfall',
            x=sorted_shap_values,
            y=sorted_feature_names,
            base=expected_value,
            hovertext=hovertext,
            orientation='h',
            connector=dict(line=dict(color=line_color)),
            increasing=dict(marker=dict(color=increasing_color)),
            decreasing=dict(marker=dict(color=decreasing_color))
        )],
        layout=dict(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            xaxis_title='Output value',
            xaxis=dict(showgrid=False, autorange=True, rangemode='normal'),
            font=dict(
                family=font_family,
                size=font_size,
                color=font_color
            ),
            annotations=[
                # Expected value indicator
                dict(
                    # x-reference is assigned to the x-values
                    xref='x',
                    # y-reference is assigned to the plot paper [0,1]
                    yref='paper',
                    x=expected_value,
                    y=0,
                    text=f'E[f(X)]={expected_value:.2e}',
                    showarrow=True,
                    arrowhead=0,
                    ax=0,
                    ay=30
                ),
                # Output value indicator
                dict(
                    # x-reference is assigned to the x-values
                    xref='x',
                    # y-reference is assigned to the plot paper [0,1]
                    yref='paper',
                    x=model_output,
                    y=1,
                    text=f'f(x)={model_output:.2e}',
                    showarrow=True,
                    arrowhead=0,
                    ax=0,
                    ay=10
                )
            ],
            shapes=[
                # Expected value line
                dict(
                    type='line',
                    # x-reference is assigned to the x-values
                    xref='x',
                    # y-reference is assigned to the plot paper [0,1]
                    yref='paper',
                    x0=expected_value,
                    y0=0,
                    x1=expected_value,
                    y1=1,
                    fillcolor=line_color,
                    line=dict(
                        color=line_color,
                        width=1
                    ),
                    opacity=0.5,
                    # NOTE: I can't put this line bellow the traces, otherwise
                    # the waterfall conectors will create a blank line in the middle of it
#                     layer='below'
                ),
                # Output value line
                dict(
                    type='line',
                    # x-reference is assigned to the x-values
                    xref='x',
                    # y-reference is assigned to the plot paper [0,1]
                    yref='paper',
                    x0=model_output,
                    y0=0,
                    x1=model_output,
                    y1=1,
                    fillcolor=line_color,
                    line=dict(
                        color=line_color,
                        width=1
                    ),
                    opacity=0.5,
                    layer='below'
                )
            ]
        )
    )
    if output_type == 'figure':
        return figure
    elif output_type == 'plotly':
        return go.Figure(figure)
    elif output_type == 'dash':
        style = dict()
        if dash_height is not None:
            style['height'] = dash_height
        if dash_width is not None:
            style['width'] = dash_width
        return dcc.Graph(
            id=dash_id,
            figure=figure,
            style=style
        )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure`, `plotly` and `dash` are currently supported.')
