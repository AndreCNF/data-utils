import colorlover as cl                                 # Get colors from colorscales
import dash_core_components as dcc                      # Dash components to be used in a dashboard
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
                     background_color='white', output_type='dash', dash_id='some_indicator',
                     dash_height='70%', show_number=False, show_delta=False,
                     ref_value=None):
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
    output_type : str, default 'dash'
        The format on which the output is presented. Available options are 'dash'
        and 'figure'.
    dash_id : str, default 'some_indicator'
        ID to be used in Dash.
    dash_height : str, default '70%'
        Height value to be used in the Dash graph.
    show_number : bool, default False
        If set to True, the number will be shown next to the plot.
    show_delta : bool, default False
        If set to True, the value's variation, based on a reference value, will
        be plotted.
    ref_value : int, default None
        Reference value to use in the delta visualization.

    Returns
    -------
    If output_type == 'figure':

    figure : dict
        Figure dictionary which can be used in Plotly.

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
    # Create the figure
    figure={
    'data': [
        dict(
            type='indicator',
            mode='gauge',
            gauge=dict(
                shape='bullet',
                bar=dict(
                    thickness=1,
                    color=color
                    ),
                axis=dict(range=[min_val, max_val])
                ),
            value=value
        )
    ],
    'layout': dict(
        paper_bgcolor=background_color,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        )
    }
    if output_type == 'figure':
        return figure
    elif output_type == 'dash':
        return dcc.Graph(id=dash_id,
            figure=figure,
            style=dict(height=dash_height)
            )
    else:
        raise Exception(f'ERROR: Invalid output type {output_type}. Only `figure` and `dash` are currently supported.')
