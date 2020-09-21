# # Outlier detection experiments
# ---
#
# Testing outlier detection algorithms on solar and metereologival time series data from the [University of Oregon](http://solardata.uoregon.edu/SelectArchival.html).
#
# * Structure of data: http://solardata.uoregon.edu/ArchivalFiles.html
# * Column codes: http://solardata.uoregon.edu/DataElementNumbers.html
# * Quality control flags: http://solardata.uoregon.edu/QualityControlFlags.html
# * Tested outlier detection algorithms: https://www.notion.so/andrecnf/Outlier-detection-06ba3f49b8114cc2a5a5bc6336c913ab

# ## Importing the necessary packages

import dask.dataframe as dd                # Dask to handle big data in dataframes
import pandas as pd                        # Pandas to load the data initially
from dask.distributed import Client        # Dask scheduler
import plotly                              # Plotly for interactive and pretty plots
import plotly.graph_objs as go
import os                                  # os handles directory/workspace changes
import numpy as np                         # NumPy to handle numeric and NaN operations
from tqdm import tqdm_notebook             # tqdm allows to track code execution progress
import glob                                # Read multiple files
import datetime                            # Filter dates and times

# **Important:** Use the following two lines to be able to do plotly plots offline:

import plotly.offline as py
plotly.offline.init_notebook_mode(connected=True)

# Allow pandas to show more columns:

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

# Change to parent directory
os.chdir("..")
import utils                               # Contains auxiliary functions

# Set the random seed for reproducibility:

utils.set_random_seed(0)

# Import the remaining custom packages:

import search_explore                      # Methods to search and explore data
import data_processing                     # Data processing and dataframe operations
# import embedding                           # Embedding and encoding related methods
# import padding                             # Padding and variable sequence length related methods
# import machine_learning                    # Common and generic machine learning related methods
# import deep_learning                       # Common and generic deep learning related methods

# Debugging packages
import pixiedust                           # Debugging in Jupyter Notebook cells

# +
# Change to parent directory (presumably "Documents")
os.chdir("../../..")

# Path to the CSV dataset files
data_path = 'Documents/Datasets/Outlier_Detection/UniversityOfOregon_SolarAndMeteorologicalData_Eugene/'
project_path = 'Documents/GitHub/eICU-mortality-prediction/'
# -

# Set up local cluster
client = Client('tcp://127.0.0.1:61980')
client

# Upload the utils.py file, so that the Dask cluster has access to relevant auxiliary functions
client.upload_file(f'{project_path}NeuralNetwork.py')
client.upload_file(f'{project_path}utils.py')
client.upload_file(f'{project_path}search_explore.py')
client.upload_file(f'{project_path}data_processing.py')

# **Problem:** Somehow, all works fine if I initialize the Dask client without specifying the tcp address. But if I specify the one obtained from the Jupyter Lab Dask extension, it returns "ModuleNotFoundError: No module named 'torch'"! Perhaps the Jupyter Lab Dask extension is associated to a different Python environment.
#
# **Solution:** Jupyter Lab must be started from within the desired virtual environment's shell.

client.run(os.getcwd)

# ## Loading data

all_files = glob.glob(f'{data_path}/*.txt')

# +
files_list = []

for filename in all_files:
    df = dd.read_csv(filename, header=0, sep='\t')
    files_list.append(df)

uoreg_df = dd.concat(files_list)
# -

uoreg_df.head()

uoreg_df.tail()

uoreg_df.npartitions

# ## Organizing data

# ### Filtering on ambient temperature
#
# There are a lot of features in this dataset, from energy production, wind, temperature and other weather events. We'll focus just on temperature, for simplicity and because it's both a familiar context and a generally smooth signal (should make it easier to detect outliers).

len(uoreg_df.columns)

column_names = list(uoreg_df.columns)
column_names

qlt_ctrl_flags = [col for col in column_names if col[0] == '0']
qlt_ctrl_flags

[uoreg_df[col].unique().compute() for col in qlt_ctrl_flags]

uoreg_df['0.20'].value_counts().compute()

uoreg_df['0.21'].value_counts().compute()

# For the downloaded dataset, there are no data points marked as problematic (with quality control flag indicating `13`). However, at least the ambient temperature features (`9300` and `9303`) have their associated quality control flags (`0.20` and `0.21`) indicating mostly raw data (quality control flag indicating `11`) and also some possible missing data (quality control flag indicating `99`).

# Select desired columns (time, ambient temperature and respective quality control flags):

uoreg_df = uoreg_df[['94255', '2019', '9300', '0.20', '9303', '0.21']]
uoreg_df.head()

# ### Renaming columns

# Rename the two temporal columns:

uoreg_df = uoreg_df.rename(columns={'94255': 'day_of_year', '2019': 'time_of_day', 
                                    '9300': 'ambient_temperature_1', '0.20': 'qlt_ctrl_flag_1',
                                    '9303': 'ambient_temperature_2', '0.21': 'qlt_ctrl_flag_2'})
uoreg_df.head()

# ### Reorganizing the temporal columns
#
# Merge the day and time columns for ease of use.

x = '2340'
y = '115'

x[-2:]

x[:-2]

f'{x[:-2]}:{x[-2:]}'


def separate_hours_n_minutes(x):
    x = str(x)
    if len(x) == 1:
        return f'00:0{x}'
    elif len(x) == 2:
        return f'00:{x}'
    else:
        return f'{x[:-2]}:{x[-2:]}'


separate_hours_n_minutes(x)

separate_hours_n_minutes(y)

uoreg_df.time_of_day = uoreg_df.time_of_day.apply(separate_hours_n_minutes)
uoreg_df.time_of_day.head()


def replace_24(x):
    x = x.replace('24:', '00:')
    return x


uoreg_df.time_of_day = uoreg_df.time_of_day.apply(replace_24)
uoreg_df.time_of_day.head()

uoreg_df.day_of_year.min().compute()


def pad_days(x):
    x = str(x)
    if len(x) == 1:
        return f'00{x}'
    elif len(x) == 2:
        return f'0{x}'
    else:
        return x


uoreg_df.day_of_year = uoreg_df.day_of_year.apply(pad_days)
uoreg_df.day_of_year.head()

uoreg_df['ts'] = uoreg_df.apply(lambda df: '2019' + ':' + df['day_of_year'] + ':' + df['time_of_day'], axis=1)
uoreg_df.head()

uoreg_df['ts'] = dd.to_datetime(uoreg_df['ts'], format='%Y:%j:%H:%M')
uoreg_df.head()

# Remove the now redundant `day_of_year` and `time_of_day` columns:

uoreg_df = uoreg_df.drop(['day_of_year', 'time_of_day'], axis=1)
uoreg_df.head()

# Sort by the timestamp:

uoreg_df = uoreg_df.set_index('ts')
uoreg_df.head()

uoreg_df.tail()

uoreg_df.visualize()

# Save current dataframe in memory to avoid accumulating several operations on the dask graph
uoreg_df = client.persist(uoreg_df)

uoreg_df.visualize()

# ## Exploring ambient temperature data
#
# Plotting the ambient temperature data to get an overview of it, try to visually identify possible outliers and check the points highlighted from each quality control flag value

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1)]
layout = go.Layout(title='Ambient temperature 1')
fig = go.FigureWidget(data, layout)
fig

uoreg_df.qlt_ctrl_flag_1.unique().compute()

len(uoreg_df)

uoreg_df_raw = uoreg_df[uoreg_df.qlt_ctrl_flag_1 == 11]
uoreg_df_bad = uoreg_df[uoreg_df.qlt_ctrl_flag_1 == 99]

len(uoreg_df_raw) + len(uoreg_df_bad)

len(uoreg_df_raw)

len(uoreg_df_bad)

uoreg_df_raw.head()

uoreg_df_bad.head()

# +
# data = [go.Scatter(x = uoreg_df_raw.index.compute(), y = uoreg_df_raw.ambient_temperature_1,
#                    name='raw data',
#                    marker=dict(color='blue')),
#         go.Scatter(x = uoreg_df_bad.index.compute(), y = uoreg_df_bad.ambient_temperature_1,
#                    name='bad data',
#                    marker=dict(color='red'))]
# layout = go.Layout(title='Ambient temperature 1')
# fig = go.Figure(data, layout)
# py.iplot(fig)
# -

# The above plot is straightforward and adds a legend indicating what each color means. However, as raw and bad data points are treated as separate data, the lines formed in the plot are misleading and unintuitive. It's better to go through a different approach, if we want to keep the lines, or just to use the markers, as done in the next cell:

data = [go.Scatter(x=uoreg_df_raw.index.compute(), y=uoreg_df_raw.ambient_temperature_1.compute(),
                   name='raw data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df_bad.index.compute(), y=uoreg_df_bad.ambient_temperature_1.compute(),
                   name='bad data',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1')
fig = go.FigureWidget(data, layout)
fig

# +
# colors = [1 if val == 99 else 0 for val in uoreg_df.qlt_ctrl_flag_1.compute()]

# +
# colors

# +
# data = [go.Scatter(x = uoreg_df.index.compute(), y = uoreg_df.ambient_temperature_1,
#                    marker=dict(color=colors,
#                                colorscale=[[0, 'blue'], [1, 'red']],
#                                cmax=1,
#                                cmin=0))]
# layout = go.Layout(title='Ambient temperature 1')
# fig = go.Figure(data, layout)
# py.iplot(fig)

# +
# data = [go.Scatter(x = uoreg_df.index.compute(), y = uoreg_df.ambient_temperature_1,
#                    marker=dict(color=colors,
#                                colorscale=[[0, 'blue'], [1, 'red']],
#                                cmax=1,
#                                cmin=0),
#                    mode='markers')]
# layout = go.Layout(title='Ambient temperature 1')
# fig = go.Figure(data, layout)
# py.iplot(fig)
# -

# The previous cell shows not only that picking colors point by point requires not using lines, which removes any advantage over separating data in different traces/plots, it also seems to be slower.

# ## Testing the outlier detection algorithms

# ### Good ol' thresholds
#
# Documentation: https://www.notion.so/andrecnf/Good-ol-thresholds-4a5786d7078a41d5806719e5ecc17068

# #### Absolute value thresholds

thresh_outliers = data_processing.threshold_outlier_detect(uoreg_df.ambient_temperature_1, max_thrs=35, min_thrs=-10)
thresh_outliers.head()

thresh_outliers.value_counts().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[thresh_outliers].index.compute(), y=uoreg_df[thresh_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with absolute value threshold outlier detection!

# #### Mean value thresholds

thresh_outliers = data_processing.threshold_outlier_detect(uoreg_df.ambient_temperature_1, max_thrs=4, min_thrs=0.25, threshold_type='mean')
thresh_outliers.head()

thresh_outliers.value_counts().compute()

uoreg_df.ambient_temperature_1.mean().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[thresh_outliers].index.compute(), y=uoreg_df[thresh_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with mean value threshold outlier detection!

# #### Median value thresholds

thresh_outliers = data_processing.threshold_outlier_detect(uoreg_df.ambient_temperature_1, max_thrs=4, min_thrs=0.25, threshold_type='median')
thresh_outliers.head()

thresh_outliers.value_counts().compute()

uoreg_df.ambient_temperature_1.compute().median()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[thresh_outliers].index.compute(), y=uoreg_df[thresh_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with median value threshold outlier detection!

# #### Standard deviation value thresholds

thresh_outliers = data_processing.threshold_outlier_detect(uoreg_df.ambient_temperature_1, max_thrs=1.5, min_thrs=-1.5, threshold_type='std')
thresh_outliers.head()

thresh_outliers.value_counts().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[thresh_outliers].index.compute(), y=uoreg_df[thresh_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Let's visualize the normalized ambient temperature for comparison:

norm_amb_tmp = (uoreg_df.ambient_temperature_1 - uoreg_df.ambient_temperature_1.mean()) / \
                uoreg_df.ambient_temperature_1.std()
norm_amb_tmp.head()

data = [go.Scatter(x=norm_amb_tmp.index.compute(), y=norm_amb_tmp.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=norm_amb_tmp[thresh_outliers].index.compute(), y=norm_amb_tmp[thresh_outliers].compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with standard deviation value threshold outlier detection!

# #### Absolute derivative thresholds

thresh_outliers = data_processing.threshold_outlier_detect(uoreg_df.ambient_temperature_1, max_thrs=4, min_thrs=-4, signal_type='derivative', time_scale='minutes')
thresh_outliers.head()

thresh_outliers.value_counts().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[thresh_outliers].index.compute(), y=uoreg_df[thresh_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Let's plot the derivative for comparison:

drvt_amb_tmp = uoreg_df.ambient_temperature_1.diff()
drvt_amb_tmp = drvt_amb_tmp / data_processing.signal_idx_derivative(drvt_amb_tmp, time_scale='minutes')
drvt_amb_tmp.head()

data = [go.Scatter(x=drvt_amb_tmp.index.compute(), y=drvt_amb_tmp.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=drvt_amb_tmp[thresh_outliers].index.compute(), y=drvt_amb_tmp[thresh_outliers].compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with absolute derivative threshold outlier detection!

# ### Jungle slopes
#
# Documentation: https://www.notion.so/andrecnf/Jungle-slopes-e9907da9a5bb415e8698e55d9a85407f

slopes_outliers = data_processing.slopes_outlier_detect(uoreg_df.ambient_temperature_1, time_scale='minutes')
slopes_outliers.head()

slopes_outliers.value_counts().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[slopes_outliers].index.compute(), y=uoreg_df[slopes_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Experiment only considering as outliers the data points that have large derivatives on both directions:

slopes_outliers = data_processing.slopes_outlier_detect(uoreg_df.ambient_temperature_1, time_scale='minutes', only_bir=True)
slopes_outliers.head()

slopes_outliers.value_counts().compute()

data = [go.Scatter(x=uoreg_df.index.compute(), y=uoreg_df.ambient_temperature_1.compute(),
                   name='data',
                   marker=dict(color='blue')),
        go.Scatter(x=uoreg_df[slopes_outliers].index.compute(), y=uoreg_df[slopes_outliers].ambient_temperature_1.compute(),
                   name='outliers',
                   mode='markers',
                   marker=dict(color='red'))]
layout = go.Layout(title='Ambient temperature 1 - With threshold outliers')
fig = go.FigureWidget(data, layout)
fig

# Everything seems to be working fine with absolute value threshold outlier detection!

# ### Rolling MAD
#
# Documentation: https://www.notion.so/andrecnf/Rolling-MAD-5e37b70dab8d4030846c3a4d4ca78afb


