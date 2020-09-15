![Data Utils diagram](https://raw.githubusercontent.com/AndreCNF/data-utils/master/docs/images/DataUtilsMap.png)

---

**Data Utils** represents set of generic, useful data science and machine learning methods.

## Install

There are two ways to install Data Utils:

* Clone the repository **(Recommended)**

  In order to install this repository with all the right packages, you should follow this steps:

  1. Clone through git

  ```
  git clone https://github.com/AndreCNF/data-utils.git
  ```

  2. Install the requirements

  If you have [Poetry](https://python-poetry.org/) installed, which is recommended, you can just enter the directory where the code has been cloned to and run the following command, which will create a new virtual environment and install all the requirements:

  ```
  poetry install
  ```

  Otherwise, you can just run:

  ```
  pip install -r requirements.txt
  ```

* Install through pip

Using pip, you can install with the following command:
`pip install -e git+https://github.com/AndreCNF/data-utils.git`

## Overview

Data pipelines can be seen as modular tasks, where sometimes even seemingly unrelatable problems share common traits. And so, Data Utils emerges as a toolbox for generic, useful data science and machine learning methods. It is divided on multiple modules, each one addressing a different type of tasks. These can be seen as separate core parts of a data science pipeline, ranging from the usual data preprocessing to training neural networks. While it mostly relies on well known best practises, such as normalization, model versioning and hyperparameter tuning, there are also some less common intuitions builtin.

## Embedding

![Embedding pipeline](https://raw.githubusercontent.com/AndreCNF/data-utils/master/docs/images/EmbeddingPipeline.png)

There is an embedding pipeline method implemented, which relies on a pre-existing PyTorch function: [embedding bag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html). It is essentially an embedding layer but with an averaging operation on top, in case we have multiple categories to embed. It is however optimized for unidimensional sequences. So, we first one hot encode the categorical features then, for those that we want to embed, the embedding pipeline multiplies each one hot encoded column with its index, counting only the columns that originated from the same categorical feature. This way, we can feed sequences of keys to the embedding bag, which it can encode and return the average of embeddings, row by row. Inside Data Utils, the code then handles all the intermediate steps required to integrate these lists of embeddings into the data and remove the former one hot encoded columns.

## Deep learning training and inference

![Training and inference](https://raw.githubusercontent.com/AndreCNF/data-utils/master/docs/images/TrainingInferencePipelines.png)

The training and inference pipelines have a modular structure, so that we can adapt to different contexts, different models and, not less important, have a cleaner code. From a smaller scale to a larger one, we start off with inference blocks, which are methods that, for a particular model type (e.g. RNNs or MLPs), take a batch of data, feedforward it through the model, calculate the loss and, if in a training environment, update the model, then, beyond some extra intermediate steps that some models might require (such as removing paddings), return the predictions, loss and raw scores. Going up in the scale, we have the full inference pipeline, that takes some data, runs the appropriate inference block on it and retrieves both the output and the performance metrics. Then, we have the training pipeline, which forcibly goes through all the training set and validation set, with the option of doing the inference pipeline on the test set in the end. During training, it can upload all the context, from hyperparameters to metrics and model, to [Comet ML](https://www.comet.ml/).

## Handling large data

![SavingLoadingMethods](https://raw.githubusercontent.com/AndreCNF/data-utils/master/docs/images/)

When we have data that is too big to process as a whole, a solution that might come to mind right away is to use it in smaller chunks. By treating the data bits by bits, we can reduce the minimum requirements and speed up the operations. So, Data Utils has methods for saving and loading the data in multiple [feather](https://arrow.apache.org/docs/python/feather.html) files, so as to facilitate these processes on large datasets.

![LargeDatasetClass](https://raw.githubusercontent.com/AndreCNF/data-utils/master/docs/images/)

Furthermore, in the Datasets component, I have developed a generic `Large_Dataset` class. It inherits from PyTorch's `Dataset` class but it is designed to dynamically load the files from disk, instead of from RAM as my other dataset classes do. Beyond this definition, `Large_Dataset` makes no assumptions about the data and lets the user define a preprocessing pipeline, which runs before returning each batch's features and labels, and an initial analysis method, which is an optional procedure where it can collect more information on the data. If needed, the user can also add more arguments when creating the dataset object, which are then integrated as attributes and can be accessed within the analysis and preprocessing steps.

## Paper and citation

For more information on how Data Utils works and its motivation, you can check my [master's thesis](http://andrecnf.github.io/master-thesis).

If you want to include Data Utils on your citations, please use the following .bib file:

```
I will put here the citation when I have the paper published :)
```
