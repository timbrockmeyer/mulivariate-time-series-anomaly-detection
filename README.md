# Multivariate Time Series Anomaly Detection with GNNs and Latent Graph Inference
Implementation of different graph neural network (GNN) based models for anomaly detection in multivariate timeseries in sensor networks. <br/>
An explicit graph structure modelling the interrelations between sensors is inferred during training and used for time series forecasting. Anomaly detection is based on the error between the predicted and actual values at each time step.

## Installation
### Requirements
* Python == 3.7
* cuda >= 10.2
* [pytorch==1.8.1] (https://pytorch.org/)
* [torch-geometric==1.7.2] (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) <br/>

Additional package files for torch-geometric (python 3.7 & pytorch 1.8.1) provided in '/whl/' as they are unavailable in artifactory as of writing. <br/>
Refer to https://pytorch-geometric.com/whl/ for other versions.
### Install python
Install python environment for example with conda:
```
  conda create -n py37 python=3.7
```
### Install packages
Run install bash script with either cpu or cuda flag depeneding on the indended use.
```
  # run after installing python
  bash install.sh cpu
  
  # or
  bash install.sh cuda
```

## Models
The repository contains several models. GNN-LSTM is used by default and achieved best performance.

### GNN-LSTM
Model with GNN feature expansion before multi-layer LSTM. A single node embedding is used to infer the latent graph through vector similary, 
and as node positional embeddings added to the GNN features before they are passed to the recurrent network.

### Convolutional-Attention Sequence-To-Sequence
Spatial-Temporal Convolution GNN with attention. Data is split into an encoder and decoder. Encoder creates a feature representation for each time step while the decoder creates a single representation. Encoder-Decoder attention is concatenated with the decoder output before passed to the prediction layer.
Uses multiple embedding layers to parameterize the latent graph diretly by the network. <br/>
Inspired by: https://arxiv.org/pdf/1705.03122.pdf.

### MTGNN
Sptial-Temporal Convolution GNN with attention and graph mix-hop propagation. <br/>
Taken from: https://arxiv.org/pdf/2005.11650.pdf.

### LSTM
Vanilla multi-layer LSTM used for benchmarking. 


## Data
### SWaT, WADI, Demo
Test dataset ('demo') included in the model folder. <br/>
SWaT and WADI datasets can be requested from [iTrust](https://itrust.sutd.edu.sg/). <br/>
The files should be opened in e.g. Excel to remove the first empty rows and save as a .csv file.<br/>
The CSV files should be placed in a folder with the same name ('swat' or 'wadi') in '/datasets/files/raw/\<name>/\<file>' <br/>

### Other
Additional datasets can either be loaded directly from CSV file using the dataset 'from_csv' <br/>
or by creating a custom dataset following the examples found in the '/datasets/' folder. <br/>
If 'from_csv' is used, the data should come in the same format as the demo data included in this repository,
with individual time series for each sensor represented by a single column. (Only) the test data should have
anomaly labels included in the last column. <br/>
The first column is assumed to be the timestamp. The files are to be placed in '/datasets/files/raw/from_csv/'.
If this option is chosen, data normalization is not available. Any preprocessing should be done manually.

## Usage

### Bash Script
Suitable parameters for the SWaT, Wadi, and Demo datasets can be found in the bash scripts,
which is the most convenient way to run models.
```
  # run from terminal
  sh run.sh [dataset] 
```

*Examples:*
```
  # example 1
  sh run.sh swat
  
  # example 2
  sh run.sh wadi
  
  # example 3
  sh run.sh demo
```

### Python File
Run the *main.py* script from your terminal (bash, powershell, etc). <br/>
To change the default model and training hyperparameters, flags can be included. <br/> 
Alternatively, those parameters can be changed within the file (argsparser default values).
```
  # run from terminal
  python main.py -[flags] 
```
*Examples:*
```
  # example 1
  python main.py -dataset demo -batch_size 4 -epochs 10 
  
  # example 2
  python main.py -dataset swat -epochs 10 -topk 20 -embed_dim 128 
  
  # example 3
  python main.py -dataset from_csv
```

**Available flags:** <br/>
`-dataset` The dataset. <br/>
`-window_size` Number of historical timesteps used in each sample. <br/>
`-horizon` Number of prediction steps. <br/>
`-val_split` Amount of data used for the validation dataset. Value between 0 and 1. <br/>
`-transform` Sampling transform applied to the model input data (e.g. median).  <br/>
`-target_transform` Sampling transform applied to the model target values. (e.g. median, max). <br/>
`-normalize` Boolean value if data normalization should be applied. <br/> 
`-shuffle_train` Boolean value if training data should be shuffled. <br/>
`-batch_size` Number of samples in each batch. <br/>

`-embed_dim` Number of node embedding dimensions (Disabled for GNN-LSTM). <br/>
`-topk` Number of allowed neighbors for each node. <br/>

`-smoothing` Error smoothing kernel size. <br/>
`-smoothing_method` Error smoothing kernel type (*mean* or *exp*). <br/>
`-thresholding` Thresholding method (*mean*, *max*, *best* (best performs an exhaustive search for theoretical performance evaluation)). <br/>

`-epochs` Number of training epochs. <br/>
`-early_stopping` Patience parameter of number of epochs without improvement for early stopping. <br/>
`-lr` Learning rate. <br/>
`-betas` Adam optimizer parameter. <br/>
`-weight_decay` Adam optimizer weight regularization parameter. <br/>
`-device` Computing device (*cpu* or *cuda*). <br/>

`-log_graph` Boolean for logging of learned graphs. <br/>

## Results
### Logs
After the initial run, a '/runs/' folder will be automatically created. <br/>
A copy of the model state dict, a loss plot, plots for the learned graph representation <br/>
and some additional information will be saved for each run of the model.

### Example Plots

Visualization of a t-SNE embedding of the learned undirected graph representation for the SWaT dataset <br/> 
with 15 neighbors per node. <br/>
<img src="https://github.airbus.corp/storage/user/7806/files/0d9dd800-38bf-11ec-8cdb-6c9ad12cb363" width="65%" height="65%">

Plot of a directly parameterized uni-directional graph adjaceny matrix with a single neighbor per node. <br/>
<img src="https://github.airbus.corp/storage/user/7806/files/c7fdbc80-7549-11ec-8a2f-a1469ed14624" width="65%" height="65%">

Node colors and labels indicate type of sensor. 


**P:** Pump <br/>
**MV:** Motorized valve <br/>
**UV:** Dechlorinator <br/>
**LIT:** Level in tank <br/>
**PIT:** Pressure in tank <br/>
**FIT:** Flow in tank <br/>
**AIT:** Analyzer in tank (different chemical analyzers; NaCl, HCl, ORP meters, etc) <br/>
**DPIT:** Differential pressure indicating transmitter <br/>
