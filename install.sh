#!/bin/bash

DIR=$1

# torch
pip install torch==1.8.1
pip install torch-geometric==1.7.2

# extra torch geometric packages
pip install --no-index torch-scatter -f whl/$DIR/
pip install --no-index torch-sparse -f whl/$DIR/
pip install --no-index torch-cluster -f whl/$DIR/ 
pip install --no-index torch-spline-conv -f whl/$DIR/ 

# extra packages
pip install matplotlib==3.4.3
pip install networkx==2.6.2
pip install scikit-learn==0.24.2
pip install scipy==1.7.1
pip install seaborn==0.11.2
pip install numpy==1.21.2
pip install pandas==1.3.2
pip install pyvis==0.1.9