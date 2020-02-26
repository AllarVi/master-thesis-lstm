# Recurrent Neural Networks

Python version 3.7.2.

Code mostly inspired by this notebook: https://github.com/WillKoehrsen/recurrent-neural-networks/blob/master/notebooks/Deep%20Dive%20into%20Recurrent%20Neural%20Networks.ipynb

## Create virtual environment

python3 -m venv venv3

source venv3/bin/activate

## Install dependencies

pip install -r requirements.txt

## Command line

cd src

PYTHONPATH=../. python3 main.py

## Tips from the Deep Learning Cookbook

* Start with a small batch size, 32 for starters (can be increased if using GPU)