#!/bin/bash

# setting up environment
pip install --upgrade pip
pip install virtualenv
virtualenv --no-site-packages ./sentiment-analaysis-vgc

# install requirements
. ./sentiment-analaysis-vgc/bin/activate
pip install -r requirements.txt
python -m pip install jupyter

