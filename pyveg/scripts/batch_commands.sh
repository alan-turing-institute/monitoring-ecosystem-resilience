#!/bin/bash

# run a single pyveg Module, with config supplied in argument $1,
# on a Linux (Ubuntu-18.04) VM.

# this script should be called via
#   /bin/bash batch_commands.sh <task_config_json> <azure_config.py>


# install some packages needed by opencv
sudo apt-get update; sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# install and setup miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
conda create -n pyvegenv -y python=3.7
conda init bash
source ~/.bashrc
conda activate pyvegenv

# clone repo
git clone https://github.com/alan-turing-institute/monitoring-ecosystem-resilience
# copy task config file to working directory
mv $1 monitoring-ecosystem-resilience
# copy azure config file to working directory
mv $2 monitoring-ecosystem-resilience/pyveg
# change to working directory and checkout branch
cd monitoring-ecosystem-resilience
git checkout develop
# install python code to local site-packages
python -m pip install .
# run the pyveg_run_module entrypoint
pyveg_run_module --config_file $1
