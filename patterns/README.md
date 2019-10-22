# Modelling vegetation patterns in arid landscapes

The code in this repository originates from some Matlab code by Stefan Dekker, Willem Bouten, Maarten Boerlijst and Max Rietkerk
(included in the "matlab" directory), implementing the scheme described in:

Rietkerk et al. 2002. Self-organization of vegetation in arid ecosystems. The American Naturalist 160(4): 524-530.

The python and R directories contain the same code translated into those languages.

## python
In order to run this simulation you need to have [anaconda](https://www.anaconda.com/distribution/) installed. 
Create an anaconda environment using the environment.yml file in this directory. Concretely:
```
    conda update conda
    conda env create -f environment.yml  
    source activate patterns
 ```   
If you don't have anaconda, then the best thing to do is to go into check the imports at the top the patterns.py file.
Install them, into whatever environment you're using. There aren't many dependencies so this shouldn't take long.

The config.py file has the initial parameters for running the simulations, you are free to change them .

To run the simulation just do:
```
python patterns.py
```
## R
Dependencies:
```
dplyr
ggplot2
gganimate
jsonlite
```
These can be installed from CRAN via ```install.packages(<package_name>)```.

To run:
From *R* or *RStudio*
```
source('patterns.R')
animate
```

Configuration parameters are loaded from the file ```config.json```.
