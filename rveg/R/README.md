## Animation of vegetation patterns

This is a translation of some `matlab` code to visualize a simple model of vegetation patterns in arid landscapes.
To run:
From *R* or *RStudio*
```
source('patterns.R')
animate
```

Configuration parameters are loaded from the file ```config.json```.


Dependencies:
```
dplyr
ggplot2
gganimate
jsonlite
```
These can be installed from CRAN via ```install.packages(<package_name>)```.
