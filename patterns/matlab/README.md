## Matlab code for calculating subgraph centrality / Euler Characteristic

The files ```mao_pollen_EC.m``` and ```mao_pollen.m``` are from Luke Mander and can be used to calculate the Euler Characteristic and count the connected components in subgraphs respectively.

If you don't have a Matlab license, these scripts can also be run in *Octave*.
On OSX you can do
```
brew install octave
```
then launch with ```octave`` and from the octave command prompt:
```
octave:1> pkg install -forge io
octave:2> pkg install -forge statistics
octave:3> pkg load statistics
octave:4> h = readcsv("../binary_image.txt")
...
```
