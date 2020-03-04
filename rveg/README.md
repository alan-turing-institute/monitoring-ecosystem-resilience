## R code for generating and analyzing vegetation patterns.

The code in this directory comprises the *rveg* package, which contains functions for generating vegetation patterns (optionally evolving them from an input starting pattern), and performing a network centrality analysis on them.

To use, fro this directory, load the package:
```
devtools::load_all()
```

### Generating a pattern:
```
pattern <-rveg::generatePattern()
```

### Calculate Euler Characteristic for a pattern

```
featureVec <- calc_EC(pattern)
```
where "pattern" is a 2D array of 1s and 0s (with 1 representing vegetation and 0 representing bare soil).


### Running tests

From this directory, do
```
Rscript -e "devtools::test()"
```