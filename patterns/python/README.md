# Pattern simulation function

In order to run this simulation you need to have [anaconda](https://www.anaconda.com/distribution/) installed. 
Create an anaconda environment using the environment.yml file in this directory. Concretely:

    conda update conda
    conda env create -f environment.yml  
    source activate patterns
    
If you don't have anaconda, then the best thing to do is to go into check the imports at the top the patterns.py file.
Install them, into whatever environment you're using. There aren't many dependencies so this shouldn't take long.

The config.py file has the initial parameters for running the simulations, you are free to change them .

To run the simulation just do:

    python patterns.py