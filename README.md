# Tetris NC2021 group project (group 15)

## Run instructions
The baseline experiments can be run by executing `optimizedGA.py`, `baselineGA.py` and `EA_NES_script.py` for the optimized Genetic algorithm, the baseline genetic algorithm and the Evolutionary strategy respectively. For other experiments make sure you comment out the desired experiment. 

All experiments are parallelized using the concurrent.futures library and ran using Ubuntu 20.04. Using windows the concurrent.futures is not installed by default and can cause errors. 

Results are placed in the `*_results` folders. These can be plotted using the `plot*.py` scripts. Make sure to give a proper filename filter as argument and an experiment number/ID.

## Credits
Credits to alexandrinaw, we used their Tetris implementation as starting point for our project https://github.com/alexandrinaw/tetris
