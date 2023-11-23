 
 
## Overview

This folder includes all python codes to reproduce the numerical results in the paper "Sparse Fr\'{e}chet Sufficient Dimension Reduction with Graphical Structure Among Predictors".

## Description of each file

* **EX1-EX2.py** : produces results for Example 1 and 2.
* **EX3-EX4** : produces results for Example 3 and 4.
* **compute_time.ipynb** : compute computation time for each algorithm.
* **BikeRental.py**: produces the results in the real data analysis and Figure 1. 
* **day.csv and hour.csv**: daily and hourly bike rental data.

## Executing steps

Executing the following steps will generate all the tables used in the manuscript. 

1. To run **EX1-EX2.py**, change parameters

    a). seedid = 1-100: for 100 simulations

    b). EX=1 for example 1 (Table 1) or EX=2 for example 2 (Table 2)

    c). covstruc = 1 or 2: produce Table 3

    d). neigh = False or True: produce Table 5

2. To run **EX3-EX4.py**, change parameters

    a). seedid = 1-100: for 100 simulations

    b). EX=3 for example 3 or EX=4 for example 2

    c). covstruc = 1 or 2
    
