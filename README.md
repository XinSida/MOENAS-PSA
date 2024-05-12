# MOENAS-PSA
> Multi-Objective Evolutionary Neural Architecture Search for Liquid State Machine
## Overview
![](images/flowchart.png)
MOENAS-PSA comprises three steps. 
We first conducted a sensitivity analysis for each parameter and perform regression analysis. Then we used UCB algorithm to restrict the search space for each parameter. Finally, we applied the multi-objective ENAS method, along with a surrogate-assisted approach to optimize LSM architectures. 
MOENAS-PSA is able to reduce the evaluation cost through search space reduction while accomplishing multi-objective optimization of network architecture.
## Code Structure
.

├── input_spike_record -- N-MNIST dataset

├── LSM_MOENAS.sh -- run the experiment in batch

├── MOENAS_PSA_k_1 -- LSM simulation

├── MOENAS.py -- main execution file

├── visualization.py -- result visualization

└── saved_state_100_16.pkl -- surrogate model
## Requirements
- Operating system: tested in Ubuntu 20.04
- Python 3.9.0
- pymoo 0.6.1.1
- TensorFlow 2.15.0
## Getting Started
Run the script file with bash, for example:
```
{./LSM_MOENAS.sh}
```


