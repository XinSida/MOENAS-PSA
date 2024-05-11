# MOENAS-PSA
> Multi-Objective Evolutionary Neural Architecture Search for Liquid State Machine
## Overview
![](images/flowchart.png)
MOENAS-PSA comprises three steps. 
We first conducted a sensitivity analysis for each parameter and perform regression analysis. Then we used UCB algorithm to restrict the search space for each parameter. Finally, we applied the multi-objective ENAS method, along with a surrogate-assisted approach to optimize LSM architectures. 
MOENAS-PSA is able to reduce the evaluation cost through search space reduction while accomplishing multi-objective optimization of network architecture.
## Code Structure
