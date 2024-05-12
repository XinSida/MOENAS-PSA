import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
import matplotlib.pyplot as plt

def parse_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    evaluations = []
    for line in lines:
        if line.startswith('accuracy'):
            accuracy = float(line.split(':')[1].strip('%\n')) / 100.0  # Convert percentage to decimal
        elif line.startswith('cost'):
            cost = int(line.split(':')[1])
            # Store 1 - accuracy and cost
            evaluations.append((1 - accuracy, cost))
    return np.array(evaluations)

def find_pareto_front(evaluations):
    # Perform non-dominated sorting
    nds = NonDominatedSorting()
    fronts = nds.do(evaluations, only_non_dominated_front=True)

    pareto_front = evaluations[fronts]

    return pareto_front

def plot_pareto_front(pareto_front, filename):
    plt.figure(figsize=(10, 6))
    if pareto_front.ndim == 1:
        pareto_front = np.array([pareto_front])  
    plt.scatter(pareto_front[:, 1], pareto_front[:, 0], color='blue', label='Pareto Front')
    plt.scatter(1000, 0.1059, color='red', marker='o', label='Tian') 
    plt.title('Pareto Front of Cost vs. Loss k=1 seed=10 FE=800')
    plt.xlabel('Cost')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, format='pdf') 
    plt.show()
    
def save_pareto_front_to_file(pareto_front, file_path):
    with open(file_path, 'w') as file:
        if pareto_front.ndim == 1:
            pareto_front = np.array([pareto_front])
        for point in pareto_front:
            file.write(f"({point[1]}, {point[0]})\n")

file_path = 'function_evaluation.txt'
evaluations = parse_file(file_path)

pareto_front = find_pareto_front(evaluations)
plot_pareto_front(pareto_front, 'pareto_front.pdf')

output_file_path = 'pareto_front.txt'  
save_pareto_front_to_file(pareto_front, output_file_path)
