import numpy as np
import pymoo
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import random

import matplotlib.pyplot as plt

import sys
import os
import _pickle as pickle
import pickle
from struct import unpack
import shutil
import time

import operator
from typing import Any

import subprocess

from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

global actual_evaluations
actual_evaluations = 0

tf.random.set_seed(1)
np.random.seed(1)

def evaluate_accuracy(x):
    global actual_evaluations

    print('___________________time begin________________________')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    old_time = time.time()

    print(x)

    Reservoir_neuronNum=[Reservoir0_neuronNum:=round(x[0]),Reservoir1_neuronNum:=round(x[1]),Reservoir2_neuronNum:=round(x[2])]
    Reservoir_excitatoryRatio=[Reservoir0_excitatoryRatio:=x[3],Reservoir1_excitatoryRatio:=x[4],Reservoir2_excitatoryRatio:=x[5]]
    Reservoir_E_E_ratio=[Reservoir0_E_E_ratio:=x[6],Reservoir1_E_E_ratio:=x[7],Reservoir2_E_E_ratio:=x[8]]
    Reservoir_E_I_ratio=[Reservoir0_E_I_ratio:=x[9],Reservoir1_E_I_ratio:=x[10],Reservoir2_E_I_ratio:=x[11]]
    Reservoir_I_E_ratio=[Reservoir0_I_E_ratio:=x[12],Reservoir1_I_E_ratio:=x[13],Reservoir2_I_E_ratio:=x[14]]
    Reservoir_I_I_ratio=[Reservoir0_I_I_ratio:=x[15],Reservoir1_I_I_ratio:=0,Reservoir2_I_I_ratio:=0]

    path_RC_defination = './MOENAS_PSA_k_1/LSM_defination'
    cell_defination_file = open('%s.pickle' % (path_RC_defination + '/cell_defination'), 'wb')

    layer1 = 3
    cell_defination_1 = [[layer1]]
    for i in range(layer1):               
        cell_defination_1.append([Reservoir_neuronNum[i],Reservoir_excitatoryRatio[i],Reservoir_E_E_ratio[i],Reservoir_E_I_ratio[i],Reservoir_I_E_ratio[i],Reservoir_I_I_ratio[i]])
    
    pickle.dump([1],cell_defination_file)
    pickle.dump(cell_defination_1, cell_defination_file)

    cell_defination_file.close()

    print('\n')
    print('cell_defination_file content:   ')
    with open('./MOENAS_PSA_k_1/LSM_defination/cell_defination.pickle', 'rb') as file:
        while True:
            try:
                obj = pickle.load(file)
                print(obj)
                print(',')
            except EOFError:
                break
    print('\n')

    interconnection_file = open('%s.pickle' % (path_RC_defination + '/interconnection'), 'wb')
    interconnection_1 = [[1,layer1],[0.1,0.0]]  # input to layer1[In2E,In2I]

    pickle.dump(interconnection_1, interconnection_file)

    # Parameter value range.
    OUT_sample_defination = np.array([[0.9, 0.0], [0.9, 0.0], [0.9, 0.0], [0.9, 0.0], [0.9, 0.0]])
    
    interconnection_file.close() 

    os.system('./MOENAS_PSA_k_1/LSM_MOENAS/talent_run.sh' + ' ' + '6000')

    ##-----------------------------------------------------------------------------------------
    ## main function
    ##-----------------------------------------------------------------------------------------
    num_example = int(sys.argv[1])
    using_resting = int(sys.argv[2])
    working_time = int(sys.argv[3])
    resting_time = int(sys.argv[4])

    OUT_population = 10 #The numbers 0 through 9, totaling ten digits.

    path_spike_record = './MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/result_training'
    path_weight_learned = './MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/weight_learned'
    
    RCe_population_array = np.load(path_RC_defination + '/RCe_population_array.npy')
    RCi_population_array = np.load(path_RC_defination + '/RCi_population_array.npy')
    
    ##--------------LSM state process--------------------##
    
    RC_num = RCe_population_array.size + RCi_population_array.size
    connection_random_combine = np.array([], dtype = int)
    spike_normalized_combine = np.array([]) 
    spike_normalized_combine_working = np.array([]) 
    spike_normalized_combine_resting = np.array([]) 
    combine_number = 0
    RC_sampled_population = 0
    firing_rate_per_neuron_per_example = np.array([], dtype = float)
    error_during_training = np.array([], dtype = float)

    for i in range(RC_num):#10 0~9
        if OUT_sample_defination[i//2][i%2] != 0:
            if i % 2 == 0:
                RC_population = RCe_population_array[i // 2] 
            else:
                RC_population = RCi_population_array[i // 2] 
            RC_OUT_prob = OUT_sample_defination[i // 2, i % 2] 
            RC_sampled_population += RC_population  
            connection_random = np.zeros((OUT_population, RC_population), dtype = int)
            array_uniform = np.random.uniform(0, 1, (OUT_population, RC_population))
            connection_random[array_uniform < RC_OUT_prob] = 1 
            if combine_number == 0:
                connection_random_combine = connection_random
            else:
                connection_random_combine = np.append(connection_random_combine, connection_random, axis = 1)
            if connection_random_combine[0].size != RC_sampled_population:
                raise Exception('Connection random combination is wrong!')
            if i == 0:
                spike_index_file = path_spike_record + '/dynamic_array_spikemonitor_i_'
                spike_time_file = path_spike_record + '/dynamic_array_spikemonitor_t_'
            else:
                spike_index_file = path_spike_record + '/dynamic_array_spikemonitor_' + str(i) + '_i_'
                spike_time_file = path_spike_record + '/dynamic_array_spikemonitor_' + str(i) + '_t_' 
            spike_monitor_index = np.fromfile(spike_index_file, dtype = np.int32, count = -1, sep = "" )
            spike_monitor_time = np.fromfile(spike_time_file, dtype = np.float64, count = -1, sep = "") #second
            if len(spike_monitor_index) != len(spike_monitor_time):
                raise Exception('Spike records may not read properly!')
            
            np.append(firing_rate_per_neuron_per_example, float(len(spike_monitor_index))/RC_population/num_example)
            
            spike_counter = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_working = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_resting = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_one = np.zeros(RC_population, dtype = int)
            spike_counter_one_working = np.zeros(RC_population, dtype = int)
            spike_counter_one_resting = np.zeros(RC_population, dtype = int)
            working_time_per_example = working_time * 0.001  # second
            resting_time_per_example = resting_time * 0.001  # second

            time_per_example = working_time_per_example + resting_time_per_example
            passed_time_one = time_per_example 
            number_example = 0
            for j, item in enumerate(spike_monitor_time):
                if item > passed_time_one: 
                    number_example = int(item/time_per_example) - 1
                    spike_counter[number_example, :] = spike_counter_one
                    spike_counter_working[number_example, :] = spike_counter_one_working
                    spike_counter_resting[number_example, :] = spike_counter_one_resting
                    spike_counter_one = np.zeros(RC_population)
                    spike_counter_one_working = np.zeros(RC_population)
                    spike_counter_one_resting = np.zeros(RC_population)
                    passed_time_one += time_per_example
            
                spike_counter_one[spike_monitor_index[j]] += 1
                if item < (passed_time_one - resting_time_per_example):
                    spike_counter_one_working[spike_monitor_index[j]] += 1
                else:
                    spike_counter_one_resting[spike_monitor_index[j]] += 1
                
            number_example = num_example - 1 
            spike_counter[number_example, :] = spike_counter_one
            spike_counter_working[number_example, :] = spike_counter_one_working
            spike_counter_resting[number_example, :] = spike_counter_one_resting 

            spike_counter_max = spike_counter.max(1).reshape(num_example, 1) 
            spike_counter_max_working = spike_counter_working.max(1).reshape(num_example, 1)
            spike_counter_max_resting = spike_counter_resting.max(1).reshape(num_example, 1)

            spike_normalized = spike_counter * 1.0 / spike_counter_max 
            spike_normalized_working = spike_counter_working * 1.0 / spike_counter_max_working
            spike_normalized_resting = spike_counter_resting * 1.0 / spike_counter_max_resting
            
            spike_normalized[np.isnan(spike_normalized)] = 0.0
            spike_normalized_working[np.isnan(spike_normalized_working)] = 0.0
            spike_normalized_resting[np.isnan(spike_normalized_resting)] = 0.0
            if combine_number == 0:
                spike_normalized_combine = spike_normalized
                spike_normalized_combine_working = spike_normalized_working
                spike_normalized_combine_resting = spike_normalized_resting
            else:
                spike_normalized_combine = np.append(spike_normalized_combine, spike_normalized, axis = 1)
                spike_normalized_combine_working = np.append(spike_normalized_combine_working, spike_normalized_working, axis = 1)
                spike_normalized_combine_resting = np.append(spike_normalized_combine_resting, spike_normalized_resting, axis = 1)
            combine_number += 1

    iteration = 0
    connection_weight = np.random.uniform(0,1, (OUT_population, RC_sampled_population))
    if using_resting == 0:
        RC_state_vector = spike_normalized_combine
    elif using_resting == 1:
        RC_state_vector = spike_normalized_combine_working
    else:
        RC_state_vector = spike_normalized_combine_resting
    
    Y = np.load('./input_spike_record/train_labels_random_6000.npy')
    train_data = RC_state_vector
    train_data_rows, train_data_cols = train_data.shape
    Y = Y - 1
    train_labels = to_categorical(Y,10)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

    model = Sequential()
    model.add(Dense(500, input_dim=train_data_cols, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,#sgd,
                metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
        epochs=300,
        batch_size=128,
        verbose=0)

    ##---------------------------------------------------------------------------##
    ## testing ##
    ##---------------------------------------------------------------------------##

    num_example = 10000 
    path_spike_record = './MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/result_testing'
    path_weight_learned = './MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/weight_learned'
    
    ##--------------LSM state process--------------------##
    
    spike_normalized_combine = np.array([]) 
    spike_normalized_combine_working = np.array([]) 
    spike_normalized_combine_resting = np.array([]) 
    combine_number = 0
    RC_sampled_population = 0

    for i in range(RC_num):
        if OUT_sample_defination[i//2][i%2] != 0:
            if i % 2 == 0:
                RC_population = RCe_population_array[i // 2]
            else:
                RC_population = RCi_population_array[i // 2]
            RC_sampled_population += RC_population
            
            if i == 0:
                spike_index_file = path_spike_record + '/dynamic_array_spikemonitor_i_'
                spike_time_file = path_spike_record + '/dynamic_array_spikemonitor_t_'
            else:
                spike_index_file = path_spike_record + '/dynamic_array_spikemonitor_' + str(i) + '_i_'
                spike_time_file = path_spike_record + '/dynamic_array_spikemonitor_' + str(i) + '_t_'
            spike_monitor_index = np.fromfile(spike_index_file, dtype = np.int32, count = -1, sep = "" )
            spike_monitor_time = np.fromfile(spike_time_file, dtype = np.float64, count = -1, sep = "")
            if len(spike_monitor_index) != len(spike_monitor_time):
                raise Exception('Spike records may not read properly!')
            
            print('firing rate for per neuron in neuron group' + str(i) + ' : ' + str(float(len(spike_monitor_index))/RC_population/num_example) + '/(neuron * example)')

            spike_counter = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_working = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_resting = np.zeros((num_example, RC_population), dtype = int)
            spike_counter_one = np.zeros(RC_population, dtype = int)
            spike_counter_one_working = np.zeros(RC_population, dtype = int)
            spike_counter_one_resting = np.zeros(RC_population, dtype = int)
            working_time_per_example = working_time * 0.001  # second
            resting_time_per_example = resting_time * 0.001  # second
            time_per_example = working_time_per_example + resting_time_per_example
            passed_time_one = time_per_example
            number_example = 0
            for j, item in enumerate(spike_monitor_time):
                if item > passed_time_one:
                    number_example = int(item/time_per_example) - 1
                    spike_counter[number_example, :] = spike_counter_one 
                    spike_counter_working[number_example, :] = spike_counter_one_working 
                    spike_counter_resting[number_example, :] = spike_counter_one_resting
                    spike_counter_one = np.zeros(RC_population)
                    spike_counter_one_working = np.zeros(RC_population)
                    spike_counter_one_resting = np.zeros(RC_population)
                    passed_time_one += time_per_example
            
                spike_counter_one[spike_monitor_index[j]] += 1
                if item < (passed_time_one - resting_time_per_example):
                    spike_counter_one_working[spike_monitor_index[j]] += 1
                else:
                    spike_counter_one_resting[spike_monitor_index[j]] += 1
                
            number_example = num_example - 1 
            spike_counter[number_example, :] = spike_counter_one 
            spike_counter_working[number_example, :] = spike_counter_one_working
            spike_counter_resting[number_example, :] = spike_counter_one_resting 

            spike_counter_max = spike_counter.max(1).reshape(num_example, 1)
            spike_counter_max_working = spike_counter_working.max(1).reshape(num_example, 1)
            spike_counter_max_resting = spike_counter_resting.max(1).reshape(num_example, 1)

            spike_normalized = spike_counter * 1.0 / spike_counter_max
            spike_normalized_working = spike_counter_working * 1.0 / spike_counter_max_working
            spike_normalized_resting = spike_counter_resting * 1.0 / spike_counter_max_resting

            spike_normalized[np.isnan(spike_normalized)] = 0.0
            spike_normalized_working[np.isnan(spike_normalized_working)] = 0.0
            spike_normalized_resting[np.isnan(spike_normalized_resting)] = 0.0
            if combine_number == 0:
                spike_normalized_combine = spike_normalized
                spike_normalized_combine_working = spike_normalized_working
                spike_normalized_combine_resting = spike_normalized_resting
            else:
                spike_normalized_combine = np.append(spike_normalized_combine, spike_normalized, axis = 1)
                spike_normalized_combine_working = np.append(spike_normalized_combine_working, spike_normalized_working, axis = 1)
                spike_normalized_combine_resting = np.append(spike_normalized_combine_resting, spike_normalized_resting, axis = 1)
            combine_number += 1

    if using_resting == 0:
        RC_state_vector = spike_normalized_combine
    elif using_resting == 1:
        RC_state_vector = spike_normalized_combine_working
    else:
        RC_state_vector = spike_normalized_combine_resting
    
    Y = np.load('./input_spike_record/label_test_LSM_60000_70000.npy')
    Y = Y - 1
    test_labels = to_categorical(Y,10)
    test_data = RC_state_vector
                

    predictlabel = model.predict(test_data)
    score = model.evaluate(test_data, test_labels, batch_size=128)
    
    print('testing performance: ' + str(score[1]*100) + '%')

    print('__________________time end____________________')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    current_time = time.time()
    expensive = current_time - old_time

    with open("function_evaluation.txt", "a") as fw:
        fw.write('evaluation time:'+str(expensive//60)+'分钟'+'\n')
        fw.write('accuracy:'+str(score[1]*100)+'%'+'\n') 
        fw.write('cost:'+str(round(x[0])+round(x[1])+round(x[2]))+'\n')
        fw.write('actual_evaluations:'+str(actual_evaluations)+'\n') 
        fw.write(str(x)+'\n'+'\n')
        
    command_delete = (
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/output/results/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/output/static_arrays/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/result/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/random_connection/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/result_testing/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/result_training/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/weight_learned/* && "
        "rm -rf ./MOENAS_PSA_k_1/LSM_defination/*"
    )
    subprocess.run(command_delete, shell=True, check=True)

    return score[1]

surrogate_accuracy = SGDRegressor(max_iter=1000, tol=1e-3)
feature_scaler = StandardScaler()
is_surrogate_trained = False

def save_state(X, Y, current_index):
    with open('saved_state.pkl', 'wb') as f:
        pickle.dump({'X': X, 'Y': Y, 'current_index': current_index}, f)

def load_initial_state():
    try:
        with open('saved_state_100_16.pkl', 'rb') as f:
            data = pickle.load(f)
            return data['X'], data['Y'], data['current_index']
    except FileNotFoundError:
        return None, None, None

def load_state():
    try:
        with open('saved_state.pkl', 'rb') as f:
            data = pickle.load(f)
            return data['X'], data['Y'], data['current_index']
    except FileNotFoundError:
        return None, None, None

def get_initial_training_data(n_samples):
    """
    Randomly generate n_samples solution vectors under specified constraints.
    """
    global is_surrogate_trained

    X_loaded, Y_loaded, start_index = load_initial_state()

    if X_loaded is None:
        
        X_initial = np.zeros((n_samples, 16))# Preallocate an array to store solution vectors, with 16 variables because two monotonic parameters were excluded.

        global_bounds = np.array([
            [50.0, 500.0], [50.0, 500.0], [50.0, 500.0],[0.09,0.8999999999999999],[0.7649999999999999,0.855],[0.49499999999999994,0.8999999999999999],[0.0,0.7500000000000001],[0.0,0.15000000000000002],[0.0,0.25],[0.0,0.9],[0.25000000000000006,0.9],[0.15000000000000002,0.6500000000000001],[0.0,0.8500000000000001],[0.35000000000000003,0.6500000000000001],[0.15000000000000002,0.9],[0.0,0.9]
        ])

        discrete_intervals = {
            3: [(0.09, 0.22499999999999998), (0.22500000000000003, 0.40499999999999997), (0.6749999999999999, 0.765), (0.8549999999999999, 0.8999999999999999)],
            6: [(0.0, 0.05), (0.25000000000000006, 0.35000000000000003), (0.65, 0.7500000000000001)],
            9: [(0.0, 0.05), (0.35000000000000003, 0.55), (0.75, 0.9)],
            10: [(0.25000000000000006, 0.45), (0.65, 0.7500000000000001), (0.85, 0.9)],
            11: [(0.15000000000000002, 0.25), (0.25000000000000006, 0.6500000000000001)],
            12: [(0.0, 0.15000000000000002), (0.35000000000000003, 0.8500000000000001)],
            14: [(0.15000000000000002, 0.25), (0.25000000000000006, 0.7500000000000001), (0.85, 0.9)],
            15: [(0.0, 0.25), (0.25000000000000006, 0.35000000000000003), (0.45, 0.55), (0.75, 0.9)],
        }

        for i in range(16): 
            if i in discrete_intervals: 
                for j in range(n_samples):
                    interval = discrete_intervals[i][np.random.randint(len(discrete_intervals[i]))] 
                    X_initial[j, i] = np.random.uniform(interval[0], interval[1])  
            else:  
                X_initial[:, i] = np.random.uniform(global_bounds[i][0], global_bounds[i][1], size=n_samples) 

        Y_accuracy_initial = []
        start_index = 0
    else:
        X_initial = X_loaded
        Y_accuracy_initial = Y_loaded
    
    X_initial[:, :3] = np.round(X_initial[:, :3])

    for i in range(start_index, len(X_initial)):
        try:
            acc = evaluate_accuracy(X_initial[i])
            Y_accuracy_initial.append(acc)
            #save_state(X_initial, Y_accuracy_initial, i + 1)  # Since a trained model is already provided, we will not run it here.
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            break

    X_initial_scaled = feature_scaler.fit_transform(X_initial)
    
    surrogate_accuracy.partial_fit(X_initial_scaled, Y_accuracy_initial)
    is_surrogate_trained = True

    return X_initial, Y_accuracy_initial

def select_indices_for_real_eval(gen, n_samples, eval_frequency=10, eval_percentage=0.2):
    """
    Select a certain percentage of individuals for real evaluation at every eval_frequency generations:
    gen: current generation number
    n_samples: current population size
    eval_frequency: frequency of real evaluations
    eval_percentage: percentage of individuals selected for real evaluation each time
    """
    if gen % eval_frequency == 0:
        n_eval = int(np.ceil(eval_percentage * n_samples))
        return np.random.choice(n_samples, n_eval, replace=False)
    else:
        return np.array([])

X_initial, Y_accuracy_initial = get_initial_training_data(n_samples=100)

class PSANAS(Problem):
    def __init__(self):
        self.continuous_intervals = { # Zero-based indexing.
            5: [(0.585, 0.675), (0.8549999999999999, 0.8999999999999999)],
            6: [(0.0, 0.05), (0.25000000000000006, 0.35000000000000003)],
            12: [(0.0, 0.05), (0.35000000000000003, 0.55), (0.65, 0.7500000000000001)],
            14: [(0.15000000000000002, 0.25), (0.25000000000000006, 0.35000000000000003), (0.65, 0.7500000000000001), (0.85, 0.9)],
            15: [(0.05, 0.15000000000000002), (0.85, 0.9)],
        }

        total_intervals = sum(len(v) for v in self.continuous_intervals.values())

        # Define the problem dimensions: 16 variables, 2 objectives, and multiple constraints.
        super().__init__(n_var=16, n_obj=2, n_constr=total_intervals + 3, xl=np.array([50.0, 50.0, 50.0,0.8549999999999999, 0.7649999999999999, 0.585, 0.0, 0.0, 0.0, 0.75, 0.35000000000000003, 0.25000000000000006, 0.0, 0.35000000000000003, 0.15000000000000002, 0.05]), xu=np.array([1000.0, 1000.0, 1000,0.8999999999999999, 0.855, 0.8999999999999999, 0.35000000000000003, 0.05, 0.25, 0.8500000000000001, 0.45, 0.6500000000000001, 0.7500000000000001, 0.45, 0.9, 0.9]))

        self.gen = 0

        self.discrete_var_indices = [0, 1, 2]     

    def _evaluate(self, X, out, *args, **kwargs):
        global is_surrogate_trained, X_initial, Y_accuracy_initial, actual_evaluations

        X_round = np.copy(X)
        X_round[:, :3] = np.round(X_round[:, :3])

        F = np.zeros((X.shape[0], 2)) # Initialize outputs.
        g = np.zeros((X.shape[0], self.n_constr)) # Initialize the degree of constraint violation.

        X_scaled = feature_scaler.transform(X_round)

        indices_to_evaluate = select_indices_for_real_eval(self.gen, X.shape[0])
        
        for i in range(X.shape[0]):
            x_scaled = X_scaled[i, :]
            F[i, 1] = X_round[i][0] + X_round[i][1] + X_round[i][2]  # The second objective: Net Scale.
            
            if i in indices_to_evaluate or not is_surrogate_trained:
                actual_evaluations += 1 
                F[i, 0] = 1 - evaluate_accuracy(X_round[i])  # Convert evaluation results into a minimization problem.

                new_x = X_round[i].reshape(1, -1) 
                new_y = np.array([1 - F[i, 0]])

                new_x_scaled = feature_scaler.transform(new_x)

                if not is_surrogate_trained:
                    surrogate_accuracy.partial_fit(new_x_scaled, new_y)
                    is_surrogate_trained = True
                else:
                    surrogate_accuracy.partial_fit(new_x_scaled, new_y)

                if X_initial.size == 0: 
                    X_initial = new_x
                    Y_accuracy_initial = new_y
                else:
                    X_initial = np.vstack([X_initial, new_x])
                    Y_accuracy_initial = np.append(Y_accuracy_initial, new_y)
            else:
                F[i, 0] = 1 - surrogate_accuracy.predict(x_scaled.reshape(1, -1))[0]    
                
        # Calculate the constraint violation for discrete variables.
        for idx in self.discrete_var_indices:
            g[i, idx] = abs(X[i, idx] - X_round[i, idx])
        
        # Calculate the constraint violation for non-continuous intervals.
        interval_idx = len(self.discrete_var_indices)
        for var_idx, intervals in self.continuous_intervals.items():
            for j, (a, b) in enumerate(intervals):
                g[i, interval_idx] = max(0, a - X_round[i, var_idx]) + max(0, X_round[i, var_idx] - b)
                interval_idx += 1
        
        out["F"] = F
        out["G"] = g

        self.gen += 1

problem = PSANAS()

algorithm = NSGA2(pop_size=100)

result = minimize(problem,
                  algorithm,
                  termination=('n_gen', 500),
                  seed=10,
                  verbose=True)
