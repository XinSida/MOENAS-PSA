##-----------------------------------------------------------------------------------------
## module import
##-----------------------------------------------------------------------------------------
import brian2 as b2
from brian2 import *

import numpy as np

import _pickle as pickle 

import os
import sys

##-----------------------------------------------------------------------------------------
## standalone code generation mode setup
##-----------------------------------------------------------------------------------------
'''
for multiple run, At the beginning of the script, i.e. after the import statements, add:
set_device('cpp_standalone', build_on_run=False)
After the last run() call, call device.build() explicitly:
device.build(directory='output', compile=True, run=True, debug=False)
'''
set_device('cpp_standalone', directory = './MOENAS_PSA_k_1/LSM_MOENAS/output', build_on_run = True, clean = True) 
## setup for C++ compilation preferences
codegen.cpp_compiler = 'gcc' #Compiler to use (uses default if empty);Should be gcc or msvc.
prefs.devices.cpp_standalone.extra_make_args_unix = ['-j']

##-----------------------------------------------------------------------------------------
## basic simulation setup
##-----------------------------------------------------------------------------------------
b2.defaultclock.dt = 0.2 * b2.ms
b2.core.default_float_dtype = float32
b2.core.default_integer_dtype = int32
##-----------------------------------------------------------------------------------------
## self-defined functions: mainly for the dataset input processing and output visualization
##-----------------------------------------------------------------------------------------
def Load_spike_record(path_spike_record, name_spike_record):
    """ 
       Load the generated spike train
    """
    spike_record_path_name = path_spike_record + name_spike_record 
    print(spike_record_path_name)
    if os.path.isfile('%s.pickle' % spike_record_path_name): 
        file_spike_record = open('%s.pickle' % spike_record_path_name)
        spike_record_index = pickle.load(file_spike_record)
        spike_record_time = pickle.load(file_spike_record)
        file_spike_record.close()
    else:
        raise Exception("Spike record hasn't been created or saved in the right path")
    return spike_record_index, spike_record_time 
def Random_connection_generator(num_src, num_tgt, prob, save_path, connection_name):
    random_uniform_array = np.random.rand(num_src, num_tgt)
    index_i = np.where(random_uniform_array < prob)[0]
    index_j = np.where(random_uniform_array < prob)[1]
    #weight_random = np.random.rand(index_i.size)
    weight_random = np.abs(0.4 * np.random.randn(index_i.size) + 0.5)
    np.save(save_path + 'index_i_' + connection_name, index_i)
    np.save(save_path + 'index_j_' + connection_name, index_j)
    np.save(save_path + 'weight_random_' + connection_name, weight_random)
    return index_i, index_j, weight_random
##-----------------------------------------------------------------------------------------
## simulation setting 
##-----------------------------------------------------------------------------------------
RC_defination_path = './MOENAS_PSA_k_1/LSM_defination'
cell_defination_file = open('%s.pickle' % (RC_defination_path + '/cell_defination'),'rb')
interconnection_file = open('%s.pickle' % (RC_defination_path + '/interconnection'),'rb')
num_RC_layer = int(pickle.load(cell_defination_file)[0])
print('###############')
print(num_RC_layer)

test_mode = bool(int(sys.argv[1])) 
num_example = int(sys.argv[2])
IN_population = int(sys.argv[3]) 
working_time = int(sys.argv[4])
resting_time = int(sys.argv[5])
if test_mode:
    using_test_dataset = True
else:
    using_test_dataset = False 
np.random.seed(0)
##-----------------------------------------------------------------------------------------
## file system
##-----------------------------------------------------------------------------------------
path_random_connection = './MOENAS_PSA_k_1/LSM_MOENAS/simulation_archive/random_connection/' 

##-----------------------------------------------------------------------------------------
## neuron and synapse dynamics definition
##-----------------------------------------------------------------------------------------
working_time = working_time * b2.ms
resting_time = resting_time * b2.ms

delay = {}
delay['excite_excite'] = (0 * b2.ms, 0 * b2.ms) 
delay['excite_inhibite'] = (0 * b2.ms, 0 * b2.ms) 
delay['inhibite_excite'] = (0 * b2.ms, 0 * b2.ms) 

## standard neuron parameters
v_rest_excite = -65.0 * b2.mV
v_rest_inhibite = -60.0 * b2.mV
v_reset_excite = -65.0 * b2.mV
v_reset_inhibite = -45.0 * b2.mV
v_thresh_excite = -52.0 * b2.mV
v_thresh_inhibite = -40.0 * b2.mV # -40
t_refrac_excite = 5.0 * b2.ms
t_refrac_inhibite = 2.0 * b2.ms
## spike conditon
v_thresh_excite_str = '(v > v_thresh_excite) and (timer > t_refrac_excite)'
v_thresh_inhibite_str = '(v > v_thresh_inhibite) and (timer > t_refrac_inhibite)'
## excitatory neuron membrane dynamics 
neuron_eqs_excite = '''
        dv/dt = ((v_rest_excite -v) + (I_excite + I_inhibite) ) / (100 * ms)  : volt (unless refractory)
        I_excite = ge * (-v)                           : volt
        I_inhibite = gi * (-100.0 * mV - v)            : volt
        dge/dt = -ge / (1.0 * ms)                           : 1
        dgi/dt = -gi / (2.0 * ms)                           : 1
        '''
neuron_eqs_inhibite = '''
        dv/dt = ((v_rest_inhibite -v) + (I_excite + I_inhibite) ) / (10 * ms)  : volt (unless refractory)
        I_excite = ge * (-v)                           : volt 
        I_inhibite = gi * (-85.0 * mV - v)             : volt
        dge/dt = -ge / (1.0 * ms)                           : 1
        dgi/dt = -gi / (2.0 * ms)                           : 1
        '''
## time
neuron_eqs_excite += '\n  dtimer/dt = 1  : second'
neuron_eqs_inhibite += '\n  dtimer/dt = 1  : second'
## reset dynamics for excitatory and inhibitory neuron
reset_excite = 'v = v_reset_excite; timer = 0 * ms; ge = 0; gi = 0;'
reset_inhibite = 'v = v_reset_inhibite; timer = 0 * ms; ge = 0; gi = 0' 
## synapse conductance model
model_synapse_base = 'w : 1'
spike_pre_excite = 'ge_post += w' 
spike_pre_inhibite = 'gi_post += w' 

##-----------------------------------------------------------------------------------------
## network structure definition
##-----------------------------------------------------------------------------------------
##-----------------------------------------------------------------------------------------
## definite and create the neuron and synapse group 
##-----------------------------------------------------------------------------------------
neuron_group = {}
synapse = {}
spike_monitor = {} 
## create input poisson neuron group
name_neuron_group = 'IN'

if using_test_dataset:
    input_spike_record_index = np.load('./input_spike_record/input_spike_test_index_60000_70000.npy')
    input_spike_record_time_ms = np.load('./input_spike_record/input_spike_test_time_ms_60000_70000.npy')
else:
    input_spike_record_index = np.load('./input_spike_record/train_index_random_6000_example.npy')
    input_spike_record_time_ms = np.load('./input_spike_record/train_time_random_6000_example.npy')

input_spike_record_time = input_spike_record_time_ms*0.001  #second

neuron_group[name_neuron_group] = SpikeGeneratorGroup(IN_population, input_spike_record_index, input_spike_record_time * b2.second) 

## create RC neuron group
RCe_population_list = []
RCi_population_list = []

for i in range(num_RC_layer):  
    cell_defination = pickle.load(cell_defination_file)
    if int(cell_defination[0][0]) != (len(cell_defination) - 1) or len(cell_defination[1]) != 6:
        raise Exception("RC cell defination is wrong!")
    for j in range(int(cell_defination[0][0])):
        ## create the excitatory neuron group 
        name_neuron_group = 'RC'+ str(i) + str(j) + 'e' 
        RCe_population = int(int(cell_defination[j+1][0]) * float(cell_defination[j+1][1])) 

        RCe_population_list.append(RCe_population)
        neuron_group[name_neuron_group] = NeuronGroup(RCe_population, neuron_eqs_excite, threshold = v_thresh_excite_str, refractory = t_refrac_excite, reset = reset_excite)
        spike_monitor[name_neuron_group] = SpikeMonitor(neuron_group[name_neuron_group])
        neuron_group[name_neuron_group].v = v_rest_excite
        neuron_group[name_neuron_group].timer = 0 * b2.ms
        neuron_group[name_neuron_group].ge = 0 
        neuron_group[name_neuron_group].gi = 0 
        ## create the inhibite neuron group 
        name_neuron_group = 'RC'+ str(i) + str(j) + 'i' 
        RCi_population = int(cell_defination[j+1][0]) - RCe_population
        RCi_population_list.append(RCi_population)
        neuron_group[name_neuron_group] = NeuronGroup(RCi_population, neuron_eqs_inhibite, threshold = v_thresh_inhibite_str, refractory = t_refrac_inhibite, reset = reset_inhibite)
        spike_monitor[name_neuron_group] = SpikeMonitor(neuron_group[name_neuron_group])
        neuron_group[name_neuron_group].v = v_rest_inhibite
        neuron_group[name_neuron_group].timer = 0 * b2.ms
        neuron_group[name_neuron_group].ge = 0 
        neuron_group[name_neuron_group].gi = 0 
        ## inner synapse connection
        ## inner synapse connection
        ## ee 
        name_neuron_group_src = 'RC' + str(i) + str(j) + 'e'
        name_neuron_group_tgt = 'RC' + str(i) + str(j) + 'e'
        connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
        num_src = RCe_population
        num_tgt = RCe_population
        prob = float(cell_defination[j+1][2])
        model_synapse = model_synapse_base
        on_pre = spike_pre_excite
        on_post = '' 
        if prob != 0:
            synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
            if test_mode:
                index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                if len(index_i_random) > 0 and len(index_j_random) > 0: 
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
            else:
                index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = weight_random 
        ## ei
        name_neuron_group_src = 'RC' + str(i) + str(j) + 'e'
        name_neuron_group_tgt = 'RC' + str(i) + str(j) + 'i'
        connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
        num_src = RCe_population
        num_tgt = RCi_population
        prob = float(cell_defination[j+1][3])
        model_synapse = model_synapse_base
        on_pre = spike_pre_excite
        on_post = '' 
        if prob != 0:
            synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
            if test_mode:
                index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                if len(index_i_random) > 0 and len(index_j_random) > 0: 
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
            else:
                index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = weight_random 
        ## ie 
        name_neuron_group_src = 'RC' + str(i) + str(j) + 'i'
        name_neuron_group_tgt = 'RC' + str(i) + str(j) + 'e'
        connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
        num_src = RCi_population
        num_tgt = RCe_population
        prob = float(cell_defination[j+1][4])
        model_synapse = model_synapse_base
        on_pre = spike_pre_inhibite
        on_post = '' 
        if prob != 0:
            synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
            if test_mode:
                index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
            else:
                index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = weight_random 
        ## ii
        name_neuron_group_src = 'RC' + str(i) + str(j) + 'i'
        name_neuron_group_tgt = 'RC' + str(i) + str(j) + 'i'
        connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
        num_src = RCi_population
        num_tgt = RCi_population
        prob = float(cell_defination[j+1][5])
        model_synapse = model_synapse_base
        on_pre = spike_pre_inhibite
        on_post = '' 
        if prob != 0:
            synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
            if test_mode:
                index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
            else:
                index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                if len(index_i_random) > 0 and len(index_j_random) > 0:
                    synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                    synapse[connection_name].w = weight_random 
##-----------------------------------------------------------------------------------------
## create the synapse from 'IN' to 'RCe'
##-----------------------------------------------------------------------------------------
for i in range(num_RC_layer):
    interconnection = pickle.load(interconnection_file)
    if len(interconnection[0]) != len(interconnection):
        raise Exception("Interconnection difination is wrong!")
    for j in range(len(interconnection[0]) - 1):
        if j == 0:
            for k in range(int(interconnection[0][-1])):
                name_neuron_group_src = 'IN'
                name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'e'
                connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                num_src = IN_population
                num_tgt = RCe_population_list[sum(interconnection[0][:-1]) -1 + k]
                prob = float(interconnection[j+1][0])
                model_synapse = model_synapse_base
                on_pre = spike_pre_excite
                on_post = '' 
                if prob != 0:
                    synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                    if test_mode:
                        index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                        index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                        if len(index_i_random) > 0 and len(index_j_random) > 0:  
                            synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                            synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                    else:
                        index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                        if len(index_i_random) > 0 and len(index_j_random) > 0:  
                            synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                            synapse[connection_name].w = weight_random 
                name_neuron_group_src = 'IN'
                name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'i'
                connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                num_src = IN_population
                num_tgt = RCi_population_list[sum(interconnection[0][:-1]) -1 + k]
                prob = float(interconnection[j+1][1])
                model_synapse = model_synapse_base
                on_pre = spike_pre_excite
                on_post = '' 
                if prob != 0:
                    synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                    if test_mode:
                        index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                        index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                        if len(index_i_random) > 0 and len(index_j_random) > 0:  
                            synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                            synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                    else:
                        index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                        if len(index_i_random) > 0 and len(index_j_random) > 0:  
                            synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                            synapse[connection_name].w = weight_random 
        else: 
            for m in range(int(interconnection[0][j])):
                for k in range(int(interconnection[0][-1])):
                    ## ee 
                    name_neuron_group_src = 'RC' + str(j-1) + str(m) + 'e'
                    name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'e'
                    connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                    num_src = RCe_population_list[sum(interconnection[0][:j]) - 1 + m]
                    num_tgt = RCe_population_list[sum(interconnection[0][:-1]) - 1 + k]
                    prob = float(interconnection[j+1][0][0])
                    model_synapse = model_synapse_base
                    on_pre = spike_pre_excite
                    on_post = '' 
                    if prob != 0:
                        synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                        if test_mode:
                            index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                            index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                            if len(index_i_random) > 0 and len(index_j_random) > 0:  
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                        else:
                            index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = weight_random 
                    ## ei
                    name_neuron_group_src = 'RC' + str(j-1) + str(m) + 'e'
                    name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'i'
                    connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                    num_src = RCe_population_list[sum(interconnection[0][:j]) - 1 + m]
                    num_tgt = RCi_population_list[sum(interconnection[0][:-1]) -1 + k]
                    prob = float(interconnection[j+1][0][1])
                    model_synapse = model_synapse_base
                    on_pre = spike_pre_excite
                    on_post = '' 
                    if prob != 0:
                        synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                        if test_mode:
                            index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                            index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                        else:
                            index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = weight_random 
                    ## ie 
                    name_neuron_group_src = 'RC' + str(j-1) + str(m) + 'i'
                    name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'e'
                    connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                    num_src = RCi_population_list[sum(interconnection[0][:j]) - 1 + m]
                    num_tgt = RCe_population_list[sum(interconnection[0][:-1]) - 1 + k]
                    prob = float(interconnection[j+1][1][0])
                    model_synapse = model_synapse_base
                    on_pre = spike_pre_excite
                    on_post = '' 
                    if prob != 0:
                        synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                        if test_mode:
                            index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                            index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                            if len(index_i_random) > 0 and len(index_j_random) > 0:
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                        else:
                            index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = weight_random 
                    ## ii
                    name_neuron_group_src = 'RC' + str(j-1) + str(m) + 'i'
                    name_neuron_group_tgt = 'RC' + str(i) + str(k) + 'i'
                    connection_name = name_neuron_group_src + '_' + name_neuron_group_tgt
                    num_src = RCi_population_list[sum(interconnection[0][:j]) - 1 + m]
                    num_tgt = RCi_population_list[sum(interconnection[0][:-1]) -1 + k]
                    prob = float(interconnection[j+1][1][1])
                    model_synapse = model_synapse_base
                    on_pre = spike_pre_excite
                    on_post = '' 
                    if prob != 0:
                        synapse[connection_name] = b2.Synapses(neuron_group[name_neuron_group_src], neuron_group[name_neuron_group_tgt], model = model_synapse, on_pre = on_pre, on_post = on_post)
                        if test_mode:
                            index_i_random = np.load(path_random_connection + 'index_i_' + connection_name + '.npy')
                            index_j_random = np.load(path_random_connection + 'index_j_' + connection_name + '.npy')
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = np.load(path_random_connection + 'weight_random_' + connection_name + '.npy')
                        else:
                            index_i_random, index_j_random, weight_random = Random_connection_generator(num_src, num_tgt, prob, path_random_connection, connection_name)
                            if len(index_i_random) > 0 and len(index_j_random) > 0: 
                                synapse[connection_name].connect(i = index_i_random, j = index_j_random)
                                synapse[connection_name].w = weight_random 
cell_defination_file.close()
interconnection_file.close()
##-----------------------------------------------------------------------------------------
## setup the network and run the simulations 
##-----------------------------------------------------------------------------------------
net = Network()
for obj_sim in [neuron_group, spike_monitor, synapse]:
    for key in obj_sim:
        net.add(obj_sim[key])
net.run(num_example * (working_time + resting_time), report = 'text', report_period = 600 * second)
np.save(RC_defination_path + '/RCe_population_array.npy', RCe_population_list)
np.save(RC_defination_path + '/RCi_population_array.npy', RCi_population_list)
