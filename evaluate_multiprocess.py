from SGRL import SGRL
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
from config import parsers
import copy
import os
import uuid
import shutil
import math

args = parsers()

num_epochs = {}
num_epochs[2], num_epochs[3], num_epochs[4] = {}, {}, {}
num_epochs[2][10], num_epochs[2][15], num_epochs[2][20] = 23000, 20000, 30000
num_epochs[3][6], num_epochs[3][8], num_epochs[3][10] = 10000, 20000, 30000
num_epochs[4][4], num_epochs[4][5], num_epochs[4][6] = 10000, 20000, 30000

def LocalImprove(g, states, energy): #greedily flip the node with the maximum energy increase
    stopCondition = False  # whether to stop
    while not stopCondition:
        best_delta, best_node = 0, 0
        for node in g.nodes:
            delta = 0
            for neigh in g.neighbors(node):
                delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
            if delta >= 0 and best_delta < delta:
                best_delta = delta
                best_node = node
        states[best_node] *= -1
        energy += best_delta
        if best_delta == 0:
            stopCondition = True
    return energy/len(g)

#maximize
def getEnergy(g, states):
    energy = 0.0
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    return energy

# random flip a certain number of spins to get a random solution
def permutation(g, old_states):
    delta_energy = 0.0
    flip_num = 2
    new_states = copy.deepcopy(old_states)
    flipped_nodes = np.random.choice(len(g),size=flip_num, replace=False)
    for node in flipped_nodes:
        for neigh in g.neighbors(node):
            delta_energy += -2 * new_states[node] * new_states[neigh] * g[node][neigh]['weight']
        new_states[node] = new_states[node]*-1
    return new_states, delta_energy

def SA(g, states):
    # simulated annealing
    Tmax = 200  # initiate temperature
    T = Tmax
    Tmin = 1  # minimum value of terperature
    energy_list = []
    current_energy = getEnergy(g, states)
    k = 50  # times of internal circulation
    step = 0  # time
    flag = True
    start = time.time()
    while flag: #连续20次结果不变才停止
        # while True:
        for i in range(k):
            # generate a new state in the neighboorhood of state by transform function
            statesNew, delta_energy = permutation(g, states)  # can set to be related to the temperature T
            if delta_energy > 0.0:
                states = statesNew
                current_energy += delta_energy
            else:
                # metropolis principle
                if np.random.uniform(low=0, high=1) < math.exp(delta_energy / T):
                    states = statesNew
                    current_energy += delta_energy
        step += 1
        #T = np.max([Tmax / (1 + step), Tmin])
        T = Tmax / (1+step)
        #print ('SA, lattice_dim: %d, test_scale: %d, step:%d, energy:%.4f'%(args.lattice_dim, args.test_scale, step, current_energy/len(g)))
        energy_list.append(current_energy)
        if step > 50:
            energy_list = energy_list[len(energy_list)-50:]
            if np.std(energy_list) < 0.0001:
                flag = False
    energy_res = LocalImprove(g, states, current_energy)
    end = time.time()
    time_cost = end - start
    return current_energy/len(g), energy_res, time_cost

def GreedyEnergy(g, states):
    energy = 0.0
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']

    stopCondition = False   # for each state, run multiple times to obtain the local optimal
    while not stopCondition:
        best_delta, best_node = 0.0, 0
        for node in g.nodes:
            delta = 0
            for neigh in g.neighbors(node):
                delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
            if delta >= 0 and best_delta < delta:
                best_delta = delta
                best_node = node
        states[best_node] *= -1
        energy += best_delta
        if best_delta == 0:
            stopCondition = True # no other flips can increase the energy
    return energy/len(g)

def Greedy(lattice_dim, test_scale, gid, ifLoadPTstates):
    fout = open('../TestData/Greedy_res/temp_data/Lattice_%dD_num_%d_Greedy_gid_%d.txt'%(lattice_dim, test_scale, gid), 'w')
    # read graph
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']

    if ifLoadPTstates:
        node_states = {}
        for line in open('../TestData/PT_res/temp_data/Lattice_%dD_num_%d_PT_nepochs_%d_gid_%d_states.txt'%(lattice_dim, test_scale, num_epochs[lattice_dim][test_scale], gid), 'r'):
    	    data = line.strip().split(',')
    	    node_id = int(data[0].strip().split(':')[1])
    	    state = int(data[1].strip().split(':')[1])
    	    node_states[node_id] = state
        states = [node_states[k] for k in range(len(node_states))]
    else:
        states = [1 for i in range(len(g))]

    g_temp = copy.deepcopy(g)
    time1 = time.time()
    energy = GreedyEnergy(g_temp, states)
    time2 = time.time()
    time_cost = time2 - time1
    fout.write('Graph: %d, result: %.16f, result_improve: %.16f, time:%.4f\n'%(gid, 0, energy, time_cost))
    print('Graph: %d, result: %.16f, result_improve: %.16f, time:%.4f\n'%(gid, 0, energy, time_cost))
    fout.close()

def SA_Test(lattice_dim, test_scale, gid, ifLoadPTstates):
    fout = open('../TestData/SA_res/temp_data/Lattice_%dD_num_%d_SA_gid_%d.txt'%(lattice_dim, test_scale, gid), 'w')
    # read graph
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']

    if ifLoadPTstates:
        node_states = {}
        for line in open('../TestData/PT_res/temp_data/Lattice_%dD_num_%d_PT_nepochs_%d_gid_%d_states.txt'%(lattice_dim, test_scale, num_epochs[lattice_dim][test_scale], gid), 'r'):
    	    data = line.strip().split(',')
    	    node_id = int(data[0].strip().split(':')[1])
    	    state = int(data[1].strip().split(':')[1])
    	    node_states[node_id] = state
        states = [node_states[k] for k in range(len(node_states))]
    else:
        states = [1 for i in range(len(g))]

    g_temp = copy.deepcopy(g)
    res, res_improve, time_cost = SA(g_temp, states)
    fout.write('Graph: %d, result: %.16f, result_improve: %.16f, time:%.4f\n'%(gid, res, res_improve, time_cost))
    print('Graph: %d, result: %.16f, result_improve: %.16f, time:%.4f\n'%(gid, res, res_improve, time_cost))
    fout.close()


def MetaQ(model_file, lattice_dim, test_scale, stepRatio, gid):
    sg = SGRL()
    sg.LoadModel(model_file)
    fout = open('./DQN_res/temp_data/Lattice_%dD_num_%d_stepRatio_%.4f_MetaQ_gid_%d.txt'%(lattice_dim, test_scale, stepRatio, gid), 'w')
    # read graph
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for node in G.nodes():
        g.nodes[int(node)]['state'] = 1
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    if stepRatio > 0:
        step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
    else:
        step = 1

    start = time.time()
    res, states, sol, _ = sg.Evaluate(g, isNetworkx=1, step=step)
    res_improve = LocalImprove(g, states, res*len(g))
    end = time.time()
    fout.write('Graph: %d, result: %.4f, result_improve:%.4f, time:%.4f\n'%(gid, res, res_improve, end-start))
    fout.close()

def MultiQ(model_file, lattice_dim, test_scale, stepRatio, gid, ifLoadPTstates):
    sg = SGRL()
    sg.LoadModel(model_file)
    fout = open('../TestData/DQN_res/temp_data/Lattice_%dD_num_%d_stepRatio_%.4f_MultiQ_gid_%d.txt'%(lattice_dim, test_scale, stepRatio, gid), 'w')
    # read graph
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['state'] = 1
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    if stepRatio > 0:
        step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
    else:
        step = 1
    start = time.time()

    if ifLoadPTstates:
        node_states = {}
        for line in open('../TestData/PT_res/temp_data/Lattice_%dD_num_%d_PT_nepochs_%d_gid_%d_states.txt'%(lattice_dim, test_scale, num_epochs[lattice_dim][test_scale], gid), 'r'):
        	data = line.strip().split(',')
        	node_id = int(data[0].strip().split(':')[1])
        	state = int(data[1].strip().split(':')[1])
        	node_states[node_id] = state
        init_states = [node_states[k] for k in range(len(node_states))]
    else:
        init_states = [1 for i in range(len(g))]
    stopCondition = False  # whether to stop
    energy_best = -1000000
    Qnum = 0
    while not stopCondition:
        g_temp = copy.deepcopy(g)
        for node in g_temp.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]
        energy, states, _, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states
        if energy_best < energy:
            energy_best = energy
        else:
            stopCondition = True
        Qnum += 1
    res_improve = LocalImprove(g, init_states, energy_best*len(g))
    end = time.time()
    fout.write('Graph: %d, result: %.4f, result_improve:%.4f, time:%.4f, Qnum:%d\n'%(gid, energy_best, res_improve, end-start, Qnum-1))
    print('Lattice_dim:%d, test_scale:%d, Graph: %d, result:%.4f, time:%.4f, Qnum:%d\n'%(lattice_dim, test_scale, gid, res_improve, end-start, Qnum-1))
    fout.close()


if __name__=="__main__":

    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'

    # MetaQ(model_file, args.lattice_dim, args.test_scale, args.stepRatio, args.gid)
    MultiQ(model_file, args.lattice_dim, args.test_scale, args.stepRatio, args.gid, ifLoadPTstates=False)
    #SA_Test(args.lattice_dim, args.test_scale, args.gid, ifLoadPTstates=True)
    #Greedy(args.lattice_dim, args.test_scale, args.gid, ifLoadPTstates=True)
