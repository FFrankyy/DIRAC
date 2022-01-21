import numpy as np
from tqdm import tqdm
import math
import networkx as nx
import time
import copy
from config import parsers


# maximize
def getEnergy(g, states):
    energy = 0.0
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    return energy

def LocalImprove(g, states):  # greedily flip the node with the maximum energy increase
    energy = getEnergy(g, states)
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
        energy += best_delta / len(g)
        if best_delta == 0:
            stopCondition = True
    return energy, states

def OneStepQ(sg, g, states):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    # init_states = [g.nodes[node]['state'] for node in range(len(g))]
    init_states = states

    g_temp = copy.deepcopy(g)
    for node in g.nodes():
        g_temp.nodes[node]['state'] = 1
    for edge in g_temp.edges():
        src, tgt = edge[0], edge[1]
        g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

    energy, states, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
    temp_states = []
    for node in range(len(g_temp)):
        temp_states.append(states[node]/init_states[node])
    init_states = temp_states

    # for node in g.nodes():
    #     g.nodes[node]['state'] = init_states[node]
    #res_improve = LocalImprove(g, res_best*len(g))
    statesNew = init_states
    return -energy*len(g), statesNew

def MultiQ(sg, g, states):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    # init_states = [g.nodes[node]['state'] for node in range(len(g))]
    init_states = states
    stopCondition = False  # whether to stop
    res_best = -1000000
    num_step = 0
    while not stopCondition:
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

        energy, states, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states

        if res_best < energy:
            res_best = energy
        else:
            stopCondition = True
    
    # for node in g.nodes():
    #     g.nodes[node]['state'] = init_states[node]
    #res_improve = LocalImprove(g, res_best*len(g))
    return -res_best*len(g), init_states

# random flip a certain number of spins to get a random solution
def permutation(g, old_states):
    delta_energy = 0.0
    flip_num = 2
    new_states = copy.deepcopy(old_states)
    flipped_nodes = np.random.choice(len(g), size=flip_num, replace=False)
    for node in flipped_nodes:
        for neigh in g.neighbors(node):
            delta_energy += -2 * new_states[node] * new_states[neigh] * g[node][neigh]['weight']
        new_states[node] = new_states[node] * -1
    return new_states, delta_energy

def SA(g, init_states, std_num):
    # simulated annealing
    Tmax = 200  # initiate temperature
    T = Tmax
    Tmin = 1  # minimum value of terperature
    energy_list = []
    states = init_states
    current_energy = getEnergy(g, states)
    k = 50  # times of internal circulation
    step = 0  # time
    flag = True
    start = time.time()
    max_energy = -1
    while flag:	
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
                    if current_energy > max_energy:
                        max_energy = current_energy
        step += 1
        T = Tmax / (1 + step)
        #print ('num_nodes:%d, step:%d, energy:%.4f'%(len(g), step, current_energy/len(g)))
        energy_list.append(current_energy)
        if T < Tmin and len(energy_list) > std_num:
            std = np.std(energy_list[-std_num:])
            if std < 0.1:
                flag = False
    end = time.time()
    time_cost = end - start
    return states, max_energy/len(g), time_cost

def SA_iter(g, nepochs, fout):
    # simulated annealing
    Tmax = 200  # initiate temperature
    T = Tmax
    Tmin = 1  # minimum value of terperature
    energy_list, time_list = [], []
    # states = [1 for i in range(len(g))]
    states = [int(np.random.choice([-1,1], 1)) for i in range(len(g))] # random
    current_energy = getEnergy(g, states)
    max_energy = -10000000000
    k = len(g) # times of internal circulation
    start = time.time()
    for epoch in range(nepochs):
        print ('Epoch: %d'%epoch)
        step = 0  # time
        flag = True
        temp_energy_list = []
        while flag:
            for i in range(k):
                # generate a new state in the neighboorhood of state by transform function
                statesNew, delta_energy = permutation(g, states)  # can set to be related to the temperature T
                if delta_energy > 0.0:
                    states = statesNew
                    current_energy += delta_energy
                    if current_energy > max_energy:
                        max_energy = current_energy
                        max_states = states
                else:
                    # metropolis principle
                    if np.random.uniform(low=0, high=1) < math.exp(delta_energy / T):
                        states = statesNew
                        current_energy += delta_energy
            step += 1
            T = Tmax / (1 + step)
            temp_energy_list.append(current_energy)
            if T < Tmin and len(temp_energy_list) > 20:
                std = np.std(temp_energy_list[-20:])
                if std < 0.1:
                    flag = False
        end = time.time()
        time_list.append(end - start)
        # max_energy, states = LocalImprove(g, states)
        energy_list.append(max_energy/len(g))
        fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(args.gid, epoch, max_energy/len(g)))
        fout.flush()
        T = Tmax
    return energy_list, time_list

def SA_iter_Q(sg, g, nepochs):
    # simulated annealing
    Tmax = 200  # initiate temperature
    Tmid = 20
    T = Tmax
    Tmin = 1  # minimum value of terperature
    energy_list, time_list = [], []
    states = [1 for i in range(len(g))]
    current_energy = getEnergy(g, states)
    k = 50  # times of internal circulation
    step = 0  # time
    # flag = True
    start = time.time()
    max_energy = -100000000000
    for epoch in range(nepochs):
        print ('Epoch: %d'%epoch)
        step = 0  # time
        while True:
            if T >= Tmid:
                for i in range(k):
                    # generate a new state in the neighboorhood of state by transform function
                    statesNew, delta_energy = permutation(g, states)  # can set to be related to the temperature T
                    if delta_energy > 0.0:
                        states = statesNew
                        current_energy += delta_energy
                        if current_energy > max_energy:
                            max_energy = current_energy
                    else:
                        # metropolis principle
                        if np.random.uniform(low=0, high=1) < math.exp(delta_energy / T):
                            states = statesNew
                            current_energy += delta_energy
                            if current_energy > max_energy:
                                max_energy = current_energy
            elif T < Tmid:
                current_energy, statesNew = OneStepQ(sg, g, states)
                if current_energy > max_energy:
                    max_energy = current_energy
                    states = statesNew
                break
            step += 1
            T = Tmax / (1 + step)
        end = time.time()
        time_list.append(end-start)
        energy_list.append(max_energy/len(g))
        fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(args.gid, epoch, max_energy/len(g)))
        fout.flush()
        T = Tmax
    return energy_list, time_cost

def SATest(method, lattice_dim, test_scale, nepochs, gid):
    if method == 'SA':
        fout = open('../TestData/SA_res/temp_data/Lattice_%dD_num_%d_SA_nepochs_%d_gid_%d.txt'%(lattice_dim, test_scale, nepochs, gid), 'w')
    else:
        sg = SGRL()
        if lattice_dim == 3:
            model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
        elif lattice_dim == 4:
            model_file = './models/4D/nrange_2_3_iter_49800.ckpt'
        sg.LoadModel(model_file)
        fout = open('../TestData/SA_res/temp_data/Lattice_%dD_num_%d_SA_RL_nepochs_%d_gid_%d.txt'%(lattice_dim, test_scale, nepochs, gid), 'w')
    
    print ('SA, lattice dim:%d, test scale:%d, gid:%d'%(lattice_dim, test_scale, gid))

    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    if method == 'SA':
        energy_list, time_list = SA_iter(g, nepochs, fout)
    else:
        energy_list, time_list = SA_iter_Q(sg, g, nepochs, fout)

    # for k in range(len(energy_list)):
    #     fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(i, k, energy_list[k]))
    #     fout.flush()
    fout.write('Time cost:%.4f'%(time_list[-1]))
    fout.close()

if __name__ == "__main__":
    args = parsers()
    # method = 'SA' # SA_RL
    SATest(args.method, args.lattice_dim, args.test_scale, args.numInits, args.gid)
