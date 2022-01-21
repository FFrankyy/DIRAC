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

def MetaQ(sg, g, states):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    # init_states = [g.nodes[node]['state'] for node in range(len(g))]
    init_states = states
    g_temp = copy.deepcopy(g)
    for node in g.nodes():
        g_temp.nodes[node]['state'] = 1
    for edge in g_temp.edges():
        src, tgt = edge[0], edge[1]
        g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]
    energy, states, _, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
    temp_states = []
    for node in range(len(g_temp)):
        temp_states.append(states[node]/init_states[node])
    init_states = temp_states
    statesNew = init_states
    numQ = 1
    return energy*len(g), statesNew, numQ

def MultiQ(sg, g, states):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    # init_states = [g.nodes[node]['state'] for node in range(len(g))]
    init_states = states
    stopCondition = False  # whether to stop
    res_best = -1000000
    numQ = 0
    while not stopCondition:
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

        energy, states, _, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states
        numQ += 1

        if res_best < energy:
            res_best = energy
        else:
            stopCondition = True
    return res_best*len(g), init_states, numQ

def EnergyDescent(g, nMCsteps, temp_energy, T):
    for MCstep in range(nMCsteps):
        node = np.random.randint(0, len(g.nodes), 1)[0]
        neibors = g[node]
        delta_energy = 0.0
        for neibor in neibors:
            delta_energy -= 2*g[node][neibor]['weight'] * g.nodes[neibor]['state'] * g.nodes[node]['state']
        if_change = 0
        if delta_energy >= 0:
            g.nodes[node]['state'] *= -1
            if_change = 1
        elif delta_energy < 0:
            rn = np.random.uniform(0, 1)
            if rn < np.exp(delta_energy/T):
                g.nodes[node]['state'] *= -1
                if_change = 1
        if if_change:
            temp_energy += delta_energy
        states = []
        for node in range(len(g)):
            states.append(g.nodes[node]['state'])
    return g, states, temp_energy

def ExchangeMC(sg, g, gid, method, fout):
    time1 = time.time()
    T = 200 # initial temperature
    nMCsteps = np.max([int(0.01*len(g)), 2])
    TStep = 0
    states = [1 for node in g.nodes()] 
    while TStep < 20001:
    	time_1 = time.time()
	    if method == 'MetaQ':
	        energy, states, numQ = MetaQ(sg, g, states)
	    elif method == 'MultiQ':
	        energy, states, numQ = MultiQ(sg, g, states)
	    TStep += numQ	
	    for node in g.nodes():
            g.nodes[node]['state'] = states[node]
        g, states, energy = EnergyDescent(g, nMCsteps, energy, T)
	    T /= TStep
	    fout.write('Graph: %d, epoch: %d, result: %.16f\n'%(gid, TStep, energy/len(g)))
        fout.flush()
        time_2 = time.time()
        print ('Lattice_dim: %d, test_scale: %d, Graph: %d, epoch: %d, time_cost: %.2f, result: %.8f'%(args.lattice_dim, args.test_scale, gid, TStep, time_2-time_1, energy/len(g)))
    time2 = time.time()
    time_cost = time2 - time1
    return energy/len(g), time_cost

def SATest(lattice_dim, test_scale, gid, method):
	sg = SGRL()
    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
    sg.LoadModel(model_file)
    fout = open('../TestData/PT_res/temp_data/Lattice_%dD_num_%d_RLSA_%s_gid_%d.txt'%(lattice_dim, test_scale, method, gid), 'w')
    print ('lattice dim:%d, test scale:%d, gid:%d'%(lattice_dim, test_scale, gid))
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
    energy, time_cost = ExchangeMC(sg, g, gid, method, fout)


if __name__ == "__main__":
    args = parsers()
    SATest(args.lattice_dim, args.test_scale, args.numInits, args.gid, args.method)
