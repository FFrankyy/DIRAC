from SGRL import SGRL
import numpy as np
from tqdm import tqdm
import math
import networkx as nx
import time
import copy
import uuid
import shutil
from config import parsers
import mcstep
import graph


def GenNetwork(g):  # networkx2four
    edges = g.edges()
    weights = []
    node_states = []
    node_coords = []
    for edge in edges:
        weights.append(g[edge[0]][edge[1]]['weight'])
    for i in range(len(g.nodes())):
        node_states.append(g.nodes[i]['state'])
        for k in range(args.lattice_dim):
            node_coords.append(g.nodes[i]['coords'][k])
    if len(edges) > 0:
        a, b = zip(*edges)
        A = np.array(a)
        B = np.array(b)
        W = np.array(weights)
    else:
        A = np.array([0])
        B = np.array([0])
        W = np.array([0])
    return graph.py_Graph(len(g.nodes()), len(edges), A, B, W, node_states, node_coords, args.lattice_dim)

def LocalImprove(g, states, energy):  # greedily flip the node with the maximum energy increase
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
    return energy

def OneStepQ(sg, g, old_states):
    old_energy = getEnergy(g, old_states)
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    init_states = [old_states[node] for node in range(len(g))]

    g_temp = copy.deepcopy(g)
    for node in g.nodes():
        g_temp.nodes[node]['state'] = 1
    for edge in g_temp.edges():
        src, tgt = edge[0], edge[1]
        g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

    energy, states, _, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
    temp_states = []
    for node in range(len(g_temp)):
        temp_states.append(states[node] / init_states[node])
    new_states = temp_states
    delta_energy = energy * len(g) - old_energy
    return new_states, delta_energy, energy*len(g)


# maximize
def getEnergy(g, states):
    energy = 0.0
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    return energy


def Randomize_beta(g, old_states, beta, betas):
    new_states = copy.deepcopy(old_states)
    for node in g.nodes():
        rn = np.random.uniform(0, 1)
        prob = 0.5 * (betas[0] - beta) / (betas[0] - betas[-1])
        if rn < prob:
            new_states[node] *= -1
    return new_states

def EnergyDescent(g, states, beta):
    mc = mcstep.py_SA(GenNetwork(g), states, beta)  # one sweep
    mc.Run() # run one sweep
    states = mc.states
    temp_max_energy = -1 * mc.lowest_energy
    temp_max_state = mc.lowest_states
    return states, temp_max_energy, temp_max_state

def DIRACDescent(sg, g, states, beta, betas):
    statesNew, delta_energy, energy = OneStepQ(sg, g, states)
    if delta_energy > 0.0:
        states = statesNew
        current_energy = energy
    else:
        # metropolis principle
        states = Randomize_beta(g, statesNew, beta, betas)
        current_energy = getEnergy(g, states)
    return states, current_energy, states

def SA(sg, g, init_states):
    NS = 50
    NT = 100
    betamin = 0.001
    betamax = 5.000
    betas = np.linspace(betamin, betamax, NT)  # beta从小到大，对应温度从高到低
    states = init_states
    max_energy = getEnergy(g, states)
    max_states = states
    start = time.time()
    for beta in betas:
        for ns in range(NS):
            if np.random.uniform(0, 1) < 0.5:
                states, temp_max_energy, temp_max_state = EnergyDescent(g, states, beta)
            else:
                states, temp_max_energy, temp_max_state = DIRACDescent(sg, g, states, beta, betas)
            if temp_max_energy > max_energy:
                max_energy = temp_max_energy
                max_states = temp_max_state
            end1 = time.time()
            print('Dim:%d, Scale: %d, Graph: %d, beta: %.3f, max_energy: %.8f, time_cost: %.2f'%(args.lattice_dim, args.test_scale, args.gid, beta, max_energy/len(g), end1 - start))
    end2 = time.time()
    time_cost = end2 - start
    return max_energy/len(g), max_states, time_cost


def SA_test(sg, g):
    init_states = [int(np.random.choice([-1, 1], 1)) for i in range(len(g))]  # random
    energy, states, time_cost = SA(sg, g, init_states)
    energy_improve = LocalImprove(g, states, energy)
    return energy, energy_improve, time_cost


def SATest(sg, lattice_dim, test_scale, gid, hardness):
    print ('lattice dim:%d, test scale:%d, gid:%d'%(lattice_dim, test_scale, gid))
    g_file = './data/%dD/%d/%d.gml'%(lattice_dim, test_scale, gid)
    g = nx.Graph()
    for line in open(g_file, 'r'):
        data = line.strip().split()
        src, tgt, weight = int(data[0]), int(data[1]), int(data[2])
        g.add_edge(src, tgt)
        g[src][tgt]['weight'] = weight
    for node in g.nodes():
        g.nodes[node]['state'] = 1
        num = 10
        x_y = int(int(node) % (num * num))
        z = int(int(node) / (num * num))
        x = int(x_y / num)
        y = int(x_y % num)
        g.nodes[node]['coords'] = (x, y, z)
        
    fout = open('../TestData/DIRAC_SA_g_%d.txt'%(hardness, gid), 'w')
    energy, energy_improve, time_cost = SA_test(sg, g)
    fout.write('Graph: %d, result: %.12f, result_improve: %.12f, time_cost: %.2f\n'%(gid, energy, energy_improve, time_cost))     
    fout.close()


if __name__ == "__main__":
    args = parsers()

    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_336300.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_295200.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_172500.ckpt'

    sg = SGRL()
    sg.LoadModel(model_file)

    SATest(sg, args.lattice_dim, args.test_scale, args.gid, args.hardness)
