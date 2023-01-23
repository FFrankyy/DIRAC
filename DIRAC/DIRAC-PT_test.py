from SGRL import SGRL
import networkx as nx
import numpy as np
import numpy as np
import time
import math
from config import parsers
import copy
import mcstep
import graph

args = parsers()

def GenNetwork(g):  # networkx format to Cython format
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

def OneStepQ(sg, g):
    # step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    step = np.max([int(args.stepRatio * nx.number_of_nodes(g)), 1])
    init_states = [g.nodes[node]['state'] for node in range(len(g))]

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

    g_update = copy.deepcopy(g)
    for node in g_update.nodes():
        g_update.nodes[node]['state'] = init_states[node]
    return -energy*len(g), g_update

# maximize
def getEnergy(g):
    energy = 0.0
    for edge in g.edges():
        energy += -g.nodes[edge[0]]['state'] * g.nodes[edge[1]]['state'] * g[edge[0]][edge[1]]['weight']
    return energy

def Randomize(g):
    for node in g.nodes():
        g.nodes[node]['state'] = 2 * np.random.randint(0, 2, 1)[0] - 1
    return g

def Randomize_beta(g, beta, betas):
    for node in g.nodes():
        rn = np.random.uniform(0, 1)
        prob = 0.5*(betas[0]-beta)/(betas[0]-betas[-1])
        if rn < prob:
            g.nodes[node]['state'] *= -1
    return g

def EnergyDescent(g, beta):
    states = [g.nodes[gid]['state'] for gid in range(len(g))]
    mc = mcstep.py_SA(GenNetwork(g), states, beta)  # one sweep
    mc.Run() # run one sweep
    states = mc.states
    temp_max_energy = -1 * mc.lowest_energy
    temp_max_state = mc.lowest_states
    for gid in range(len(states)):
        g.nodes[gid]['state'] = states[gid]
    return g, temp_max_energy

def DIRACDescent(sg, g, energy_prev, beta, betas):
    energy_current, g_current = OneStepQ(sg, g)
    delta_energy = energy_prev - energy_current
    if delta_energy > 0:
        g = g_current
    elif delta_energy <= 0:
        g = Randomize_beta(g, beta, betas)
        energy_current = getEnergy(g)
    return g, energy_current

def ExchangeMC(sg, g, nepochs, gid, fout):
    betas = 1.0 / np.linspace(0.1, 1.6, 20)
    m = len(betas)
    Randomize(g)
    lowest_energy = getEnergy(g)
    lowest_g = g
    greps = {}
    energys = {}
    for beta in betas:
        g_copy = Randomize(g.copy())
        energys[beta] = getEnergy(g_copy)
        greps[beta] = copy.deepcopy(g_copy)

    time_1 = time.time()
    for epoch in range(nepochs):
        for beta in betas:  # 对每个复本进行标准MC步骤s
            if np.random.uniform(0, 1) < 0.5:
               greps[beta], energys[beta] = EnergyDescent(greps[beta], beta)
            else:
               greps[beta], energys[beta] = DIRACDescent(sg, greps[beta], energys[beta], beta, betas)

            if energys[beta] < lowest_energy:
                lowest_energy = energys[beta]
                lowest_g = copy.deepcopy(greps[beta])
            time_2 = time.time()
            print ('Dim: %d, Scale: %d, Graph: %d, epoch: %d, beta: %.3f, result: %.8f, time_cost: %.2f'%(args.lattice_dim, args.test_scale, gid, epoch, beta, -lowest_energy/len(g), time_2-time_1))
        rindex = np.random.randint(0, m - 1, 1)[0]
        left = betas[rindex]
        right = betas[rindex + 1]
        delta = -(left - right) * (energys[left] - energys[right])
        rn = np.random.uniform(0, 1)
        if delta <= 0 or rn < np.exp(-delta):  # delta作为是否交换复本的判据
            for node in greps[left].nodes:
                temp = greps[right].nodes[node]['state']
                greps[right].nodes[node]['state'] = greps[left].nodes[node]['state']
                greps[left].nodes[node]['state'] = temp
            temp_energy = energys[right]
            energys[right] = energys[left]
            energys[left] = temp_energy
    
        time_3 = time.time()
        fout.write('Graph: %d, epoch: %d, result: %.16f, time: %.4f\n'%(gid, epoch, -lowest_energy/len(g), time_3-time_1))
        fout.flush()
    return -lowest_energy/len(g)
    

def DIRAC_PT_Test(lattice_dim, test_scale, nepochs, gid):
    sg = SGRL()

    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_336300.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_295200.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_172500.ckpt'

    sg.LoadModel(model_file)
    fout = open('DIRAC_PT_nepochs_%d_g_%d.txt'%(nepochs, gid), 'w')
    g_file = './data/%dD/%d/%d.gml'%(lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        src, tgt, weight = int(edge[0]), int(edge[1]), G[edge[0]][edge[1]]['weight']
        g.add_edge(src, tgt)
        g[src][tgt]['weight'] = weight
    for node in G.nodes():
        g.nodes[int(node)]['state'] = int(G.nodes[node]['state'])
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    ExchangeMC(sg, g, nepochs, gid, fout)
    fout.close()


if __name__=="__main__":
    args = parsers()
    DIRAC_PT_Test(args.lattice_dim, args.test_scale, args.numInits, args.gid)

