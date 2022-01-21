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

args = parsers()


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

def RL(sg, g, init_states, stepRatio):
    #stepRatio = 0.0
    step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])
    time1 = time.time()
    stopCondition = False  # whether to stop
    energy_best = -1000000
    num_step = 0
    while not stopCondition:
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

        time_start = time.time()
        energy, states, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        time_end = time.time()
        print ('step:%d, energy:%.4f, time_cost:%.4f'%(num_step, energy, time_end-time_start))
        num_step += 1

        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states

        if energy_best < energy:
            energy_best = energy
        else:
            stopCondition = True
    time2 = time.time()
    energy_res = LocalImprove(g, init_states, energy_best*len(g))
    time3 = time.time()
    time_cost1 = time3 - time2
    time_cost2 = time3 - time1
    return energy_res, time_cost1, time_cost2, num_step-1

def RLTest(model_file, lattice_dim, test_scale, stepRatio, numInits, gid):
    sg = SGRL()
    # model_file = sg.findModel()
    sg.LoadModel(model_file)
    file_path = '../TestData/DQN_res/temp_data/%dD/%d/Step_%.4f'%(lattice_dim, test_scale, stepRatio)
    if not os.path.exists(file_path):
        os.mkdir(file_path)
 
    fout = open('%s/Lattice_%dD_num_%d_RL_numInits_%d_stepRatio_%.4f_gid_%d.txt'%(file_path, lattice_dim, test_scale, numInits, stepRatio, gid), 'w')
    # read graph
    print ('RL, lattice dim:%d, test scale:%d, stepRatio:%.4f, gid:%d'%(lattice_dim, test_scale, stepRatio, gid))
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['state'] = G.nodes[node]['state']
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
    
    for init_num in tqdm(range(numInits)):
        g_temp = copy.deepcopy(g)
        if numInits == 1:
            init_states = [1 for i in range(len(g_temp))]
        else:
            init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]
        energy, time_cost1, time_cost2, Q_nums = RL(sg, g_temp, init_states, stepRatio)
        fout.write('Graph: %d, init_state:%d, result: %.4f, time_LocalImprove:%.4f, time_all:%.4f, Q_nums:%d\n'%(gid, init_num, energy, time_cost1, time_cost2, Q_nums))
        fout.flush()
    fout.close()

if __name__=="__main__":

    args = parsers()
    model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
    #model_file = './models/4D/nrange_2_3_iter_49800.ckpt'
    RLTest(model_file, args.lattice_dim, args.test_scale, args.stepRatio, args.numInits, args.gid)
