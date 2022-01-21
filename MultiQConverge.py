# -*- coding: utf-8 -*-

from SGRL import SGRL
import networkx as nx
import os
import copy
import numpy as np
from config import parsers
# import seaborn as sns
from tqdm import tqdm

args = parsers()

def MultipleQCalc(sg, lattice_dim, test_scale, gid, numInits):
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['state'] = G.nodes[node]['state']
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    # step = 1  # step size

    # read graph
    # energy_list = []
    for i in tqdm(range(numInits)):
        fout = open('./test/MultiQConverge/%dD/%d/Graph_%d_InitState_%d_MultiQ.txt'%(lattice_dim, test_scale, gid, i), 'w')
        g_temp = copy.deepcopy(g)
        init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]
        # energy_temp_list = []
        for k in range(10):# iterate 30 times of multiQ computation
            print ('Graph:%d, init_state_num:%d, Q_iter:%d\n'%(gid, i, k))
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

            # energy_temp_list.append(energy)
            fout.write('%.4f\n'%energy)
            fout.flush()
        fout.close()
        # energy_list.append(energy_temp_list)


if __name__=="__main__":

    sg = SGRL()
    model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
    sg.LoadModel(model_file)

    file_path = './test/MultiQConverge/%dD/%d'%(args.lattice_dim, args.test_scale)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for gid in range(5):
        MultipleQCalc(sg, args.lattice_dim, args.test_scale, gid, numInits=2)