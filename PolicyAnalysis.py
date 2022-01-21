# -*- coding: utf-8 -*-

from SGRL import SGRL
import networkx as nx
import os
import copy
import numpy as np

def get_energy(g, states):
    energy = 0.0
    for edge in g.edges():
        src, tgt = edge[0], edge[1]
        energy += states[src]*states[tgt]*g[src][tgt]['weight']
    return energy

def LocalImprove(g, states, energy): #greedily flip the node with the maximum energy increase
    stopCondition = False  # whether to stop
    sol = []
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
        if best_delta > 0.0:
        	sol.append(best_node)
        if best_delta == 0:
            stopCondition = True
    return energy/len(g), sol


def GreedyStrategy(g):
    sol = []
    best_energy = -1000000.0
    states = [1 for i in range(len(g))]  # random
    energy = 0.0
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    stopCondition = False   # for each state, run multiple times to obtain the local optimal
    while not stopCondition:
        best_delta, best_node = 0.0, 0
        for node in g.nodes:
            delta = 0.0
            for neigh in g.neighbors(node):
                delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
            if delta >= 0 and best_delta < delta:
                best_delta = delta
                best_node = node
        states[best_node] *= -1
        energy += best_delta
        if best_delta > 0.0:
        	sol.append(best_node)
        if best_delta == 0.0:
            stopCondition = True # no other flips can increase the energy
    return energy/len(g), sol


def RLStrategy(sg, g):
    # energy, states, sol = sg.Evaluate(g, isNetworkx=1, step=1)
    # energy_improve, sol_improve = LocalImprove(g, states, energy*len(g))
    # return energy, sol, energy_improve, sol_improve
    sol_total = []
    stopCondition = False  # whether to stop
    energy_best = -1000000
    init_states = [1 for i in range(len(g))]
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    iter_num = 0
    while not stopCondition:
        print ('iter:%d, energy:%.4f\n'%(iter_num, get_energy(g, init_states)))
        iter_num += 1
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

        energy, states, sol = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        sol_total += sol
        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states

        if energy_best < energy:
            energy_best = energy
        else:
            stopCondition = True
    res_improve, sol = LocalImprove(g, init_states, energy_best*len(g))
    sol_total += sol
    return res_improve, sol_total


sg = SGRL()
model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
sg.LoadModel(model_file)

lattice_scale = 4
for gid in range(50):
	g_file = '../TestData/Lattice/3D/%d/%d.gml'%(lattice_scale, gid)
	G = nx.read_gml(g_file)
	g = nx.Graph()
	for edge in G.edges():
	    g.add_edge(int(edge[0]), int(edge[1]))
	    g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	for node in G.nodes():
	    g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
	    g.nodes[int(node)]['state'] = G.nodes[node]['state']

	# energy_Greedy, sol_Greedy = GreedyStrategy(g)
	energy_RL, sol_RL = RLStrategy(sg, g)

	file_dir = './test/PolicyAnalysis/%d'%lattice_scale
	if not os.path.exists(file_dir):
	    os.mkdir(file_dir)

	# fout1 = open('%s/GreedySol_g%d.txt'%(file_dir, gid), 'w')
	# for item in sol_Greedy:
	#     fout1.write('%d\n' % item)
	# fout1.close()

	fout2 = open('%s/RLSol_g%d.txt'%(file_dir, gid), 'w')
	for item in sol_RL:
	    fout2.write('%d\n' % item)
	fout2.close()

	# fout3 = open('%s/RLSolImprove_g%d.txt'%(file_dir, gid), 'w')
	# for item in sol_RL_improve:
	#     fout3.write('%d\n' % item)
	# fout3.close()

	# print ('Graph:%d, Greedy energy:%.4f'%(gid, energy_Greedy))
	print ('Graph:%d, RL energy:%.4f'%(gid, energy_RL))
	# print ('Graph:%d, RL energy_improve:%.4f'%(gid, energy_RL_improve))
