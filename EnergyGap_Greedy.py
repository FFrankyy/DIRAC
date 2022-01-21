import networkx as nx
import numpy as np
from tqdm import tqdm
import numpy as np
import time
import math
from config import parsers
from SGRL import SGRL
import copy
import os
import uuid
import shutil

def getEnergyList(g, sol):
    states = [1 for i in range(len(g))]  # random
    energy, energy_list = 0.0, []
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    energy_list.append(energy/len(g))
    for node in sol:
        delta = 0.0 
        for neigh in g.neighbors(node):
            delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
        states[node] *= -1
        energy += delta
        print ('Lattice_dim: %d, test_scale: %d, MetaQ energy: %.4f'%(args.lattice_dim, args.test_scale, energy/len(g)))
        energy_list.append(energy/len(g))
    return energy_list

def MetaQ(sg, g):
    # step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    step = 1
    energy, states, sol, _ = sg.Evaluate(g, isNetworkx=1, step=step)  # Q result
    energy_list = getEnergyList(g, sol)
    return energy_list

def MultiQ(sg, g):
    # step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    step  = 1
    init_states = [g.nodes[node]['state'] for node in range(len(g))]
    stopCondition = False  # whether to stop
    res_best = -1000000
    sol_all = []
    while not stopCondition:
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

        energy, states, sol = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        sol_all += sol

        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states

        if res_best < energy:
            res_best = energy
        else:
            stopCondition = True

    energy_list = getEnergyList(g, sol_all)
    return energy_list

def Greedy(g, states):
    # sol = []
    # states = [1 for i in range(len(g))]  # random
    energy, energy_list = 0.0, []
    for edge in g.edges():
        energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
    energy_list.append(energy/len(g))
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
        if best_delta > 0.0:
	        states[best_node] *= -1
	        energy += best_delta
        	# sol.append(best_node)
        	print ('Lattice_dim: %d, test_scale: %d, Greedy energy: %.4f'%(args.lattice_dim, args.test_scale, energy/len(g)))
        	energy_list.append(energy/len(g))
        if best_delta == 0.0:
            stopCondition = True # no other flips can increase the energy
    return energy_list

def GapTest(method, lattice_dim, test_scale, gid, file_path):
	g_file = '../TestData/Lattice/EnergyGap/%dD/%d/%d.gml'%(lattice_dim, test_scale, gid)
	G = nx.read_gml(g_file)
	g = nx.Graph()
	for edge in G.edges():
	    g.add_edge(int(edge[0]), int(edge[1]))
	    g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	for node in G.nodes():
	    g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
	    g.nodes[int(node)]['state'] = G.nodes[node]['state']

	if method == 'Greedy':
		states = [g.nodes[node]['state'] for node in range(len(g))]
		energy_list_Greedy = Greedy(g, states)
		fout = open('%s/Greedy_EnergyList_%d.txt'%(file_path, gid), 'w')
		for item in energy_list_Greedy:
			fout.write('%.8f\n'%item)
		fout.close()
	elif method == 'MetaQ':
		sg = SGRL()
		if lattice_dim == 2:
		    model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
		elif lattice_dim == 3:
		    model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
		elif lattice_dim == 4:
		    model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
		sg.LoadModel(model_file)
		energy_list_DIRAC = MetaQ(sg, g)
		fout = open('%s/MetaQ_EnergyList_%d.txt'%(file_path, gid), 'w')
		for item in energy_list_DIRAC:
		    fout.write('%.8f\n'%item)
		fout.close()

if __name__=="__main__":
	args = parsers()
	file_path = './test/PolicyAnalysis/%dD/%d'%(args.lattice_dim, args.test_scale)
	if not os.path.exists('./test/PolicyAnalysis/%dD'%args.lattice_dim):
		os.mkdir('./test/PolicyAnalysis/%dD'%args.lattice_dim)
	if not os.path.exists(file_path):
		os.mkdir(file_path)

	method = 'Greedy' # Greedy, MetaQ
	GapTest(method, args.lattice_dim, args.test_scale, args.gid, file_path)

