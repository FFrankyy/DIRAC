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

def MultiQ(sg, g, init_states):
	step = np.max([int(0.01 * len(g)), 1])
	g_temp = copy.deepcopy(g)
    # init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]
	stopCondition = False  # whether to stop
	energy_best = -1000000
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

	    if energy_best < energy:
	        energy_best = energy
	    else:
	        stopCondition = True
	res_improve = LocalImprove(g, init_states, energy_best*len(g))
	return res_improve


if __name__=="__main__":
	args = parsers()
    
	s = args.scale
	lattice_dim = len(s.split('x'))

	if lattice_dim == 2:
		model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
	elif lattice_dim == 3:
		model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
	elif lattice_dim == 4:
		model_file = './models/4D/nrange_2_3_iter_49800.ckpt'
    
	g_file = '../TestData/Lattice/%dD/%s/%d.gml' % (lattice_dim, args.scale, args.gid)
	G = nx.read_gml(g_file)
	g = nx.Graph()
	for edge in G.edges():
		g.add_edge(int(edge[0]), int(edge[1]))
		g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	for node in G.nodes():
		g.nodes[int(node)]['state'] = G.nodes[node]['state']
		g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

	sg = SGRL()
	sg.LoadModel(model_file)

	num = 0
	for line in open('../TestData/Gurobi_res/Lattice_%s_Gurobi.txt'%(args.scale), 'r'):
		if num == args.gid:
		    data = float(line.strip().split(',')[1].strip().split(':')[1])
		    print (data)
		    true_val = float('%.6f'%data)
		    break
		num += 1
	# print ('Dim: %d, scale: %d, gid: %d, true_val: %.6f'%(args.lattice_dim, args.test_scale, args.gid, true_val))

	energy = -1000
	numInits = 0
	while energy != true_val:
		numInits += 1
		init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g))]
		energy = MultiQ(sg, g, init_states)
		energy = float('%.6f'%energy)

	print ('Scale: %s, gid: %d, numInists: %d'%(args.scale, args.gid, numInits))

	# uid = uuid.uuid4()
	if not os.path.exists('./test/GSMatch/num_run_%d'%args.num_run):
		os.mkdir('./test/GSMatch/num_run_%d'%args.num_run)

	write_path = './test/GSMatch/num_run_%d/%s'%(args.num_run, args.scale)
	if not os.path.exists(write_path):
		os.mkdir(write_path)

	fout = open('%s/Lattice_%s_gid_%d.txt'%(write_path, args.scale, args.gid), 'w')
	fout.write('%d'%numInits)
	fout.close()