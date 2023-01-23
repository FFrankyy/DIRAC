from SGRL import SGRL
import networkx as nx
import numpy as np
from config import parsers
import copy

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

def DIRAC1(sg, g, init_states):
	step = np.max([int(0.01 * len(g)), 1])
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
	res_improve = LocalImprove(g, init_states, energy*len(g))
	return res_improve


def DIRAC1_test(sg, lattice_dim, test_scale):
    fout = open('DIRAC1_%dD_%d.txt'%(lattice_dim, test_scale),'w')
    for gid in range(50):
        g_file = './data/%dD/%s/%d.gml'%(lattice_dim, test_scale, gid)
        G = nx.read_gml(g_file)
        g = nx.Graph()
        for edge in G.edges():
            g.add_edge(int(edge[0]), int(edge[1]))
            g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
        for node in G.nodes():
            g.nodes[int(node)]['state'] = G.nodes[node]['state']
            g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

  		init_states = [g.nodes[node]['state'] for node in range(len(g))]
        res = DIRAC1(sg, g, init_states)
        fout.write('DIRAC1, Dim: %d, Size: %d, Graph: %d, res: %.8f\n'%(lattice_dim, test_scale, gid, res))
        fout.flush()
    fout.close()


if __name__=="__main__":
	args = parsers()

	if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_336300.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_295200.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_172500.ckpt'

	sg = SGRL()
	sg.LoadModel(model_file)

	DIRAC1_test(sg, args.lattice_dim, args.test_scale)