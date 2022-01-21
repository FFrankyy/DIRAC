# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import os
from tqdm import tqdm

NUM_MIN, NUM_MAX = 10, 11
Dim = 2
g_type = 'lattice'
# NUM_MIN, NUM_MAX = 16, 20
# Dim = 3

Periodic = True #If periodic=True then join grid edges with periodic boundary conditions.

def gen_graph(NUM_MIN, NUM_MAX, g_type):
    num = np.random.randint(NUM_MIN, NUM_MAX, 1)[0]
    if g_type == 'lattice':
        # num = np.random.randint(NUM_MIN, NUM_MAX, 1)[0]
        if Dim == 2:
            G = nx.grid_graph(dim=[num, num], periodic=Periodic)
        elif Dim == 3:
            G = nx.grid_graph(dim=[num, num, num], periodic=Periodic)
        numrealnodes = 0
        node_map = {}
        for node in G.nodes():
            node_map[node] = numrealnodes
            numrealnodes += 1
        g = nx.Graph()
        for edge in G.edges():
            src_idx = node_map[edge[0]]
            dst_idx = node_map[edge[1]]
            g.add_edge(src_idx, dst_idx)

    elif g_type == 'complete':
        g = nx.complete_graph(num)

    for edge in nx.edges(g):
        g[edge[0]][edge[1]]['weight'] = np.random.normal(0, 1)
    for node in g.nodes():
        # node state
        g.nodes[node]['state'] = 1
        # g.nodes[node]['state'] = int(np.random.choice([-1, 1], 1))
        if Dim == 2:
            x = int(int(node) / num)
            y = int(int(node) % num)
            g.nodes[node]['coords'] = (x, y)
        elif Dim == 3:
            # node coords
            z = int(int(node) / (num * num))
            x_y = int(int(node) % (num * num))
            x = int(x_y / num)
            y = int(x_y % num)
            g.nodes[node]['coords'] = (x, y, z)
        elif Dim == 4:
            w = int(node / (num*num*num))
            x_y_z = int(node % (num*num*num))
            z = int(x_y_z / (num*num))
            x_y = int(x_y_z % (num*num))
            y = int(x_y / num)
            x = int(x_y % num)
            g.nodes[node]['coords'] = (x, y, z, w)
    return g

def transform_weight(g):
    N = nx.number_of_nodes(g)
    n = int(np.sqrt(N))
    # print (x axis)
    weights = []
    for j in range(n):
        for i in range(n - 1):
            weight = g[i + n * j][i + 1 + n * j]['weight']
            weights.append(weight)
    for l in range(n - 1):
        for m in range(n):
            weight = g[m + n * l][m + n * (l + 1)]['weight']
            weights.append(weight)
    return weights


# for num in [6,7,8,9,10,30,50,70]:
#     for i in tqdm(range(100)):
#         g = gen_graph(num)
#         if Dim == 2:
#             if not os.path.exists('./2D/%d'%num):
#                 os.mkdir('./2D/%d'%num)
#             nx.write_gml(g, './2D/%d/%d.gml'%(num,i))
#             weights = transform_weight(g)
#             f_out = open('./2D/%d/%d_weight.txt'%(num,i), 'w')
#             for j in range(len(weights)):
#                 f_out.write('%.4f\n' % weights[j])
#             f_out.close()
#         elif Dim ==3:
#

for i in tqdm(range(100)):
    g = gen_graph(NUM_MIN, NUM_MAX, g_type)
    if not os.path.exists('./validation/%s/%dD/%d_%d' % (g_type, Dim, NUM_MIN, NUM_MAX)):
        os.mkdir('./validation/%s/%dD/%d_%d' % (g_type, Dim, NUM_MIN, NUM_MAX))
    nx.write_gml(g, './validation/%s/%dD/%d_%d/%d.gml' % (g_type, Dim, NUM_MIN, NUM_MAX, i))
    print ('Graph %d finished!'%i)

# for i in tqdm(range(100)):
#     g = nx.read_gml('./Validation_Data/%d.gml' %(i))
#     num_node = nx.number_of_nodes(g)
#     num = int(pow(num_node, 1/3))
#     for node in g.nodes():
#         # node coords
#         x_y = int(int(node) % (num * num))
#         z = int(int(node) / (num * num))
#         x = int(x_y / num)
#         y = int(x_y % num)
#         g.nodes[node]['coords'] = (x, y, z)
#     if not os.path.exists('./Validation_Data_coords'):
#         os.mkdir('./Validation_Data_coords')
#     nx.write_gml(g, './Validation_Data_coords/%d.gml' % (i))
#     print ('Graph %d finished!'%i)