# -*- coding: utf-8 -*-
from gurobipy import *
import networkx as nx
import numpy as np
import os


def getGuroSol(g, writesol=True, gid=None, path=None):
    nodes = nx.nodes(g)
    edges = nx.edges(g)
    try:
        # Create a new model
        m = Model("Spin Glass")
        m.setParam('TimeLimit', 3600)  # 设置运行时间
        # m.Params.timelimit = 3600
        # m.Params.MIPFocus = 1
        m.setParam('Threads', 2)  # 设置线程
        # Create variables
        nodesVar = {}
        for i, node in enumerate(nodes):
            nodesVar[i] = m.addVar(vtype=GRB.BINARY, name=str(node))
        # Set objective
        m.setObjective(quicksum([g[edge[0]][edge[1]]['weight'] * (2 * nodesVar[edge[0]] - 1) * (2 * nodesVar[edge[1]] - 1) for edge in edges]), GRB.MAXIMIZE)
        # optimize
        m.optimize()
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    if writesol:
        f_out = open('./%s/%d_sol.txt'%(path, gid), 'w')
        for i in range(len(nodes)):
            f_out.write('%d,%d\n'%(i, 2*nodesVar[i].x-1))
    return m.objVal

NUM = 8
Dim = 3
g_type = 'lattice'

path = '../../TestData/Lattice/%dD/%d'%(Dim, NUM)
file_path = '%s/Gurobi_sol'%path
if not os.path.exists(file_path):
    os.mkdir(file_path)

# result_list = []
for i in range(100):
    print ('gid:%d'%i)
    G = nx.read_gml('%s/%d.gml'%(path, i))
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    result = getGuroSol(g, writesol=True, gid=i, path=file_path)
    # result_list.append(result)
    # f_out.write('%.4f\n'%(result))
    # f_out.flush()
# f_out.write('%.4f+%.4f\n'%(np.mean(result_list), np.std(result_list)))
# f_out.close()