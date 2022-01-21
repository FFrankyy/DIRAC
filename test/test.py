# -*- coding: utf-8 -*-
from gurobipy import *
import networkx as nx
import numpy as np


def getGuroSol(g, writesol=False, gid=None):
    nodes = nx.nodes(g)
    edges = nx.edges(g)
    try:
        # Create a new model
        m = Model("Spin Glass")
        m.setParam('TimeLimit', 3600)  # 设置运行时间
        # m.Params.timelimit = 3600
        # m.Params.MIPFocus = 1
        m.setParam('Threads', 10)  # 设置线程
        # Create variables
        nodesVar = {}
        for i, node in enumerate(nodes):
            nodesVar[i] = m.addVar(vtype=GRB.BINARY, name=str(node))
        # Set objective
        m.setObjective(
            quicksum([g[edge[0]][edge[1]]['weight'] * (2 * nodesVar[edge[0]] - 1) * (2 * nodesVar[edge[1]] - 1) for
                      edge in edges]), GRB.MAXIMIZE)
        # optimize
        m.optimize()
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    if writesol:
        f_out = open('./sol/%d.txt'%gid, 'w')
        for i in range(len(nodes)):
            f_out.write('Node:%d, SG:%d\n'%(i, 2*nodesVar[i].x-1))
    return m.objVal


f_out = open('./3D/reslut_gurobi3.txt', 'w')
for i in range(20, 30):
    G = nx.read_gml('./3D/%d.gml'%i)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    result = getGuroSol(g, writesol=False, gid=None)
    f_out.write('gid:%d, -%.4f\n'%(i,result))
    f_out.flush()
f_out.close()