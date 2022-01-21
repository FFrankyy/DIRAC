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
        prob = 0.3*(betas[0]-beta)/(betas[0]-betas[-1]) + 0.2
        if rn < prob:
            g.nodes[node]['state'] *= -1
    return g

def greedyDescent(g):
    if_can_lower = 1
    while if_can_lower:
        if_can_lower = 0
        for node in g.nodes():
            neibors = g[node]
            delta_energy = 0
            for neibor in neibors:
                delta_energy += - g.nodes[node]['state'] * g.nodes[neibor]['state'] * g[node][neibor]['weight']
            if delta_energy > 0:
                if_can_lower = 1
                g.nodes[node]['state'] *= -1
    return g

def ExchangeMC(sg, g, nepochs, gid, fout):
    time1 = time.time()
    # betas=np.linspace(0.1, 1.6, 30)
    betas = 1.0 / np.linspace(0.1, 1.6, 20)
    NNN = len(g.nodes)  # 系统大小
    nMCsteps = NNN  # 每隔多少步会进行置换操作
    # nepochs = int(100)  # 进行操作组的数量
    m = len(betas)
    Randomize(g)
    # res_epoch_list = []
    lowest_energy = getEnergy(g)
    lowest_g = g
    greps = {}
    for beta in betas:
        greps[beta] = Randomize(g.copy())
    for epoch in range(nepochs):
        time_1 = time.time()
        for beta in betas:  # 对每个复本进行标准MC步骤
            energy_prev = getEnergy(greps[beta])
            energy_current, g_current = OneStepQ(sg, greps[beta])
            delta_energy = energy_prev - energy_current
            # print ('energy_prev: %.4f, energy_current: %.4f, delta_energy: %.4f'%(energy_prev, energy_current, delta_energy))
            if delta_energy > 0:
                greps[beta] = g_current
                # if_change = 1
            elif delta_energy <= 0:
                greps[beta] = Randomize_beta(g.copy(), beta, betas)
                # rn = np.random.uniform(0, 1)
                # if rn < np.exp(beta * delta_energy):
                #     greps[beta] = Randomize(g.copy())
            # if if_change:
            if energy_current < lowest_energy:
                lowest_energy = energy_current
                lowest_g = copy.deepcopy(greps[beta])
        rindex = np.random.randint(0, m - 1, 1)[0]
        left = betas[rindex]
        right = betas[rindex + 1]
        delta = -(left - right) * (getEnergy(greps[left]) - getEnergy(greps[right]))
        rn = np.random.uniform(0, 1)
        if delta <= 0 or rn < np.exp(-delta):  # delta作为是否交换复本的判据
            for node in greps[left].nodes:
                temp = greps[right].nodes[node]['state']
                greps[right].nodes[node]['state'] = greps[left].nodes[node]['state']
                greps[left].nodes[node]['state'] = temp
    
        # res_epoch_list.append(-lowest_energy/len(g))
        fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(gid, epoch, -lowest_energy/len(g)))
        fout.flush()

        time_2 = time.time()
        print ('epoch: %d, time_cost: %.2f, result: %.4f'%(epoch, time_2-time_1, -lowest_energy/len(g)))

    real_lowest_energy = getEnergy(greedyDescent(lowest_g))    
    # res_epoch_list[-1] = -real_lowest_energy/len(g)
    fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(gid, epoch+1, -real_lowest_energy/len(g)))
    fout.flush()
    time2 = time.time()
    time_cost = time2 - time1
    return -lowest_energy/len(g), -real_lowest_energy/len(g), time_cost
    
def OneStepQ(sg, g):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
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
    #res_improve = LocalImprove(g, res_best*len(g))
    return -energy*len(g), g_update

def MultiQ(sg, g):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    init_states = [g.nodes[node]['state'] for node in range(len(g))]
    stopCondition = False  # whether to stop
    res_best = -1000000
    num_step = 0
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

        if res_best < energy:
            res_best = energy
        else:
            stopCondition = True
    
    g_update = copy.deepcopy(g)
    for node in g.nodes():
        g_update.nodes[node]['state'] = init_states[node]
    return -res_best*len(g), g_update

def PTTest(lattice_dim, test_scale, nepochs, gid):
    sg = SGRL()
    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
    sg.LoadModel(model_file)

    fout = open('../TestData/PT_res/temp_data/Lattice_%dD_num_%d_RLPT_RandomBeta_nepochs_%d_gid_%d.txt'%(lattice_dim, test_scale, nepochs, gid), 'w')

    print ('lattice dim:%d, test scale:%d, gid:%d'%(lattice_dim, test_scale, gid))
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    best_energy, best_energy_improve, time_cost, res_epoch_list = ExchangeMC(sg, g, nepochs, gid, fout)
    # best_energy_improve = LocalImprove(real_lowest_g, best_energy*len(g))
    # for k in range(len(res_epoch_list)):
    #     fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(i, k, res_epoch_list[k]))
    #     fout.flush()
    fout.write('Time cost:%.4f'%(time_cost))
    fout.close()

if __name__=="__main__":

    args = parsers()
    PTTest(args.lattice_dim, args.test_scale, args.numInits, args.gid)
