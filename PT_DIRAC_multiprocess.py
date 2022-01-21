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

def read_gurobi(lattice_dim, test_scale):
    Gurobi_res = {}
    num = 0
    gid = 0
    for line in open('../TestData/Gurobi_res/Lattice_%dD-%d_Gurobi.txt'%(lattice_dim, test_scale), 'r'):
        if gid < 1000:
            data = float(line.strip().split(',')[1].strip().split(':')[1])
            Gurobi_res[gid] = data
        gid += 1
    # print('Gurobi, lattice_dim:%d, test_scale:%d, read done!'%(lattice_dim, test_scale))
    return Gurobi_res

def getEnergy(g):
    energy = 0.0
    for edge in g.edges():
        energy += -g.nodes[edge[0]]['state'] * g.nodes[edge[1]]['state'] * g[edge[0]][edge[1]]['weight']
    return energy

def Randomize(g):
    for node in g.nodes():
        g.nodes[node]['state'] = 2 * np.random.randint(0, 2, 1)[0] - 1
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

def ExchangeMC(sg, g, nepochs, gid, fout, Gurobi_res):
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

    epoch_energy_list = []
    for epoch in range(nepochs):
        print ('gid:%d, epoch:%d'%(gid, epoch))
        start = time.time()
        for beta in betas[:-1]:  # 对每个复本进行标准MC步骤
            temp_energy = getEnergy(greps[beta])
            for MCstep in range(nMCsteps):
                node = np.random.randint(0, NNN, 1)[0]
                neibors = greps[beta][node]
                spin_energy = 0
                for neibor in neibors:
                    spin_energy += greps[beta][node][neibor]['weight'] * greps[beta].nodes[neibor]['state']
                spin_energy *= -greps[beta].nodes[node]['state']
                if_change = 0
                if spin_energy >= 0:
                    greps[beta].nodes[node]['state'] *= -1
                    if_change = 1
                elif spin_energy < 0:
                    rn = np.random.uniform(0, 1)
                    if rn < np.exp(beta * 2 * spin_energy):
                        greps[beta].nodes[node]['state'] *= -1
                        if_change = 1
                if if_change:
                    temp_energy = temp_energy - 2 * spin_energy
                    if temp_energy < lowest_energy:
                        lowest_energy = temp_energy
                        lowest_g = copy.deepcopy(greps[beta])

        # energy, greps[betas[-1]] = MultiQ(sg, greps[betas[-1]])
        energy, greps[betas[-1]] = OneStepQ(sg, greps[betas[-1]])
        if energy < lowest_energy:
            lowest_energy = energy
            lowest_g = copy.deepcopy(greps[betas[-1]])       

        end = time.time()
        fout.write('Graph: %d, epoch: %d, result: %.16f, time: %.4f\n'%(gid, epoch, -lowest_energy/len(g), end-start))
        fout.flush()

        if np.abs(-lowest_energy/len(g) - Gurobi_res[gid]) < 0.00000001:
            break
 
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
    
        # # res_epoch_list.append(-lowest_energy/len(g))
        # epoch_lowest_energy = getEnergy(greedyDescent(lowest_g))    
        # fout.write('Graph: %d, epoch: %d, result: %.4f, result_improve: %.4f\n'%(gid, epoch, -lowest_energy/len(g), -epoch_lowest_energy/len(g)))
        # fout.flush()

        # time_2 = time.time()
        # print ('Epoch: %d, result: %.4f, time_cost: %.2f'%(epoch, -lowest_energy/len(g), time_2-time_1))

    #energy, lowest_g = MultiQ(sg, lowest_g)
    #energy, greps[betas[-1]] = OneStepQ(sg, greps[betas[-1]])
    #if energy < lowest_energy:
    #    lowest_energy = energy
        #lowest_g = copy.deepcopy(greps[betas[-1]])
    
    real_lowest_energy = getEnergy(greedyDescent(lowest_g))    
    # res_epoch_list[-1] = -real_lowest_energy/len(g)
    time2 = time.time()
    time_cost = time2 - time1
    return -lowest_energy/len(g), -real_lowest_energy/len(g), time_cost, lowest_g
    
def OneStepQ(sg, g):
    step = np.max([int(0.01 * nx.number_of_nodes(g)), 1])  # step size
    init_states = [g.nodes[node]['state'] for node in range(len(g))]

    g_temp = copy.deepcopy(g)
    for node in g.nodes():
        g_temp.nodes[node]['state'] = 1
    for edge in g_temp.edges():
        src, tgt = edge[0], edge[1]
        g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

    energy, states, _, _  = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
    temp_states = []
    for node in range(len(g_temp)):
        temp_states.append(states[node]/init_states[node])
    init_states = temp_states

    for node in g.nodes():
        g.nodes[node]['state'] = init_states[node]
    #res_improve = LocalImprove(g, res_best*len(g))
    return -energy*len(g), g

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

        energy, states, _, _  = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        temp_states = []
        for node in range(len(g_temp)):
            temp_states.append(states[node]/init_states[node])
        init_states = temp_states

        if res_best < energy:
            res_best = energy
        else:
            stopCondition = True
    
    for node in g.nodes():
        g.nodes[node]['state'] = init_states[node]
    #res_improve = LocalImprove(g, res_best*len(g))

    return -res_best*len(g), g

def PTTest(lattice_dim, test_scale, nepochs, gid, Gurobi_res):
    sg = SGRL()

    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
    
    sg.LoadModel(model_file)

    fout = open('../TestData/PT_res/temp_data/PT_DIRAC/Lattice_%dD_num_%d_PT_DIRAC_nepochs_%d_gid_%d.txt'%(lattice_dim, test_scale, nepochs, gid), 'w')

    print ('lattice dim:%d, test scale:%d, gid:%d'%(lattice_dim, test_scale, gid))
    g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, test_scale, gid)
    G = nx.read_gml(g_file)
    g = nx.Graph()
    for edge in G.edges():
        g.add_edge(int(edge[0]), int(edge[1]))
        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
    for node in G.nodes():
        g.nodes[int(node)]['coords'] = G.nodes[node]['coords']

    best_energy, best_energy_improve, time_cost, lowest_g = ExchangeMC(sg, g, nepochs, gid, fout, Gurobi_res)
    # best_energy_improve = LocalImprove(real_lowest_g, best_energy*len(g))
    #for k in range(len(res_epoch_list)):
    #    fout.write('Graph: %d, epoch: %d, result: %.4f\n'%(gid, k, res_epoch_list[k]))
    #    fout.flush()
    # fout.write('Graph: %d, result: %.4f, result_improve: %.4f, time: %.2f\n'%(gid, best_energy, best_energy_improve, time_cost))
    # fout.close()
    fout.close()
    fout_state = open('../TestData/PT_res/temp_data/PT_DIRAC/Lattice_%dD_num_%d_PT_DIRAC_nepochs_%d_gid_%d_states.txt'%(lattice_dim, test_scale, nepochs, gid), 'w')
    for i in range(len(lowest_g)):
        fout_state.write('node: %d, state:%d\n'%(i, int(lowest_g.nodes[i]['state'])))
    fout_state.close()

if __name__=="__main__":

    args = parsers()
    Gurobi_res = read_gurobi(args.lattice_dim, args.test_scale)
    PTTest(args.lattice_dim, args.test_scale, args.numInits, args.gid, Gurobi_res)
