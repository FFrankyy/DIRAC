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

args = parsers()

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

def MetaQ(model_file, file_path, lattice_dim, lattice_scale, stepRatio):
    sg = SGRL()
    #model_file = sg.findModel()
    sg.LoadModel(model_file)
    uid = uuid.uuid4()
    fout = open('%s/Lattice_%dD_num_%d_stepRatio_%.4f_MetaQ_%s.txt'%(file_path, lattice_dim, lattice_scale, stepRatio, uid), 'w')
    # result_list, result_improve_list, time_list = [], [], []
    # read graph
    # for i in tqdm(range(1000)):
    for i in [186]:
        g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, lattice_scale, i)
        G = nx.read_gml(g_file)
        g = nx.Graph()
        for edge in G.edges():
            g.add_edge(int(edge[0]), int(edge[1]))
            g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
        for node in G.nodes():
            g.nodes[int(node)]['state'] = 1
            g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
        if stepRatio > 0:
            step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
        else:
            step = 1
        init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g))] # random
        g_temp = copy.deepcopy(g)
        for node in g.nodes():
            g_temp.nodes[node]['state'] = 1
        for edge in g_temp.edges():
            src, tgt = edge[0], edge[1]
            g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]
        start = time.time()
        energy, states, sol, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
        end = time.time()
        init_states = [states[node]/init_states[node] for node in range(len(g_temp))]
        res_improve = LocalImprove(g, init_states, energy*len(g))
        fout.write('Graph: %d, result: %.16f, result_improve: %.16f, time: %.2f\n' % (i, energy, res_improve, end-start))
        print('Graph: %d, result: %.16f, result_improve: %.16f, time: %.2f\n' % (i, energy, res_improve, end-start))
        fout.flush()
    fout.close()

def MultiQ(model_file, file_path, lattice_dim, lattice_scale, stepRatio, numInits=1):
    sg = SGRL()
    #model_file = sg.findModel()
    sg.LoadModel(model_file)
    uid = uuid.uuid4()
    fout = open('%s/Lattice_%dD_num_%d_stepRatio_%.4f_MultiQ_%s.txt'%(file_path, lattice_dim, lattice_scale, stepRatio, uid), 'w')
    # read graph
    for i in tqdm(range(50)):
    # for i in [110, 607, 629, 745, 764]:
        g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (lattice_dim, lattice_scale, i)
        G = nx.read_gml(g_file)
        g = nx.Graph()
        for edge in G.edges():
            g.add_edge(int(edge[0]), int(edge[1]))
            g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
        for node in G.nodes():
            g.nodes[int(node)]['state'] = 1
            g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
        if stepRatio > 0:
            step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
        else:
            step = 1

        time1 = time.time()
        res_best, res_improve_best = -9999999, -99999999
        for init_num in range(numInits):
            g_temp = copy.deepcopy(g)
            init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]
            stopCondition = False  # whether to stop
            energy_best = -1000000
            num_step = 0
            while not stopCondition:
                g_temp = copy.deepcopy(g)
                for node in g.nodes():
                    g_temp.nodes[node]['state'] = 1
                for edge in g_temp.edges():
                    src, tgt = edge[0], edge[1]
                    g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]
 
                time_start = time.time()
                energy, states, _, _  = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
                time_end = time.time()
                #print ('step:%d, energy:%.4f, time_cost:%.2f'%(num_step, energy, time_end-time_start))
                num_step += 1

                temp_states = []
                for node in range(len(g_temp)):
                    temp_states.append(states[node]/init_states[node])
                init_states = temp_states

                if energy_best < energy:
                    energy_best = energy
                else:
                    stopCondition = True

            res_improve = LocalImprove(g, init_states, energy_best*len(g))
            # record the best results
            if res_improve_best < res_improve:
                res_best = energy_best
                res_improve_best = res_improve

        time2 = time.time()
        exec_time = time2 - time1
        fout.write('Graph: %d, result: %.16f, improved result: %.16f, time:%.2f, Q_nums:%d\n' % (i, res_best, res_improve_best, exec_time, num_step-1))
        print('Graph: %d, result: %.8f, improved result: %.8f, time:%.2f, Q_nums:%d\n' % (i, res_best, res_improve_best, exec_time, num_step-1))
        fout.flush()
    fout.close()


if __name__=="__main__":

    # model_name = 'Lattice_3D_4_5'
    # model_name = 'Lattice_4D_2_3'
    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'

    # lattice_dim = 4
    # lattice_scale = 4

    file_path = '../TestData/DQN_res/temp_data/MetaQ/%dD/%d'%(args.lattice_dim, args.test_scale)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    MetaQ(model_file, file_path, args.lattice_dim, args.test_scale, args.stepRatio)
    # MultiQ(model_file, file_path, args.lattice_dim, args.test_scale, args.stepRatio, numInits=1)
