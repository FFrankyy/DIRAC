from SGRL import SGRL
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
from config import parsers
import copy

args = parsers()

def Train():
    sg = SGRL()
    sg.Train()

def MetaQ():
    sg = SGRL()
    # model_file = sg.findModel()
    if args.lattice_dim == 2:
        #model_file = './models/Lattice_2D_5_6_noPE/nrange_5_6_iter_229500.ckpt'
        model_file = './models/Lattice_2D_5_6_noPE/nrange_5_6_iter_230100.ckpt'
        #model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        #model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
        model_file = './models/Lattice_3D_4_5_noPE/nrange_4_5_iter_153300.ckpt'
    elif args.lattice_dim == 4:
        #model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
        #model_file = './models/Lattice_4D_2_3_noPE/nrange_2_3_iter_82200.ckpt'
        model_file = './models/Lattice_4D_2_3_noPE/nrange_2_3_iter_51600.ckpt'
    #print ('best model is:')
    #print (model_file)
    sg.LoadModel(model_file)
    # spin scale, stepRatio
    stepRatio_list = [0.01]
    #stepRatio = 0.01

    if args.lattice_dim == 2:
        scales = [25, 30, 35]
    elif args.lattice_dim == 3:
        scales = [10,15,20]
    elif args.lattice_dim == 4:
        scales = [6, 7, 8]

    for stepRatio in stepRatio_list:
	    for num in scales:
	        fout = open('../TestData/DQN_res/%s_num_%d_stepRatio_%.4f_MetaQ.txt'%(args.model_name, num, stepRatio), 'w')
	        result_list, result_improve_list = [], []
	        time_list = []
	        for i in tqdm(range(50)):
	            g_file = '../TestData/Lattice/%dD/%d/%d.gml'%(args.lattice_dim, num, i)
	            G = nx.read_gml(g_file)
	            g = nx.Graph()
	            for edge in G.edges():
	                g.add_edge(int(edge[0]), int(edge[1]))
	                g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	            for node in G.nodes():
	                g.nodes[int(node)]['state'] = 1
	                # g.nodes[int(node)]['state'] = int(np.random.choice([-1,1], 1))
	                g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
	            if stepRatio > 0:
	                step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
	            else:
	                step = 1
	            time1 = time.time()
	            res, states, sol, _  = sg.Evaluate(g, isNetworkx=1, step=step)
	            res_improve, _ = LocalImprove(g, states, res*len(g))
	            time2 = time.time()
	            exec_time = time2 - time1
	            result_list.append(res)
	            result_improve_list.append(res_improve)
	            time_list.append(exec_time)
	            fout.write('Graph: %d, result: %.4f, result_improve: %.4f\n' % (i, res, res_improve))
	            print ('Lattice_dim:%d, test_scale:%d, Graph:%d, StepRatio:%.4f, result:%.4f, time:%.2f'%(args.lattice_dim, num, i, stepRatio, res_improve, exec_time))
	            fout.flush()
	        fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
	        fout.write('Result_improve:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
	        fout.close()

def MultiQ():
    sg = SGRL()
    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
    sg.LoadModel(model_file)
    stepRatio = 0.01

    if args.lattice_dim == 2:
        scales = [25, 30, 35]
    elif args.lattice_dim == 3:
        scales = [20]
    elif args.lattice_dim == 4:
        scales = [6, 7, 8]

    for num in scales:
        fout = open('../TestData/DQN_res/%s_num_%d_stepRatio_%.4f_MultiQ_GPU.txt'%(args.model_name, num, stepRatio), 'w')
        result_list, result_improve_list = [], []
        time_list = []
        for i in tqdm(range(50)):
            g_file = '../TestData/Lattice/%dD/%d/%d.gml'%(args.lattice_dim, num, i)
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
            init_states = [1 for i in range(len(g))]
            stopCondition = False  # whether to stop
            energy_best = -1000000
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
                if energy_best < energy:
                    energy_best = energy
                else:
                    stopCondition = True
            res_improve, _ = LocalImprove(g, init_states, energy_best*len(g))
            time2 = time.time()

            exec_time = time2 - time1
            result_list.append(energy_best)
            result_improve_list.append(res_improve)
            time_list.append(exec_time)
            fout.write('Graph: %d, result: %.4f, result_improve: %.4f\n' % (i, energy_best, res_improve))
            print ('Lattice_dim:%d, test_scale:%d, Graph:%d, result:%.4f, time:%.2f'%(args.lattice_dim, num, i, res_improve, exec_time))
            fout.flush()
        fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
        fout.write('Result_improve:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
        fout.close()


def EvaluateMultiInit(numInits):
    sg = SGRL()
    # model_file = sg.findModel()
    model_file = './models/3D/nrange_4_5_iter_416700.ckpt'
    #print ('best model is:')
    #print (model_file)
    sg.LoadModel(model_file)
    # spin scale, stepRatio
    stepRatio = 0.01
    for num in [4]:
        with open('../TestData/DQN_res/%s_num_%d_stepRatio_%.4f_%dInits.txt'%(args.model_name, num, stepRatio, numInits), 'w') as fout:
            result_list, result_improve_list = [], []
            time_list = []
            for i in tqdm(range(50)):
                g_file = '../TestData/Lattice/%dD/%d/%d.gml'%(args.lattice_dim, num, i)
                G = nx.read_gml(g_file)
                g = nx.Graph()
                for edge in G.edges():
                    g.add_edge(int(edge[0]), int(edge[1]))
                    g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
                for node in G.nodes():
                    g.nodes[int(node)]['state'] = G.nodes[node]['state']
                    # g.nodes[int(node)]['state'] = int(np.random.choice([-1,1], 1))
                    g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
                if stepRatio > 0:
                    step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
                else:
                    step = 1

                time1 = time.time()
                res_best, res_improve_best = -9999999, -99999999
                for init_num in range(numInits):
                    g_temp = copy.deepcopy(g)
                    random_init = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]
                    for edge in g_temp.edges():
                        src, tgt = edge[0], edge[1]
                        g_temp[src][tgt]['weight'] *= random_init[src]*random_init[tgt]
                    res, states, sol = sg.Evaluate(g_temp, isNetworkx=1, step=step)
                    res_improve, _ = LocalImprove(g_temp, states, res*len(g_temp))
                    if res_improve_best < res_improve:
                        res_best = res
                        res_improve_best = res_improve
                time2 = time.time()
                exec_time = time2 - time1
                result_list.append(res_best)
                result_improve_list.append(res_improve_best)
                time_list.append(exec_time)
                fout.write('Graph: %d, result: %.4f, improved result: %.4f, time:%.2f\n' % (i, res_best, res_improve_best, exec_time))
                fout.flush()
            fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
            # print ('%s, num:%d, mean:%.4f, mean_improve:%.4f, time:%.4f'%(args.model_name, num, np.mean(result_list),np.mean(result_improve_list), np.mean(time_list)))
            fout.write('Result:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
            fout.flush()

def LocalSearchNode():
    for num in [4,6,8,10,15,20]:#8, 9, 10, 30, 50
        with open('../TestData/Greedy_res/%s_%d_LocalSearch_numInits1.txt'%(args.model_name, num), 'w') as fout:
            for numInits in [1]:
                # # baseline
                # best_res = []
                # for line in open('../TestData/3D/%d/%d_best.txt' % (num, num), 'r'):
                #     best_res.append(float(line.strip().split()[0]))
                # read graph
                # appro_ratio = []
                result_list, time_list = [], []
                for i in tqdm(range(50)):
                    g_file = '../TestData/Lattice/3D/%d/%d.gml' % (num, i)
                    G = nx.read_gml(g_file)
                    g = nx.Graph()
                    for edge in G.edges():
                        g.add_edge(int(edge[0]), int(edge[1]))
                        g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']

                    best_energy = -1000000.0
                    time1 = time.time()
                    for iter in range(numInits):
                        if numInits == 1:
                            states = [1 for i in range(len(g))]  # random
                        else:
                            states = [int(np.random.choice([-1,1], 1)) for i in range(len(g))] # random
                        energy = 0.0
                        for edge in g.edges():
                            energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']

                        stopCondition = False   # for each state, run multiple times to obtain the local optimal
                        while not stopCondition:
                            best_delta, best_node = 0.0, 0
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
                                stopCondition = True # no other flips can increase the energy

                        if best_energy < energy:
                            best_energy = energy
                    best_energy /= len(g)
                    time2 = time.time()
                    result_list.append(best_energy)
                    time_list.append(time2 - time1)
                    fout.write('Graph: %d, result: %.4f, time:%.4f\n' % (i, best_energy, time2-time1))
                    fout.flush()
                fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
                fout.flush()

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
    return energy/len(g), states

def MultipleInitialState():
    sg = SGRL()
    model_file = './models/Model_6_7_Dim_2/nrange_6_7_iter_21000.ckpt'
    sg.LoadModel(model_file)
    num, stepRatio = 10, 0
    # baseline
    best_res = []
    for line in open('./test/2D/%d/%d_best.txt' % (num, num), 'r'):
        best_res.append(float(line.strip().split()[0]))
    # read graph
    appro_ratio = []
    for i in tqdm(range(100)):
        energy_best = -10000
        print ('###############')
        for j in range(30):
            g_file = './test/2D/%d/%d.gml' % (num, i)
            G = nx.read_gml(g_file)
            g = nx.Graph()
            for edge in G.edges():
                g.add_edge(int(edge[0]), int(edge[1]))
                g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
            if stepRatio > 0:
                step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
            else:
                step = 1

            energy, states = sg.Evaluate(g, step)  # Q result, run multiple times, each time, different graph states
            if energy_best < energy:
                energy_best = energy
            # stopCondition = False  # whether to stop
            # while not stopCondition:
            #     best_delta, best_node = 0, 0
            #     for node in g.nodes:
            #         delta = 0
            #         for neigh in g.neighbors(node):
            #             delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
            #         if delta >= 0 and best_delta < delta:
            #             best_delta = delta
            #             best_node = node
            #     states[best_node] *= -1
            #     energy += best_delta
            #     if best_delta == 0:
            #         stopCondition = True
        appro_ratio.append(energy_best / best_res[i])
    print (np.mean(appro_ratio))
    print (np.std(appro_ratio))

# (RL+Greedy)^(30) + numInits
def RLStrategy(numInits):
    sg = SGRL()
    model_file = './models/%dD/nrange_4_5_iter_416700.ckpt'%args.lattice_dim
    # model_file = './models/%dD/nrange_2_3_iter_49800.ckpt'%args.lattice_dim
    sg.LoadModel(model_file)
    stepRatio = 0.01
    for num in [4]:
        fout =  open('../TestData/DQN_res/%s_num_%d_stepRatio_%.4f_RL_strategy_30MultiQ_%dInits.txt'%(args.model_name, num, stepRatio, numInits), 'w')
        result_list, result_improve_list = [], []
        time_list = []
        # read graph
        for i in tqdm(range(50)):
            g_file = '../TestData/Lattice/%dD/%d/%d.gml' % (args.lattice_dim, num, i)
            G = nx.read_gml(g_file)
            g = nx.Graph()
            for edge in G.edges():
                g.add_edge(int(edge[0]), int(edge[1]))
                g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
            for node in G.nodes():
                g.nodes[int(node)]['state'] = G.nodes[node]['state']
                g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
            if stepRatio > 0:
                step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])  # step size
            else:
                step = 1

            res_best, res_improve_best = -9999999, -99999999

            time1 = time.time()
            for init_num in range(numInits):
                g_temp = copy.deepcopy(g)
                if numInits == 1:
                    init_states = [1 for i in range(len(g_temp))]
                else:
                    init_states = [int(np.random.choice([-1,1], 1)) for i in range(len(g_temp))]

                energy_best, energy_improve_best = -1000000, -1000000

                # # continue DQN for 30 times, recode the best during the process
                # for iter_num in range(30):
                #     g_temp = copy.deepcopy(g)
                #     for node in g.nodes():
                #         g_temp.nodes[node]['state'] = 1
                #     for edge in g_temp.edges():
                #         src, tgt = edge[0], edge[1]
                #         g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]
                #     res, states, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
                #     temp_states = []
                #     for node in range(len(g_temp)):
                #         temp_states.append(states[node]/init_states[node])
                #     init_states = temp_states
                #     res_improve, temp_states = LocalImprove(g, init_states, res*len(g))
                #     init_states = temp_states
                #     if energy_improve_best < res_improve:
                #         energy_best = res
                #         energy_improve_best = res_improve
                stopCondition = False  # whether to stop
                energy_best, energy_improve_best = -1000000, -1000000
                while not stopCondition:
                    g_temp = copy.deepcopy(g)
                    for node in g.nodes():
                        g_temp.nodes[node]['state'] = 1
                    for edge in g_temp.edges():
                        src, tgt = edge[0], edge[1]
                        g_temp[src][tgt]['weight'] *= init_states[src] * init_states[tgt]

                    res, states, _ = sg.Evaluate(g_temp, isNetworkx=1, step=step)  # Q result
                    temp_states = []
                    for node in range(len(g_temp)):
                        temp_states.append(states[node]/init_states[node])
                    init_states = temp_states
                    res_improve, temp_states = LocalImprove(g, init_states, res*len(g))
                    init_states = temp_states

                    if energy_improve_best < res_improve:
                        energy_best = res
                        energy_improve_best = res_improve
                    else:
                        stopCondition = True

                if res_improve_best < energy_improve_best:
                    res_best = energy_best
                    res_improve_best = energy_improve_best

            time2 = time.time()
            exec_time = time2 - time1
            result_list.append(res_best)
            result_improve_list.append(res_improve_best)
            time_list.append(exec_time)
            fout.write('Graph: %d, result: %.4f, improved result: %.4f, time:%.2f\n' % (i, res_best, res_improve_best, exec_time))
            fout.flush()
        fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
        fout.write('Result:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
        fout.flush()
        fout.close()



if __name__=="__main__":

    #Train()
    # MetaQ()
    MultiQ()
