from SGRL import SGRL
import networkx as nx
import numpy as np
from tqdm import tqdm
import time
from config import parsers
import copy

args = parsers()

## help function
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
        prob = 0.5*(betas[0]-beta)/(betas[0]-betas[-1])
        if rn < prob:
            g.nodes[node]['state'] *= -1
    return g

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
    return -energy*len(g), g

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

def EnergyDescent(g, nMCsteps, temp_energy, beta):
    for MCstep in range(nMCsteps):
        node = np.random.randint(0, len(g.nodes), 1)[0]
        neibors = g[node]
        spin_energy = 0
        for neibor in neibors:
            spin_energy += g[node][neibor]['weight'] * g.nodes[neibor]['state']
        spin_energy *= -g.nodes[node]['state']
        if_change = 0
        if spin_energy >= 0:
            g.nodes[node]['state'] *= -1
            if_change = 1
        elif spin_energy < 0:
            rn = np.random.uniform(0, 1)
            if rn < np.exp(beta * 2 * spin_energy):
                g.nodes[node]['state'] *= -1
                if_change = 1
        if if_change:
            temp_energy = temp_energy - 2 * spin_energy
    return g, temp_energy

def DIRACDescent(sg, g, energy_prev, beta, betas):
    energy_current, g_current = OneStepQ(sg, g)
    delta_energy = energy_prev - energy_current
    if delta_energy > 0:
        g = g_current
    elif delta_energy <= 0:
        g = Randomize_beta(g, beta, betas)
    return g, energy_current

def ExchangeMC_alpha(sg, g, nepochs, gid, fout):
    time1 = time.time()
    betas = 1.0 / np.linspace(0.1, 1.6, 20)
    NNN = len(g.nodes)  # 系统大小
    nMCsteps = NNN  # 每隔多少步会进行置换操作
    m = len(betas)
    Randomize(g)
    lowest_energy = getEnergy(g)
    lowest_g = g
    greps = {}
    for beta in betas:
        greps[beta] = Randomize(g.copy())

    epoch_energy_list = []
    for epoch in range(nepochs):
        start = time.time()
        for beta in betas[:-1]: 
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

        energy, greps[betas[-1]] = OneStepQ(sg, greps[betas[-1]])
        if energy < lowest_energy:
            lowest_energy = energy
            lowest_g = copy.deepcopy(greps[betas[-1]])       

        end = time.time()
        fout.write('Lattice: %d, epoch: %d, result: %.16f, time: %.4f\n'%(gid, epoch, -lowest_energy/len(g), end-start))
        print('DIRAC_PT_alpha, gid: %d, epoch: %d, result: %.8f, time: %.2f'%(gid, epoch, -lowest_energy/len(g), end-start))
        fout.flush()

        rindex = np.random.randint(0, m - 1, 1)[0]
        left = betas[rindex]
        right = betas[rindex + 1]
        delta = -(left - right) * (getEnergy(greps[left]) - getEnergy(greps[right]))
        rn = np.random.uniform(0, 1)
        if delta <= 0 or rn < np.exp(-delta):  
            for node in greps[left].nodes:
                temp = greps[right].nodes[node]['state']
                greps[right].nodes[node]['state'] = greps[left].nodes[node]['state']
                greps[left].nodes[node]['state'] = temp
 
    real_lowest_energy = getEnergy(greedyDescent(lowest_g))    
    time2 = time.time()
    time_cost = time2 - time1
    return -lowest_energy/len(g), -real_lowest_energy/len(g), time_cost

def ExchangeMC_beta(sg, g, nepochs, gid, fout):
    time1 = time.time()
    betas = 1.0 / np.linspace(0.1, 1.6, 20)
    NNN = len(g.nodes)  # 系统大小
    nMCsteps = NNN  # 每隔多少步会进行置换操作
    m = len(betas)
    Randomize(g)
    lowest_energy = getEnergy(g)
    lowest_g = g
    greps = {}
    for beta in betas:
        greps[beta] = Randomize(g.copy())
    for epoch in range(nepochs):
        time_1 = time.time()
        for beta in betas:  # 
            energy_prev = getEnergy(greps[beta])
            if np.random.uniform(0, 1) < (betas[0]-beta)/(betas[0]-betas[-1]):
                greps[beta], energy_current = EnergyDescent(greps[beta], nMCsteps, energy_prev, beta)
            else:
                greps[beta], energy_current = DIRACDescent(sg, greps[beta], energy_prev, beta, betas)
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
        fout.write('Lattice: %d, epoch: %d, result: %.16f\n'%(gid, epoch, -lowest_energy/len(g)))
        fout.flush()
        time_2 = time.time()
        print ('DIRAC_PT_beta, gid: %d, epoch: %d, result: %.8f, time: %.2f'%(gid, epoch, -lowest_energy/len(g), time_2-time_1))
    time2 = time.time()
    time_cost = time2 - time1
    return -lowest_energy/len(g), -real_lowest_energy/len(g), time_cost

## different DIRAC test variants
def DIRAC_1():
    sg = SGRL()
    # model_file = sg.findModel()
    if args.lattice_dim == 2:
        model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
    elif args.lattice_dim == 3:
        model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
    elif args.lattice_dim == 4:
        model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
    sg.LoadModel(model_file)
    stepRatio = 0.01

    if args.lattice_dim == 2:
        scales = [15, 20, 25]
    elif args.lattice_dim == 3:
        scales = [6, 8, 10]
    elif args.lattice_dim == 4:
        scales = [4, 5, 6]

    for scale in scales:
        fout = open('./result/DIRAC_1_dim_%d_scale_%d.txt'%(args.lattice_dim, scale), 'w')
        result_list, result_improve_list = [], []
        time_list = []
        for i in tqdm(range(50)):
            g_file = './data/%dD/%d/%d.gml'%(args.lattice_dim, scale, i)
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
            fout.write('Lattice_dim:%d, test_scale:%d, gid: %d, result: %.4f, time:%.2f\n' % (args.lattice_dim, scale, i, res_improve, exec_time))
            print ('DIRAC_1, Lattice_dim:%d, test_scale:%d, Lattice:%d, result:%.4f, time:%.2f'%(args.lattice_dim, scale, i, res_improve, exec_time))
            fout.flush()
        fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
        fout.write('Result_improve:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
        fout.close()

def DIRAC_m():
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
        scales = [15, 20, 25]
    elif args.lattice_dim == 3:
        scales = [6, 8, 10]
    elif args.lattice_dim == 4:
        scales = [4, 5, 6]

    for scale in scales:
        fout = open('./result/DIRAC_m_dim_%d_scale_%d.txt'%(args.lattice_dim, scale), 'w')
        result_list, result_improve_list = [], []
        time_list = []
        for i in tqdm(range(50)):
            g_file = './data/%dD/%d/%d.gml'%(args.lattice_dim, scale, i)
            G = nx.read_gml(g_file)
            g = nx.Graph()
            for edge in G.edges():
                g.add_edge(int(edge[0]), int(edge[1]))
                g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
            for node in G.nodes():
                g.nodes[int(node)]['state'] = 1
                g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
            if stepRatio > 0:
                step = np.max([int(stepRatio * nx.number_of_nodes(g)), 1])
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
            fout.write('Lattice_dim:%d, test_scale:%d, gid: %d, result: %.4f, time:%.2f\n' % (args.lattice_dim, scale, i, res_improve, exec_time))
            print ('DIRAC_m, Lattice_dim:%d, test_scale:%d, Lattice:%d, result:%.4f, time:%.2f'%(args.lattice_dim, scale, i, res_improve, exec_time))
            fout.flush()
        fout.write('Result:%.4f+%.4f, Time:%.2f+%.2f\n' % (np.mean(result_list), np.std(result_list), np.mean(time_list), np.std(time_list)))
        fout.write('Result_improve:%.4f+%.4f\n' % (np.mean(result_improve_list), np.std(result_improve_list)))
        fout.close()

def DIRAC_PT_alpha():
	sg = SGRL()
	if args.lattice_dim == 2:
	    model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
	elif args.lattice_dim == 3:
	    model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
	elif args.lattice_dim == 4:
	    model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
	sg.LoadModel(model_file)
	fout = open('./result/DIRAC_PT_alpha_dim_%d_scale_%d.txt'%(args.lattice_dim, args.test_scale), 'w')
	print ('DIRAC_PT_alpha, lattice dim:%d, test scale:%d, gid:%d'%(args.lattice_dim, args.test_scale, args.gid))
	g_file = './data/%dD/%d/%d.gml' %(args.lattice_dim, args.test_scale, args.gid)
	G = nx.read_gml(g_file)
	g = nx.Graph()
	for edge in G.edges():
	    g.add_edge(int(edge[0]), int(edge[1]))
	    g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	for node in G.nodes():
	    g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
	ExchangeMC_alpha(sg, g, args.numInits, args.gid, fout)

def DIRAC_PT_beta():
	sg = SGRL()
	if args.lattice_dim == 2:
	    model_file = './models/2D/nrange_5_6_iter_287100.ckpt'
	elif args.lattice_dim == 3:
	    model_file = './models/3D/nrange_4_5_iter_332400.ckpt'
	elif args.lattice_dim == 4:
	    model_file = './models/4D/nrange_2_3_iter_49200.ckpt'
	sg.LoadModel(model_file)
	fout = open('./result/DIRAC_PT_beta_dim_%d_scale_%d.txt'%(args.lattice_dim, args.test_scale), 'w')
	print ('DIRAC_PT_beta, lattice dim:%d, test scale:%d, gid:%d'%(args.lattice_dim, args.test_scale, args.gid))
	g_file = './data/%dD/%d/%d.gml' % (args.lattice_dim, args.test_scale, args.gid)
	G = nx.read_gml(g_file)
	g = nx.Graph()
	for edge in G.edges():
	    g.add_edge(int(edge[0]), int(edge[1]))
	    g[int(edge[0])][int(edge[1])]['weight'] = G[edge[0]][edge[1]]['weight']
	for node in G.nodes():
	    g.nodes[int(node)]['coords'] = G.nodes[node]['coords']
	ExchangeMC_beta(sg, g, args.numInits, args.gid, fout)

if __name__=="__main__":

    # DIRAC_1()
    # DIRAC_m()
    # DIRAC_PT_alpha()
    # DIRAC_PT_beta()
