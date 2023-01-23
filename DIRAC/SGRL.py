
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 00:33:33 2017

@author: fanchangjun
"""

from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import networkx as nx
import random
import time
import sys
from tqdm import tqdm
import PrepareBatchGraph
import nstep_replay_mem
import maxcut_env
import copy
import graph
import os
from config import parsers

args = parsers()

# random Ising model: maximize (\sum_\sigma_i*\sigma_j*S_ij - \sum h_i * S_i)
class SGRL:

    def __init__(self):
        # init some parameters
        self.TrainSet = graph.py_GSet()
        self.TestSet = graph.py_GSet()
        self.inputs = dict()
        self.IsDoubleDQN = False
        self.IsHuberLoss = False
        self.ngraph_train = 0
        self.ngraph_test = 0
        #Simulator
        self.env_list=[]
        self.g_list=[]
        self.covered=[]
        self.pred=[]
        # self.nStepReplayMem=NStepReplayMem(MEMORY_SIZE)
        self.nStepReplayMem = nstep_replay_mem.py_NStepReplayMem(args.memory_size)

        for i in range(args.num_env):
            self.env_list.append(maxcut_env.py_MaxcutEnv())
            self.covered.append(-1)
            self.g_list.append(graph.py_Graph())

        self.test_env = maxcut_env.py_MaxcutEnv()

        # [batch_size, node_cnt]
        self.action_select = tf.sparse_placeholder(tf.float32, name="action_select")
        # [node_cnt, batch_size]
        self.rep_global = tf.sparse_placeholder(tf.float32, name="rep_global")
        # [edge_cnt, node_cnt]
        self.n2esum_param = tf.sparse_placeholder(tf.float32, name="n2esum_param")
        # [node_cnt, edge_cnt]
        self.e2nsum_param = tf.sparse_placeholder(tf.float32, name="e2nsum_param")
        # [batch_size, node_cnt]
        self.subgsum_param = tf.sparse_placeholder(tf.float32, name="subgsum_param")
        # [batch_size,1]
        self.label = tf.placeholder(tf.float32, name="label")

        # [node_cnt, 2]
        self.node_input = tf.placeholder(tf.float32, name="node_input")
        # [edge_cnt, 4]
        self.edge_input = tf.placeholder(tf.float32, name="edge_input")

        # init Q network
        self.loss,self.trainStep,self.q_pred, self.q_on_all, self.Q_param_list = self.BuildNet()
        #init Target Q Network
        self.lossT, self.trainStepT, self.q_predT, self.q_on_allT, self.Q_param_listT = self.BuildNet()

        #takesnapsnot
        self.copyTargetQNetworkOperation = [a.assign(b) for a,b in zip(self.Q_param_listT,self.Q_param_list)]
        self.UpdateTargetQNetwork = tf.group(*self.copyTargetQNetworkOperation)
        # saving and loading networks
        self.saver = tf.train.Saver(max_to_keep=None)

        config = tf.ConfigProto(device_count={"CPU": 8},  # limit to num_cpu_core CPU usage
                                inter_op_parallelism_threads=100,
                                intra_op_parallelism_threads=100,
                                log_device_placement=False)

        config.gpu_options.allow_growth = True
        self.session = tf.InteractiveSession(config = config)
        self.session.run(tf.global_variables_initializer())


#################################################New code for graphDQN#####################################
    def BuildNet(self):
        ############## params define
        # [node_dim, embed_dim]
        w_n2l = tf.Variable(tf.truncated_normal([args.lattice_dim, args.embed_dim], stddev=0.01), tf.float32)
        # [edge_dim, embed_dim]
        w_e2l = tf.Variable(tf.truncated_normal([args.edge_feat_init, args.embed_dim], stddev=0.01), tf.float32)
        # [embed_dim, embed_dim]
        p_node_conv1 = tf.Variable(tf.truncated_normal([args.embed_dim, args.embed_dim], stddev=0.01),tf.float32)
        # [embed_dim, embed_dim]
        trans_node_1 = tf.Variable(tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01),tf.float32)
        trans_node_2 = tf.Variable(tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01),tf.float32)

        # [embed_dim, embed_dim]
        trans_edge_1 = tf.Variable(tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01),tf.float32)
        trans_edge_2 = tf.Variable(tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01),tf.float32)

        w_l = tf.Variable(tf.truncated_normal([2*args.embed_dim, args.embed_dim], stddev=0.01), tf.float32)

        # concat and MLP
        # [2*embed_dim, reg_hidden]
        h11_weight = tf.Variable( tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01), tf.float32)
        h12_weight = tf.Variable( tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01), tf.float32)
        # [reg_hidden, 1]
        h2_weight = tf.Variable( tf.truncated_normal([args.embed_dim, args.reg_hidden], stddev=0.01), tf.float32)
        last_w = tf.Variable(tf.truncated_normal([args.reg_hidden, 1], stddev=0.01), tf.float32)

        ############# get embeddings
        # [node_cnt, node_dim] * [node_dim, embed_dim] = [node_cnt, embed_dim], no sparse
        node_init = tf.matmul(tf.cast(self.node_input, tf.float32), w_n2l)
        cur_node_embed = tf.nn.relu(node_init)
        cur_node_embed = tf.nn.l2_normalize(cur_node_embed, axis=1)

        # [edge_cnt, edge_dim] * [edge_dim, embed_dim] = [edge_cnt, embed_dim]
        edge_init = tf.matmul(tf.cast(self.edge_input, tf.float32), w_e2l)
        cur_edge_embed = tf.nn.relu(edge_init)
        cur_edge_embed = tf.nn.l2_normalize(cur_edge_embed, axis=1)

        lv = 0
        while lv < args.max_bp_iter:
            cur_node_embed_prev = cur_node_embed
            lv = lv + 1
            ###################### update edges ####################################
            # [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim]
            msg_linear_node = tf.matmul(cur_node_embed, p_node_conv1)
            # [edge_cnt, node_cnt] * [node_cnt, embed_dim] = [edge_cnt, embed_dim]
            n2e = tf.sparse_tensor_dense_matmul(tf.cast(self.n2esum_param, tf.float32), msg_linear_node)
            # [edge_cnt, embed_dim] + [edge_cnt, embed_dim] = [edge_cnt, embed_dim]
            # n2e_linear = tf.add(n2e, edge_init)
            n2e_linear = tf.concat([tf.matmul(n2e, trans_edge_1), tf.matmul(edge_init, trans_edge_2)], axis=1)    #we used
            # n2e_linear = tf.concat([tf.matmul(n2e, trans_edge_1), tf.matmul(cur_edge_embed, trans_edge_2)], axis=1)    #we used
            # n2e_linear = tf.concat([n2e, edge_init], axis=1)    # [edge_cnt, 2*embed_dim]
            # [edge_cnt, embed_dim]
            cur_edge_embed = tf.nn.relu(n2e_linear)
            ### if MLP
            # cur_edge_embed_temp = tf.nn.relu(tf.matmul(n2e_linear, trans_edge_1))   #[edge_cnt, embed_dim]
            # cur_edge_embed = tf.nn.relu(tf.matmul(cur_edge_embed_temp, trans_edge_2))   #[edge_cnt, embed_dim]
            cur_edge_embed = tf.nn.l2_normalize(cur_edge_embed, axis=1)

            ###################### update nodes ####################################
            # msg_linear_edge = tf.matmul(cur_edge_embed, p_node_conv2)
            # [node_cnt, edge_cnt] * [edge_cnt, embed_dim] = [node_cnt, embed_dim]
            e2n = tf.sparse_tensor_dense_matmul(tf.cast(self.e2nsum_param, tf.float32), cur_edge_embed)
            # [node_cnt, embed_dim] * [embed_dim, embed_dim] + [node_cnt, embed_dim] * [embed_dim, embed_dim] = [node_cnt, embed_dim]
            # node_linear = tf.add(tf.matmul(e2n, trans_node_1), tf.matmul(cur_node_embed, trans_node_2))
            node_linear = tf.concat([tf.matmul(e2n, trans_node_1), tf.matmul(cur_node_embed, trans_node_2)], axis=1)    #we used
            # node_linear = tf.concat([e2n, cur_node_embed], axis=1)  #[node_cnt, 2*embed_dim]
            # [node_cnt, embed_dim]
            cur_node_embed = tf.nn.relu(node_linear)
            ## if MLP
            # cur_node_embed_temp = tf.nn.relu(tf.matmul(node_linear, trans_node_1))  # [node_cnt, embed_dim]
            # cur_node_embed = tf.nn.relu(tf.matmul(cur_node_embed_temp, trans_node_2))   # [node_cnt, embed_dim]
            cur_node_embed = tf.nn.l2_normalize(cur_node_embed, axis=1)
            cur_node_embed = tf.matmul(tf.concat([cur_node_embed, cur_node_embed_prev], axis=1), w_l)

        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim], dense
        y_potential = tf.sparse_tensor_dense_matmul(tf.cast(self.subgsum_param, tf.float32), cur_node_embed)
        # [batch_size, node_cnt] * [node_cnt, embed_dim] = [batch_size, embed_dim]
        action_embed = tf.sparse_tensor_dense_matmul(tf.cast(self.action_select, tf.float32), cur_node_embed)

        # concate and MLP
        ##[[batch_size, embed_dim], [batch_size, embed_dim]] = [batch_size, 2*embed_dim], dense
        embed_s_a = tf.concat([tf.matmul(y_potential, h11_weight), tf.matmul(action_embed, h12_weight)], 1)
        hidden1 = tf.nn.relu(embed_s_a)
        # [batch_size, embed_dim] * [embed_dim, reg_hidden] = [batch_size, reg_hidden]
        hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weight))
        # hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weight))
        last_output = hidden2
        # [batch_size, reg_hidden] * [reg_hidden, 1] = [batch_size, 1]
        q_pred = tf.matmul(last_output, last_w)

        if self.IsHuberLoss:
            loss = tf.losses.huber_loss(self.label, q_pred)
        else:
            loss = tf.losses.mean_squared_error(self.label, q_pred)

        trainStep = tf.train.AdamOptimizer(args.lr).minimize(loss)

        # [node_cnt, batch_size] * [batch_size, embed_dim] = [node_cnt, embed_dim]
        rep_y = tf.sparse_tensor_dense_matmul(tf.cast(self.rep_global, tf.float32), y_potential)

        # concate and MLP
        embed_s_a_all = tf.concat([tf.matmul(rep_y, h11_weight), tf.matmul(cur_node_embed, h12_weight)], 1)
        hidden1 = tf.nn.relu(embed_s_a_all)
        # [node_cnt, embed_dim] * [embed_dim, reg_hidden] = [node_cnt, reg_hidden]
        hidden2 = tf.nn.relu(tf.matmul(hidden1, h2_weight))
        last_output = hidden2
        # [node_cnt, reg_hidden] * [reg_hidden, 1] = [node_cnt, 1]
        q_on_all = tf.matmul(last_output, last_w)

        return loss, trainStep, q_pred, q_on_all, tf.trainable_variables()
    #pass
    def gen_graph(self, g_type):
        num = np.random.randint(args.lattice_num_min, args.lattice_num_max, 1)[0]
        if g_type == 'lattice':
            if args.lattice_dim == 2:
                G = nx.grid_graph(dim=[num, num], periodic=args.lattice_Periodic)
            elif args.lattice_dim == 3:
                G = nx.grid_graph(dim=[num, num, num], periodic=args.lattice_Periodic)
            elif args.lattice_dim == 4:
                G = nx.grid_graph(dim=[num, num, num, num], periodic=args.lattice_Periodic)
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
            g = nx.complete_graph(n=num)
        elif g_type == 'erdos_renyi':
            g = nx.erdos_renyi_graph(n=num, p=0.15)
        elif g_type == 'powerlaw':
            g = nx.powerlaw_cluster_graph(n=num, m=4, p=0.05)
        elif g_type == 'small-world':
            g = nx.connected_watts_strogatz_graph(n=num, k=8, p=0.1)
        elif g_type == 'barabasi_albert':
            g = nx.barabasi_albert_graph(n=num, m=4)

        for edge in nx.edges(g):
            if args.lattice_weight_distribution == 'normal':
                g[edge[0]][edge[1]]['weight'] = np.random.normal(0, 1)
            elif args.lattice_weight_distribution == 'uniform':
                g[edge[0]][edge[1]]['weight'] = np.random.uniform(-1, 1)
            elif args.lattice_weight_distribution == 'negative uniform':
                g[edge[0]][edge[1]]['weight'] = - np.random.uniform(0, 1)
            elif args.lattice_weight_distribution == 'bimodal':
                g[edge[0]][edge[1]]['weight'] = np.double(np.random.choice([-1.0,1.0], 1))

        for node in g.nodes():  # add node state, random
            # node state
            # g.nodes[node]['state'] = int(np.random.choice([-1,1], 1))
            g.nodes[node]['state'] = 1
            if g_type == 'lattice':
                if args.lattice_dim == 2:
                    x = int(int(node) / num)
                    y = int(int(node) % num)
                    g.nodes[node]['coords'] = (x, y)
                elif args.lattice_dim == 3:
                    x_y = int(int(node) % (num * num))
                    z = int(int(node) / (num * num))
                    x = int(x_y / num)
                    y = int(x_y % num)
                    g.nodes[node]['coords'] = (x, y, z)
                elif args.lattice_dim == 4:
                    w = int(node / (num*num*num))
                    x_y_z = int(node % (num*num*num))
                    z = int(x_y_z / (num*num))
                    x_y = int(x_y_z % (num*num))
                    y = int(x_y / num)
                    x = int(x_y % num)
                    g.nodes[node]['coords'] = (x, y, z, w)
        return g

    def gen_new_graphs(self):
        print('\ngenerating new training graphs...')
        sys.stdout.flush()
        self.ClearTrainGraphs()
        # cdef int i
        for i in tqdm(range(1000)):
            g = self.gen_graph(args.g_type)
            self.InsertGraph(g, is_test=False)
    #pass
    def ClearTrainGraphs(self):
        self.ngraph_train = 0
        self.TrainSet.Clear()

    def ClearTestGraphs(self):
        self.ngraph_test = 0
        self.TestSet.Clear()
    #pass
    def InsertGraph(self, g, is_test):
        if is_test:
            t = self.ngraph_test
            self.ngraph_test += 1
            self.TestSet.InsertGraph(t, self.GenNetwork(g))
        else:
            t = self.ngraph_train
            self.ngraph_train += 1
            self.TrainSet.InsertGraph(t, self.GenNetwork(g))

    def PrepareValidData(self):
        print('\ngenerating validation graphs...')
        self.gnx_valid_list = []
        self.bench_greedy = []
        for i in tqdm(range(args.n_valid)):
            # # # generate during this training
            g = self.gen_graph(args.g_type)
            self.gnx_valid_list.append(g)
            res_greedy = self.greedySearch(g)
            self.bench_greedy.append(res_greedy)
            self.InsertGraph(g, is_test=True)

    def Run_simulator(self, num_seq, eps, TrainSet, n_step):
        args.num_env = len(self.env_list)
        n = 0
        while n < num_seq:
            for i in range(args.num_env):
                if self.env_list[i].graph.num_nodes == 0 or self.env_list[i].isTerminal():
                    if self.env_list[i].graph.num_nodes > 0 and self.env_list[i].isTerminal():
                        n = n + 1
                        self.nStepReplayMem.Add(self.env_list[i], n_step)
                    self.env_list[i].s0(TrainSet.Sample())
                    self.g_list[i] = self.env_list[i].graph
                    self.covered[i] = self.env_list[i].action_list
            if n >= num_seq:
                break

            Random = False
            if random.uniform(0,1) >= eps:
                pred = self.Predict(self.g_list, [self.env_list[i].action_list],False)
            else:
                Random = True
            for i in range(args.num_env):
                if (Random):
                    a_t = self.env_list[i].randomAction()
                else:
                    a_t = np.argmax(pred[i])
                self.env_list[i].step(a_t)

    def PlayGame(self, n_traj, eps):
        self.Run_simulator(n_traj, eps, self.TrainSet, args.n_step)

    def SetupTrain(self, idxes, g_list, covered, actions, target):
        self.inputs['label'] = target
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(args.aggregatorID)
        prepareBatchGraph.SetupTrain(idxes, g_list, covered, actions, args.PE_dim)
        self.inputs['action_select'] = prepareBatchGraph.act_select
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2egsum_param'] = prepareBatchGraph.n2esum_param
        self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['edge_input'] = prepareBatchGraph.edge_feat

    def SetupPredAll(self, idxes, g_list, covered):
        prepareBatchGraph = PrepareBatchGraph.py_PrepareBatchGraph(args.aggregatorID)
        prepareBatchGraph.SetupPredAll(idxes, g_list, covered, args.PE_dim)
        self.inputs['rep_global'] = prepareBatchGraph.rep_global
        self.inputs['n2esum_param'] = prepareBatchGraph.n2esum_param
        self.inputs['e2nsum_param'] = prepareBatchGraph.e2nsum_param
        self.inputs['subgsum_param'] = prepareBatchGraph.subgsum_param
        self.inputs['node_input'] = prepareBatchGraph.node_feat
        self.inputs['edge_input'] = prepareBatchGraph.edge_feat

    def Predict(self, g_list, covered, isSnapSnot):
        n_graphs = len(g_list)
        for i in range(0, n_graphs, args.batch_size):
            bsize = args.batch_size
            if (i + args.batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):

                batch_idxes[j - i] = j
            batch_idxes = np.int32(batch_idxes)

            self.SetupPredAll(batch_idxes, g_list, covered)

            if isSnapSnot:
                result = self.session.run([self.q_on_allT], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2esum_param: self.inputs['n2esum_param'],
                    self.e2nsum_param: self.inputs['e2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.edge_input: self.inputs['edge_input']
                })
            else:
                result = self.session.run([self.q_on_all], feed_dict={
                    self.rep_global: self.inputs['rep_global'],
                    self.n2esum_param: self.inputs['n2esum_param'],
                    self.e2nsum_param: self.inputs['e2nsum_param'],
                    self.subgsum_param: self.inputs['subgsum_param'],
                    self.node_input: self.inputs['node_input'],
                    self.edge_input: self.inputs['edge_input']
                })
            raw_output = result[0]
            pos = 0
            pred = []
            for j in range(i, i + bsize):
                g = g_list[j]
                cur_pred = np.zeros(g.num_nodes)
                for k in range(g.num_nodes):
                    cur_pred[k] = raw_output[pos]
                    pos += 1
                for k in covered[j]:
                    cur_pred[k] = -args.inf
                pred.append(cur_pred)
            assert (pos == len(raw_output))
        return pred

    def PredictWithSnapshot(self, g_list, covered):
        result = self.Predict(g_list,covered,True)
        return result

    def TakeSnapShot(self):
       self.session.run(self.UpdateTargetQNetwork)

    def Fit(self, iter):
        sample = self.nStepReplayMem.Sampling(args.batch_size)
        ness = False
        for i in range(args.batch_size):
            if (not sample.list_term[i]):
                ness = True
                break
        if ness:
            if self.IsDoubleDQN:
                double_list_pred = self.Predict(sample.g_list, sample.list_s_primes, False)
                double_list_predT = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)
                list_pred = [a[np.argmax(b)] for a, b in zip(double_list_predT, double_list_pred)]
            else:
                list_pred = self.PredictWithSnapshot(sample.g_list, sample.list_s_primes)

        list_target = np.zeros([args.batch_size, 1])

        for i in range(args.batch_size):
            q_rhs = 0
            if (not sample.list_term[i]):
                q_rhs = args.gamma * np.max(list_pred[i])
            if q_rhs < 0:
                q_rhs = 0
            q_rhs += sample.list_rt[i]
            list_target[i] = q_rhs

        loss = self.fit(sample.g_list, sample.list_st, sample.list_at, list_target, iter)

        train_greedy, train_greedy_improve = 0.0, 0.0
        if iter % 300 == 0:
            for g in sample.g_list: #cython graph
                g_nx = self.DeGenNetwork(g)
                val, sol, _, _ = self.Evaluate(g, isNetworkx=0)
                val_improve = self.localImprove(g_nx, sol)
                val_greedy = self.greedySearch(g_nx)

                train_greedy += val / val_greedy / args.batch_size
                train_greedy_improve += val_improve / val_greedy / args.batch_size

        return loss, train_greedy, train_greedy_improve

    def fit(self, g_list, covered, actions, list_target, iter):
        loss = 0
        n_graphs = len(g_list)
        for i in range(0,n_graphs, args.batch_size):
            bsize = args.batch_size
            if (i + args.batch_size) > n_graphs:
                bsize = n_graphs - i
            batch_idxes = np.zeros(bsize)
            for j in range(i, i + bsize):
                batch_idxes[j - i] = j

            self.SetupTrain(np.int32(batch_idxes), g_list, covered, actions, list_target)

            result = self.session.run([self.loss, self.trainStep],feed_dict={
                                        self.action_select : self.inputs['action_select'],
                                        self.rep_global : self.inputs['rep_global'],
                                        self.n2esum_param: self.inputs['n2esum_param'],
                                        self.e2nsum_param: self.inputs['e2nsum_param'],
                                        self.subgsum_param: self.inputs['subgsum_param'],
                                        self.node_input: self.inputs['node_input'],
                                        self.edge_input: self.inputs['edge_input'],
                                        self.label : self.inputs['label']})
            loss += result[0]*bsize
        return loss / len(g_list)
    
    def Train(self):
        self.PrepareValidData()
        self.gen_new_graphs()
        for i in range(10):
            self.PlayGame(100, 1)
        self.TakeSnapShot()

        eps_start = 1
        eps_end = 0.05
        eps_step = 50000.0

        save_dir = './models/%s'%args.model_name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        VCFile = '%s/%s_ModelVC_%d_%d.csv'%(save_dir, args.model_name, args.lattice_num_min, args.lattice_num_max)
        f_out = open(VCFile, 'w')
        f_out.write('train_greedy, train_greedy_improve, valid_greedy, valid_greedy_improve\n')
        current_greedy_best, best_iter = 0.0, 0 # record the best result until now
        
        fout_loss = open('./models/Loss_%d_%d.csv'%(args.lattice_num_min, args.lattice_num_max), 'w')

        for iter in range(args.max_iteration):
            start = time.clock()
            if iter and iter % 5000 == 0:
                self.gen_new_graphs()
            eps = eps_end + max(0., (eps_start - eps_end) * (eps_step - iter) / eps_step)
            # play game
            if iter % 10 == 0:
                self.PlayGame(10, eps)
            # update params
            loss, train_greedy, train_greedy_improve = self.Fit(iter)
            fout_loss.write('iter: %d, loss: %.6f\n'%(iter, loss))
            fout_loss.flush()
            # validation test
            if iter % 300 == 0:
                if(iter == 0):
                    N_start = start
                else:
                    N_start = N_end
                test_start = time.time()
                valid_greedy, valid_greedy_improve, valid_gurobi, valid_gurobi_improve = 0.0, 0.0, 0.0, 0.0
                for idx in range(args.n_valid):
                    val, sol = self.TestNoStop(idx)
                    val_improve = self.localImprove(self.gnx_valid_list[idx], sol)

                    valid_greedy += val / self.bench_greedy[idx] / args.n_valid
                    valid_greedy_improve += val_improve / self.bench_greedy[idx] / args.n_valid
                test_end = time.time()
                f_out.write('%.4f, %.4f, %.4f, %.4f\n'%(train_greedy, train_greedy_improve, valid_greedy, valid_greedy_improve)) 
                f_out.flush()
                print('Iter:%d, eps:%.4f, Loss:%.6f'%(iter, eps, loss))
                print('train_greedy(improve):%.4f(%.4f), valid_greedy(improve):%.4f(%.4f)'%(train_greedy, train_greedy_improve, valid_greedy, valid_greedy_improve))
                # print the current best ratio result
                if iter > 5000 and current_greedy_best < valid_greedy:
                    current_greedy_best = valid_greedy
                    best_iter = iter
                print ('Current_greedy_best: %.4f, best_iter:%d'%(current_greedy_best, best_iter))
                print ('testing 100 graphs time: %.2fs'%(test_end-test_start))
                N_end = time.clock()
                print ('300 iterations total time: %.2fs\n'%(N_end-N_start))
                sys.stdout.flush()
                model_path = '%s/nrange_%d_%d_iter_%d.ckpt' % (save_dir, args.lattice_num_min, args.lattice_num_max, iter)
                self.SaveModel(model_path)
            # update target network
            if iter % args.update_time == 0:
                self.TakeSnapShot()
        f_out.close()
        fout_loss.close()

    def findModel(self):
        VCFile = './models/%s/%s_ModelVC_%d_%d.csv'%(args.model_name, args.model_name, args.lattice_num_min, args.lattice_num_max)
        vc_list = []
        num = 0
        for line in open(VCFile):
            if num > 0: #do not read the first line
                data = line.strip().split(',')
                vc_list.append(float(data[2]))
            num += 1
        start_loc = 33
        min_vc = start_loc + np.argmax(vc_list[start_loc:])
        best_model_iter = 300 * min_vc
        print ('best iter:%d'%min_vc)
        print ('best iter vc:%.4f'%vc_list[min_vc])
        print ('best vc:%.4f'%np.max(vc_list[start_loc:]))
        best_model = './models/%s/nrange_%d_%d_iter_%d.ckpt'%(args.model_name, args.lattice_num_min, args.lattice_num_max, best_model_iter)
        print ('best model')
        print (best_model)
        return best_model

    def TestNoStop(self, gid):
        g_list = []
        self.test_env.s0(self.TestSet.Get(gid))
        g_list.append(self.test_env.graph)
        initial = copy.deepcopy(self.test_env.graph.node_state) # initial node states
        norm = self.test_env.graph.num_edges
        init_energy = self.test_env.graph.init_energy
        cost = init_energy  # spin glass
        best, length = cost, 0
        best_len = -1
        sol = []
        while (not self.test_env.isTerminal()):
            list_pred = self.Predict(g_list, [self.test_env.action_list],False)
            scores = list_pred[0]
            new_action = np.argmax(list_pred[0])
            cost += self.test_env.step4reward(new_action) * norm
            length += 1
            sol.append(new_action)
            if cost > best:
                best = cost
                best_len = length
        if best_len > 0:
            sol = sol[0:best_len]   # set of nodes to be flipped
            current_node_states = [initial[i] if i not in sol else -1*initial[i] for i in range(len(initial))]  # final node states
        else:
            current_node_states = initial
        assert(len(current_node_states) == len(initial))
        return best/self.test_env.graph.num_nodes, current_node_states

    def Evaluate(self, g, isNetworkx=1, step=1, partialRate=1.0):
        if isNetworkx==1:  # input graph is a networkx graph
            num_node = len(g)
            self.ClearTestGraphs()
            self.InsertGraph(g, is_test=True)
            g_input = self.TestSet.Get(0)
        else:   # input graph is a Cython graph
            num_node = g.num_nodes
            g_input = g
        g_list = []
        self.test_env.s0(g_input)
        g_list.append(self.test_env.graph)

        initial = copy.deepcopy(self.test_env.graph.node_state) # initial node states
        norm = self.test_env.graph.num_edges
        init_energy = self.test_env.graph.init_energy

        cost = init_energy
        best, length = cost, 0
        best_len = -1
        sol, sol_all = [], []
        while (not self.test_env.isTerminal()):
            list_pred = self.Predict(g_list, [self.test_env.action_list], False)
            batchSol = np.argsort(-list_pred[0])[:step]
            for new_action in batchSol:
                if not self.test_env.isTerminal():
                    cost += self.test_env.step4reward(new_action)*norm
                    length += 1
                    sol_all.append(new_action)
                    if (cost > best):
                        best = cost
                        best_len = length
                else:
                    break

        if best_len > 0:
            sol = sol_all[0:max(1, int(partialRate*best_len))] 
            current_node_states = [initial[i] if i not in sol else -1*initial[i] for i in range(len(initial))]  # final node states
        else:
            best = init_energy
            current_node_states = initial
        assert(len(current_node_states) == len(initial))
        return best/num_node, current_node_states, sol, sol_all

    def SaveModel(self, model_path):
        self.saver.save(self.session, model_path)
        print('model has been saved success!')
    
    def LoadModel(self, model_path):
        self.saver.restore(self.session, model_path)
        print('restore model from file successfully')

    def greedySearch(self, g_input, isNetworkx=True):
        g = g_input.copy()
        energy = 0.0
        for edge in g.edges():
            energy += g.nodes[edge[0]]['state'] * g.nodes[edge[1]]['state'] * g[edge[0]][edge[1]]['weight']

        stopCondition = False   # for each state, run multiple times to obtain the local optimal
        while not stopCondition:
            best_delta, best_node = 0.0, -1
            for node in g.nodes():
                delta = 0
                for neigh in g.neighbors(node):
                    delta -= 2 * g.nodes[node]['state'] * g.nodes[neigh]['state'] * g[node][neigh]['weight']
                if delta >= 0 and best_delta < delta:
                    best_delta = delta
                    best_node = node
            if best_node > -1:
                g.nodes[best_node]['state'] *= -1
            energy += best_delta
            if best_delta == 0:
                stopCondition = True # no other flips can increase the energy
        return energy/len(g)

    def localImprove(self, g_origin, states):
        g = g_origin.copy()
        energy = 0.0
        for edge in g.edges():
            energy += states[edge[0]] * states[edge[1]] * g[edge[0]][edge[1]]['weight']
        stopCondition = False   # for each state, run multiple times to obtain the local optimal
        while not stopCondition:
            best_delta, best_node = 0.0, -1
            for node in g.nodes():
                delta = 0
                for neigh in g.neighbors(node):
                    delta -= 2 * states[node] * states[neigh] * g[node][neigh]['weight']
                if delta >= 0 and best_delta < delta:
                    best_delta = delta
                    best_node = node
            if best_node > -1:
                states[best_node] *= -1
            energy += best_delta
            if best_delta == 0:
                stopCondition = True # no other flips can increase the energy
        return energy/len(g)

    def GenNetwork(self, g):  # networkx2four
        edges = g.edges()
        weights = []
        node_states = []
        node_coords = []
        for edge in edges:
            weights.append(g[edge[0]][edge[1]]['weight'])
        for i in range(len(g.nodes())):
            node_states.append(g.nodes[i]['state'])
            for k in range(args.lattice_dim):
                node_coords.append(g.nodes[i]['coords'][k])
        if len(edges) > 0:
            a, b = zip(*edges)
            A = np.array(a)
            B = np.array(b)
            W = np.array(weights)
        else:
            A = np.array([0])
            B = np.array([0])
            W = np.array([0])
        return graph.py_Graph(len(g.nodes()), len(edges), A, B, W, node_states, node_coords, args.lattice_dim)

    def DeGenNetwork(self, g_c):  # four2networkx
        g = nx.Graph()
        num_nodes = g_c.num_nodes
        adj_list = g_c.adj_list
        for i in range(num_nodes):
            edge_src = i
            g.add_node(edge_src)
            g.nodes[edge_src]['state'] = g_c.node_state[i]
            for j in range(len(adj_list[i])):
                if adj_list[i][j][0] >= i:
                    edge_tgt = adj_list[i][j][0]
                    g.add_edge(edge_src, edge_tgt)
                    g[edge_src][edge_tgt]['weight'] = adj_list[i][j][1]
        return g
