from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp cimport bool
import graph
import numpy as np
cdef class py_ReplaySample:
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    def __cinit__(self):
        self.inner_ReplaySample = shared_ptr[ReplaySample](new ReplaySample())
    @property
    def g_list(self):
        result = []
        for graphPtr in deref(self.inner_ReplaySample).g_list:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def list_st(self):
        return deref(self.inner_ReplaySample).list_st
    @property
    def list_s_primes(self):
        return deref(self.inner_ReplaySample).list_s_primes
    @property
    def list_at(self):
        return deref(self.inner_ReplaySample).list_at
    @property
    def list_rt(self):
        return deref(self.inner_ReplaySample).list_rt
    @property
    def list_term(self):
        return deref(self.inner_ReplaySample).list_term

    cdef G2P(self, Graph g):
        num_nodes = g.num_nodes     #得到Graph对象的节点个数
        num_edges = g.num_edges    #得到Graph对象的连边个数
        adj_list = g.adj_list
        node_coords_dim = g.node_coords_dim
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        weights = np.zeros([num_edges],dtype=np.double)
        node_state = np.zeros([num_nodes],dtype=np.int)
        node_coords = np.zeros([num_nodes*node_coords_dim], dtype=np.int)

        cdef int k = 0
        for i in range(num_nodes):
            node_state[i] = g.node_state[i]
            for l in range(node_coords_dim):
                node_coords[i*node_coords_dim+l] = g.node_coords[i*node_coords_dim+l]
            for j in range(adj_list[i].size()):
                # k = k + 1
                if adj_list[i][j].first >= i:
                    cint_edges_from[k]= i
                    cint_edges_to[k] = adj_list[i][j].first
                    weights[k] = adj_list[i][j].second
                    k = k +1
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,weights, node_state, node_coords, node_coords_dim)

cdef class py_NStepReplayMem:
    cdef shared_ptr[NStepReplayMem] inner_NStepReplayMem#使用unique_ptr优于shared_ptr
    cdef shared_ptr[Graph] inner_Graph
    cdef shared_ptr[MaxcutEnv] inner_MaxcutEnv
    cdef shared_ptr[ReplaySample] inner_ReplaySample
    #__cinit__会在__init__之前被调用
    def __cinit__(self,int memory_size):
        '''默认构造函数，暂不调用Graph的默认构造函数，
        默认构造函数在栈上分配的内存读写速度比较快，
        但实际情况下网络的结构一旦变化就要重新在堆上创建对象，因此基本上栈上分配的内存不会被使用
        除非将类的实现文件重写，加入python的调用接口，否则无法避免在堆上创建对象'''
        #print('默认构造函数。')
        self.inner_NStepReplayMem = shared_ptr[NStepReplayMem](new NStepReplayMem(memory_size))

    def Add(self,maxcut_env,int nstep):
        self.inner_Graph =shared_ptr[Graph](new Graph())
        deref(self.inner_Graph).num_nodes=maxcut_env.graph.num_nodes
        deref(self.inner_Graph).num_edges=maxcut_env.graph.num_edges
        deref(self.inner_Graph).adj_list = maxcut_env.graph.adj_list
        deref(self.inner_Graph).node_state = maxcut_env.graph.node_state
        deref(self.inner_Graph).node_coords = maxcut_env.graph.node_coords
        deref(self.inner_Graph).node_coords_dim = maxcut_env.graph.node_coords_dim
        self.inner_MaxcutEnv = shared_ptr[MaxcutEnv](new MaxcutEnv())
        deref(self.inner_MaxcutEnv).graph = self.inner_Graph
        deref(self.inner_MaxcutEnv).cutWeight = maxcut_env.cutWeight
        deref(self.inner_MaxcutEnv).cut_set = maxcut_env.cut_set
        deref(self.inner_MaxcutEnv).avail_list = maxcut_env.avail_list
        deref(self.inner_MaxcutEnv).norm = maxcut_env.norm
        deref(self.inner_MaxcutEnv).state_seq = maxcut_env.state_seq
        deref(self.inner_MaxcutEnv).act_seq = maxcut_env.act_seq
        deref(self.inner_MaxcutEnv).action_list = maxcut_env.action_list
        deref(self.inner_MaxcutEnv).reward_seq = maxcut_env.reward_seq
        deref(self.inner_MaxcutEnv).sum_rewards = maxcut_env.sum_rewards
        deref(self.inner_NStepReplayMem).Add(self.inner_MaxcutEnv,nstep)
    def Sampling(self,batch_size):
        self.inner_ReplaySample =  deref(self.inner_NStepReplayMem).Sampling(batch_size)
        result=py_ReplaySample()
        result.inner_ReplaySample=self.inner_ReplaySample
        return result

    @property
    def graphs(self):
        result = []
        for graphPtr in deref(self.inner_NStepReplayMem).graphs:
            result.append(self.G2P(deref(graphPtr)))
        return  result
    @property
    def actions(self):
        return deref(self.inner_NStepReplayMem).actions
    @property
    def rewards(self):
        return deref(self.inner_NStepReplayMem).rewards
    @property
    def states(self):
        return deref(self.inner_NStepReplayMem).states
    @property
    def s_primes(self):
        return deref(self.inner_NStepReplayMem).s_primes
    @property
    def terminals(self):
        return deref(self.inner_NStepReplayMem).terminals
    @property
    def current(self):
        return deref(self.inner_NStepReplayMem).current
    @property
    def count(self):
        return deref(self.inner_NStepReplayMem).count
    @property
    def memory_size(self):
        return deref(self.inner_NStepReplayMem).memory_size

    cdef G2P(self, Graph g):
        num_nodes = g.num_nodes     #得到Graph对象的节点个数
        num_edges = g.num_edges    #得到Graph对象的连边个数
        adj_list = g.adj_list
        node_coords_dim = g.node_coords_dim
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        weights = np.zeros([num_edges],dtype=np.double)
        node_state = np.zeros([num_nodes],dtype=np.int)
        node_coords = np.zeros([num_nodes*node_coords_dim], dtype=np.int)

        cdef int k = 0
        for i in range(num_nodes):
            node_state[i] = g.node_state[i]
            for l in range(node_coords_dim):
                node_coords[i*node_coords_dim+l] = g.node_coords[i*node_coords_dim+l]
            for j in range(adj_list[i].size()):
                # k = k + 1
                if adj_list[i][j].first >= i:
                    cint_edges_from[k]= i
                    cint_edges_to[k] = adj_list[i][j].first
                    weights[k] = adj_list[i][j].second
                    k = k +1
        return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,weights, node_state, node_coords, node_coords_dim)
