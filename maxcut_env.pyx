from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import graph
from graph cimport Graph

cdef class py_MaxcutEnv:
    cdef shared_ptr[MaxcutEnv] inner_MaxcutEnv
    cdef shared_ptr[Graph] inner_Graph
    def __cinit__(self):
        self.inner_MaxcutEnv = shared_ptr[MaxcutEnv](new MaxcutEnv())
        self.inner_Graph =shared_ptr[Graph](new Graph())

    def s0(self,_g):
        deref(self.inner_Graph).num_nodes = _g.num_nodes
        deref(self.inner_Graph).num_edges = _g.num_edges
        deref(self.inner_Graph).adj_list = _g.adj_list
        deref(self.inner_Graph).total_abs_weight = _g.total_abs_weight
        deref(self.inner_Graph).total_weight = _g.total_weight
        deref(self.inner_Graph).node_state = _g.node_state
        deref(self.inner_Graph).node_coords = _g.node_coords
        deref(self.inner_Graph).node_coords_dim = _g.node_coords_dim
        deref(self.inner_Graph).init_energy = _g.init_energy
        deref(self.inner_MaxcutEnv).s0(self.inner_Graph)

    def step(self,int a):
        return deref(self.inner_MaxcutEnv).step(a)

    def step4reward(self,int a):
        return deref(self.inner_MaxcutEnv).step4reward(a)

    def randomAction(self):
        return deref(self.inner_MaxcutEnv).randomAction()

    def isTerminal(self):
        return deref(self.inner_MaxcutEnv).isTerminal()

    def getReward(self,double old_cutWeight):
        return deref(self.inner_MaxcutEnv).getReward(old_cutWeight)

    @property
    def graph(self):
        return self.G2P(deref(self.inner_Graph))
        # return self.G2P(deref(deref(self.inner_MaxcutEnv).graph))
    @property
    def cutWeight(self):
        return deref(self.inner_MaxcutEnv).cutWeight
    @property
    def cut_set(self):
        return deref(self.inner_MaxcutEnv).cut_set
    @property
    def avail_list(self):
        return deref(self.inner_MaxcutEnv).avail_list
    @property
    def norm(self):
        return deref(self.inner_MaxcutEnv).norm
    @property
    def state_seq(self):
        return deref(self.inner_MaxcutEnv).state_seq
    @property
    def act_seq(self):
        return deref(self.inner_MaxcutEnv).act_seq
    @property
    def action_list(self):
        return deref(self.inner_MaxcutEnv).action_list
    @property
    def reward_seq(self):
        return deref(self.inner_MaxcutEnv).reward_seq
    @property
    def sum_rewards(self):
        return deref(self.inner_MaxcutEnv).sum_rewards

    # cdef G2P(self,Graph g):
    #     num_nodes = g.num_nodes     #得到Graph对象的节点个数
    #     num_edges = g.num_edges    #得到Graph对象的连边个数
    #     adj_list = g.adj_list
    #     cint_edges_from = np.zeros([num_edges],dtype=np.int)
    #     cint_edges_to = np.zeros([num_edges],dtype=np.int)
    #     weights = np.zeros([num_edges],dtype=np.double)
    #     node_state = np.zeros([num_nodes],dtype=np.int)
    #
    #     cdef int k = 0
    #     for i in range(num_nodes):
    #         node_state[i] = g.node_state[i]
    #         for j in range(adj_list[i].size()):
    #             # k = k + 1
    #             if adj_list[i][j].first >= i:
    #                 cint_edges_from[k]= i
    #                 cint_edges_to[k] = adj_list[i][j].first
    #                 weights[k] = adj_list[i][j].second
    #                 k = k +1
    #     return graph.py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,weights,node_state)

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
