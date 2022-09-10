'''
#file:graph.pyx类graph的实现文件
#可以自动导入相同路径下相同名称的.pxd的文件
#可以省略cimport graph命令
#需要重新设计python调用的接口，此文件
'''
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np

cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph#使用unique_ptr优于shared_ptr
    #__cinit__会在__init__之前被调用
    def __cinit__(self,*arg):
        '''doing something before python calls the __init__.
        cdef 的C/C++对象必须在__cinit__里面完成初始化，否则没有为之分配内存
        可以接收参数，使用python的变参数模型实现类似函数重载的功能。'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        self.inner_graph = shared_ptr[Graph](new Graph())
        cdef int _num_nodes
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        cdef double[:] weights
        # cdef int[:] node_state
        # cdef int[:] node_coords
        cdef int node_coords_dim

        if len(arg)==0:
            #这两行代码为了防止内存没有初始化，没有实际意义
            deref(self.inner_graph).num_edges=0
            deref(self.inner_graph).num_nodes=0
        # elif len(arg)==5:
        #     _num_nodes=arg[0]
        #     _num_edges=arg[1]
        #     edges_from=np.array([int(x) for x in arg[2]], dtype=np.int32)
        #     edges_to=np.array([int(x) for x in arg[3]], dtype=np.int32)
        #     weights=np.array([x for x in arg[4]], dtype=np.double)
        #     node_state=np.array([1 for x in range(_num_nodes)], dtype=np.int32)   # constant states
        #     self.reshape_Graph(_num_nodes,  _num_edges,  edges_from, edges_to, weights, node_state)
        # elif len(arg)==6:
        #     _num_nodes=arg[0]
        #     _num_edges=arg[1]
        #     edges_from=np.array([int(x) for x in arg[2]], dtype=np.int32)
        #     edges_to=np.array([int(x) for x in arg[3]], dtype=np.int32)
        #     weights=np.array([x for x in arg[4]], dtype=np.double)
        #     node_state=np.array([x for x in arg[5]], dtype=np.int32)
        #     self.reshape_Graph(_num_nodes,  _num_edges,  edges_from, edges_to, weights, node_state)
        elif len(arg)==8:
            _num_nodes = arg[0]
            _num_edges = arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            weights = np.array([x for x in arg[4]], dtype=np.double)
            node_state = np.array([x for x in arg[5]], dtype=np.int32)
            node_coords = np.array([x for x in arg[6]], dtype=np.int32)
            node_coords_dim = arg[7]
            self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, weights, node_state, node_coords, node_coords_dim)
        else:
            print('Error：py_Graph类为被成功初始化，因为提供参数数目不匹配，参数个数为0或8。')

    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def total_abs_weight(self):
        return deref(self.inner_graph).total_abs_weight

    @property
    def total_weight(self):
        return deref(self.inner_graph).total_weight

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def node_state(self):
        return deref(self.inner_graph).node_state

    @property
    def node_coords(self):
        return deref(self.inner_graph).node_coords

    @property
    def node_coords_dim(self):
        return deref(self.inner_graph).node_coords_dim

    @property
    def init_energy(self):
        return deref(self.inner_graph).init_energy

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to, double[:] weights, int[:] node_state, int[:] node_coords, int node_coords_dim):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_node_state = <int*>malloc(_num_nodes*sizeof(int))
        cdef int *cint_node_coords = <int*>malloc(node_coords_dim*_num_nodes*sizeof(int))
        cdef double *cdouble_weights = <double*>malloc(_num_edges*sizeof(double))
        cdef int i
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
            cint_edges_to[i] = edges_to[i]
            cdouble_weights[i] = weights[i]

        cdef int j
        for j in range(_num_nodes):
            cint_node_state[j] = node_state[j]
            for k in range(node_coords_dim):
                cint_node_coords[j*node_coords_dim+k] = node_coords[j*node_coords_dim+k]

        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes, _num_edges, &cint_edges_from[0], &cint_edges_to[0], &cdouble_weights[0], &cint_node_state[0], &cint_node_coords[0], node_coords_dim))
        free(cint_edges_from)
        free(cint_edges_to)
        free(cdouble_weights)
        free(cint_node_state)
        free(cint_node_coords)

    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to,double[:] weights, int[:] node_state, int[:] node_coords, int _node_coords_dim):
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to, weights, node_state, node_coords, _node_coords_dim)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)

    def Sample(self):
        temp_innerGraph=deref(deref(self.inner_gset).Sample())   #得到了Graph 对象
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()   #得到了Graph 对象

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   #得到了Graph 对象
        return self.G2P(temp_innerGraph)


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
        return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to,weights, node_state, node_coords, node_coords_dim)


