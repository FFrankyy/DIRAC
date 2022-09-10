from cython.operator import dereference as deref
import numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport malloc
from libc.stdlib cimport free
from libcpp.memory cimport shared_ptr
from graph cimport Graph

cdef class py_SA:
    cdef shared_ptr[SA] inner_sa#使用unique_ptr优于shared_ptr
    #__cinit__会在__init__之前被调用
    def __cinit__(self,*arg):
        '''doing something before python calls the __init__.
        cdef 的C/C++对象必须在__cinit__里面完成初始化，否则没有为之分配内存
        可以接收参数，使用python的变参数模型实现类似函数重载的功能。'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        _g = arg[0]
        cdef shared_ptr[Graph] inner_Graph
        inner_Graph = shared_ptr[Graph](new Graph())
        deref(inner_Graph).num_nodes = _g.num_nodes
        deref(inner_Graph).num_edges = _g.num_edges
        deref(inner_Graph).node_state = _g.node_state
        deref(inner_Graph).node_coords = _g.node_coords
        deref(inner_Graph).node_coords_dim = _g.node_coords_dim
        deref(inner_Graph).init_energy = _g.init_energy
        deref(inner_Graph).adj_list = _g.adj_list
        deref(inner_Graph).total_weight = _g.total_weight
        deref(inner_Graph).total_abs_weight = _g.total_abs_weight

        node_state = np.array([x for x in arg[1]], dtype=np.int32)

        cdef int j
        cdef int *cint_node_state = <int*>malloc(_g.num_nodes*sizeof(int))
        for j in range(_g.num_nodes):
            cint_node_state[j] = node_state[j]
        beta = arg[2]
        self.inner_sa = shared_ptr[SA](new SA(inner_Graph, &cint_node_state[0], beta))
        free(cint_node_state)

    def Run(self):
        return deref(self.inner_sa).Run()

    # def Print_Graph(self):
    #     return deref(self.inner_sa).Print_Graph()

    @property
    def lowest_states(self):
        return deref(self.inner_sa).lowest_states

    @property
    def states(self):
        return deref(self.inner_sa).states

    @property
    def lowest_energy(self):
        return deref(self.inner_sa).lowest_energy