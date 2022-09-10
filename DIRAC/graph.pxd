'''
file:graph.pxd 类graph的定义文件对应graph.h
'''
#Cython已经编译了C++的std模板库，位置在~/Cython/Includes/lincpp/
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.map cimport map
from libcpp.pair cimport pair
cdef extern from "./src/lib/graph.h":
    cdef cppclass Graph:
        Graph()except+
        Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const double* weights, const int* node_state, const int* node_coords, const int node_coords_dim) except+
        int num_nodes
        int num_edges
        double total_abs_weight
        double total_weight
        vector[vector[pair[int,double]]] adj_list
        double init_energy
        vector[int] node_state
        vector[int] node_coords
        int node_coords_dim

cdef extern from "./src/lib/graph.h":
    cdef cppclass GSet:
        GSet()except+
        void InsertGraph(int gid, shared_ptr[Graph] graph)except+
        shared_ptr[Graph] Sample()except+
        void Clear()except+
        shared_ptr[Graph] Get(int gid)except+
        map[int, shared_ptr[Graph]] graph_pool

