from cython.operator import dereference as deref
from libcpp.memory cimport shared_ptr
import numpy as np
import scipy.sparse as sp
import graph
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from graph cimport Graph
import tensorflow as tf

from scipy.sparse import coo_matrix
cdef class py_sparseMatrix:
    cdef shared_ptr[sparseMatrix] inner_sparseMatrix
    def __cinit__(self):
        self.inner_sparseMatrix =shared_ptr[sparseMatrix](new sparseMatrix())

    @property
    def rowIndex(self):
        return deref(self.inner_sparseMatrix).rowIndex
    @property
    def colIndex(self):
        return deref(self.inner_sparseMatrix).colIndex
    @property
    def value(self):
        return deref(self.inner_sparseMatrix).value
    @property
    def rowNum(self):
        return deref(self.inner_sparseMatrix).rowNum
    @property
    def colNum(self):
        return deref(self.inner_sparseMatrix).colNum


cdef class py_PrepareBatchGraph:
    cdef shared_ptr[PrepareBatchGraph] inner_PrepareBatchGraph
    cdef sparseMatrix matrix
    def __cinit__(self,int _aggregatorID):
        self.inner_PrepareBatchGraph =shared_ptr[PrepareBatchGraph](new PrepareBatchGraph(_aggregatorID))

    def SetupTrain(self, idxes, g_list, covered, list actions, PE_dim):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
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
            inner_glist.push_back(inner_Graph)

        cdef int *refint = <int*>malloc(len(actions)*sizeof(int))
        cdef int i
        for i in range(len(actions)):
            refint[i] = actions[i]
        deref(self.inner_PrepareBatchGraph).SetupTrain(idxes, inner_glist, covered, refint, PE_dim)
        free(refint)

    def SetupPredAll(self, idxes, g_list, covered, PE_dim):
        cdef shared_ptr[Graph] inner_Graph
        cdef vector[shared_ptr[Graph]] inner_glist
        for _g in g_list:
            # inner_glist.push_back(_g.inner_Graph)
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
            inner_glist.push_back(inner_Graph)
        deref(self.inner_PrepareBatchGraph).SetupPredAll(idxes, inner_glist, covered, PE_dim)

    @property
    def act_select(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).act_select)
        return self.ConvertSparseToTensor(self.matrix)


    @property
    def rep_global(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).rep_global)
        return self.ConvertSparseToTensor(self.matrix)

    @property
    def n2nsum_param(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).n2nsum_param)
        return self.ConvertSparseToTensor(self.matrix)

    @property
    def n2esum_param(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).n2esum_param)
        return self.ConvertSparseToTensor(self.matrix)

    @property
    def e2nsum_param(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).e2nsum_param)
        return self.ConvertSparseToTensor(self.matrix)

    @property
    def subgsum_param(self):
        self.matrix = deref(deref(self.inner_PrepareBatchGraph).subgsum_param)
        return self.ConvertSparseToTensor(self.matrix)

    @property
    def edge_feat(self):
        return deref(self.inner_PrepareBatchGraph).edge_feat

    @property
    def node_feat(self):
        return deref(self.inner_PrepareBatchGraph).node_feat

    cdef ConvertSparseToTensor(self,sparseMatrix matrix):
        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()
        return tf.SparseTensorValue(indices, data, (rowNum,colNum))








