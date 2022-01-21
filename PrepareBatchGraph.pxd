from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from graph cimport Graph
cdef extern from "./src/lib/PrepareBatchGraph.h":
    cdef cppclass sparseMatrix:
        sparseMatrix()except+
        vector[int] rowIndex
        vector[int] colIndex
        vector[double] value
        int rowNum
        int colNum

    cdef cppclass PrepareBatchGraph:
        PrepareBatchGraph(int _aggregatorID)
        void SetupTrain(vector[int] idxes,vector[shared_ptr[Graph] ] g_list,vector[vector[int]] covered,const int* actions, int PE_dim)except+
        void SetupPredAll(vector[int] idxes,vector[shared_ptr[Graph] ] g_list,vector[vector[int]] covered, int PE_dim)except+
        shared_ptr[sparseMatrix] act_select
        shared_ptr[sparseMatrix] rep_global
        shared_ptr[sparseMatrix] n2nsum_param
        shared_ptr[sparseMatrix] n2esum_param
        shared_ptr[sparseMatrix] e2nsum_param
        shared_ptr[sparseMatrix] subgsum_param
        vector[vector[double]]  edge_feat
        vector[vector[double]]  node_feat
        int aggregatorID
