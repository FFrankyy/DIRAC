from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.pair cimport pair
from libcpp.map cimport map
from graph cimport Graph
cdef extern from "./src/lib/mcstep.h":
    cdef cppclass SA:
        SA(shared_ptr[Graph] _graph, const int* _states, double _beta)except+
        void Run()except+
        # void Print_Graph()except+
        vector[int] lowest_states
        double lowest_energy
        vector[int] states

