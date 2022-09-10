
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.memory cimport shared_ptr
from libcpp cimport bool
from graph cimport Graph

cdef extern from "./src/lib/maxcut_env.h":
    cdef cppclass MaxcutEnv:
        MaxcutEnv()except+
        void s0(shared_ptr[Graph] _g)except+
        double step(int a)
        double step4reward(int a)
        int randomAction()except+
        bool isTerminal()except+
        double getReward(double old_cutWeight)except+

        shared_ptr[Graph] graph
        double cutWeight
        set[int] cut_set
        vector[int] avail_list
        double norm
        vector[vector[int]]  state_seq
        vector[int] act_seq
        vector[int] action_list
        vector[double] reward_seq
        vector[double] sum_rewards




