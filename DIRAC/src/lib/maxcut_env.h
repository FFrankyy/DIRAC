#ifndef MVC_ENV_H
#define MVC_ENV_H

#include <vector>
#include <set>
#include "graph.h"

class MaxcutEnv
{
public:
    MaxcutEnv();

    void s0(std::shared_ptr<Graph> _g);

    double step(int a);
    double step4reward(int a);

    int randomAction();

    bool isTerminal();

    double getReward(double old_cutWeight);

    std::shared_ptr<Graph> graph;

    double cutWeight;

    std::set<int> cut_set;

    std::vector<int> avail_list;

    double norm;

    std::vector< std::vector<int> > state_seq;

    std::vector<int> act_seq, action_list;

    std::vector<double> reward_seq, sum_rewards;

};

#endif