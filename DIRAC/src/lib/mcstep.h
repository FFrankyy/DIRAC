#ifndef MCSTEP_H
#define MCSTEP_H
#include "graph.h"
#include <random>
#include <vector>

//SA类.
class SA {
public:
	SA(std::shared_ptr<Graph> _graph, const int* _node_state, double _beta);	//构造函数
    std::vector<int> lowest_states;													//最低能量的节点状态
	double lowest_energy;
	void Run();
	std::vector<int> states;												//节点状态(在运行时一定要正确地维护!!!!!!)
//	void Print_Graph();															//初始化后运行
private:
	double energy;																//系统当前能量(在运行时一定要正确地维护!!!!!!)
	double beta;												//最低能量
	std::map<std::pair<int, int>, double> linkenergies;
	std::shared_ptr<Graph> graph;
};

#endif