#ifndef MCSTEP_H
#define MCSTEP_H
#include "graph.h"
#include <random>
#include <vector>

//SA��.
class SA {
public:
	SA(std::shared_ptr<Graph> _graph, const int* _node_state, double _beta);	//���캯��
    std::vector<int> lowest_states;													//��������Ľڵ�״̬
	double lowest_energy;
	void Run();
	std::vector<int> states;												//�ڵ�״̬(������ʱһ��Ҫ��ȷ��ά��!!!!!!)
//	void Print_Graph();															//��ʼ��������
private:
	double energy;																//ϵͳ��ǰ����(������ʱһ��Ҫ��ȷ��ά��!!!!!!)
	double beta;												//�������
	std::map<std::pair<int, int>, double> linkenergies;
	std::shared_ptr<Graph> graph;
};

#endif