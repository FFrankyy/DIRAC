#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<set>
#include<numeric>
#include<map>
#include<cmath>
#include "mcstep.h"
#include "graph.h"
using namespace std;

//参数:1.要读取的文件名;2.初始状态;3.逆温度(数组);4.Sweep次数.
//会读取文件里的图并且将能量,状态和温度等初始化.
SA::SA(std::shared_ptr<Graph> _graph, const int* _node_state, double _beta)
{
    int num_node = _graph->num_nodes;
    graph = _graph;
    states.resize(num_node);
	beta = _beta;

    for (int i = 0; i < num_node; ++i)
    {
        states[i] = _node_state[i];
    }

    for(int i =0;i<num_node;i++)
    {
        for (auto& neigh : graph->adj_list[i])
        {
            linkenergies[pair<int, int>{i, neigh.first}] = -neigh.second * states[i] * states[neigh.first];
        }
    }

	energy = 0;
	auto iter = linkenergies.begin();
	for (;iter!=linkenergies.end();iter++)
		energy += iter->second;
	energy /= 2;
	lowest_energy = energy;
	lowest_states = states;
//	current_states = states;
	//Print_Info("test.txt");
}

//void SA::Print_Graph()
//{
//     for(int i =0;i<num_node;i++)
//     {
//         cout << "Node: " << i << ", Neighbors: ";
//               for (auto& neigh : graph->adj_list[i]) {
//                cout << neigh.first << ":" << neigh.second << "\t";
//              }
//        cout << endl;
//     }
//}

//运行SA.运行完毕后最重要的结果当然是SA类里的lowest_energy和lowest_state.
//如果要在中途引用其他代码对系统状态进行变更,一定要正确地维护linkenergies,energy,states!!!!!!!!!维护方式详细见下.
void SA::Run()
{
    for (int step = 0; step < graph->num_nodes; step++)
    {
        int rnindex = rand() % graph->num_nodes;
//        cout << rnindex << "\t"<<endl;
        double node_egy = 0;
        for (auto& neib : graph->adj_list[rnindex])
        {
            node_egy += -2*linkenergies[pair<int, int>{rnindex, neib.first}];
//            node_egy += linkenergies[pair<int, int>{neib.first, rnindex}];
        }

        if (node_egy<0 || exp(-beta*node_egy) > rand()/double(RAND_MAX))
        {		//当判定该翻转某节点时:
            states[rnindex] *= -1;												//首先,更新该节点的状态
            for (auto neib : graph->adj_list[rnindex])
            {
                linkenergies[pair<int, int>{rnindex, neib.first}] *= -1;				//其次,该节点相连边的能量都都需要更新.
                linkenergies[pair<int, int>{neib.first, rnindex}] *= -1;
            }
            energy += node_egy;												//最后,由于node_egy变为-node_egy,系统能量需要更新.
        }
        if (energy < lowest_energy)
        {										//当然,最小能量和状态也需要维护.
            lowest_energy = energy;
            lowest_states = states;
        }
    }
}

