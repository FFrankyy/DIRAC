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

//����:1.Ҫ��ȡ���ļ���;2.��ʼ״̬;3.���¶�(����);4.Sweep����.
//���ȡ�ļ����ͼ���ҽ�����,״̬���¶ȵȳ�ʼ��.
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

//����SA.������Ϻ�����Ҫ�Ľ����Ȼ��SA�����lowest_energy��lowest_state.
//���Ҫ����;�������������ϵͳ״̬���б��,һ��Ҫ��ȷ��ά��linkenergies,energy,states!!!!!!!!!ά����ʽ��ϸ����.
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
        {		//���ж��÷�תĳ�ڵ�ʱ:
            states[rnindex] *= -1;												//����,���¸ýڵ��״̬
            for (auto neib : graph->adj_list[rnindex])
            {
                linkenergies[pair<int, int>{rnindex, neib.first}] *= -1;				//���,�ýڵ������ߵ�����������Ҫ����.
                linkenergies[pair<int, int>{neib.first, rnindex}] *= -1;
            }
            energy += node_egy;												//���,����node_egy��Ϊ-node_egy,ϵͳ������Ҫ����.
        }
        if (energy < lowest_energy)
        {										//��Ȼ,��С������״̬Ҳ��Ҫά��.
            lowest_energy = energy;
            lowest_states = states;
        }
    }
}

