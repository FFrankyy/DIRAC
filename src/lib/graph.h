#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
class Graph
{
public:
    Graph();
    Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to, const double* weights, const int* _node_state, const int* _node_coords, const int _node_coords_dim);
    int num_nodes;
    int num_edges;
    double total_abs_weight;
    double total_weight;
    std::vector<int> node_state;
    std::vector<int> node_coords;
    double init_energy;
    int node_coords_dim;

    std::vector< std::vector< std::pair<int, double> > > adj_list;
};

class GSet
{
public:
    GSet();
    void InsertGraph(int gid, std::shared_ptr<Graph> graph);
    std::shared_ptr<Graph> Sample();
    std::shared_ptr<Graph> Get(int gid);
    void Clear();
    std::map<int, std::shared_ptr<Graph> > graph_pool;
};

extern GSet GSetTrain;
extern GSet GSetTest;

#endif