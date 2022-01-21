#include "graph.h"
#include <cassert>
#include <iostream>
#include <random>

Graph::Graph() : num_nodes(0), num_edges(0), node_coords_dim(0)
{
    adj_list.clear();
    total_abs_weight = 0.0;
    total_weight = 0.0;
    init_energy = 0.0;
    node_state.clear();
    node_coords.clear();
}

Graph::Graph(const int _num_nodes, const int _num_edges, const int* edges_from, const int* edges_to,const double* weights, const int* _node_state, const int* _node_coords, const int _node_coords_dim)
        : num_nodes(_num_nodes), num_edges(_num_edges), node_coords_dim(_node_coords_dim)
{
    total_abs_weight = 0.0;
    total_weight = 0.0;
    adj_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        adj_list[i].clear();

    init_energy = 0.0;
    node_state.resize(num_nodes);
    node_coords.resize(num_nodes * node_coords_dim);
    for (int i = 0; i < num_nodes; ++i)
    {
        adj_list[i].clear();
        node_state[i] = _node_state[i];

        for (int k = 0; k < node_coords_dim; ++k)
        {
            node_coords[i*node_coords_dim+k] = _node_coords[i*node_coords_dim+k];
        }
    }

    for (int i = 0; i < num_edges; ++i)
    {
        int x = edges_from[i], y = edges_to[i];
        double w = weights[i];
        init_energy += node_state[x] * node_state[y] * w;
        total_abs_weight += abs(w);
        total_weight += w;
        adj_list[x].push_back( std::make_pair(y, w) );
        adj_list[y].push_back( std::make_pair(x, w) );
    }
}

GSet::GSet()
{
    graph_pool.clear();
}
void GSet::Clear()
{
    graph_pool.clear();
}

void GSet::InsertGraph(int gid, std::shared_ptr<Graph> graph)
{
    assert(graph_pool.count(gid) == 0);

    graph_pool[gid] = graph;
}

std::shared_ptr<Graph> GSet::Get(int gid)
{
    assert(graph_pool.count(gid));
    return graph_pool[gid];
}

std::shared_ptr<Graph> GSet::Sample()
{
    assert(graph_pool.size());
    int gid = rand() % graph_pool.size();
    assert(graph_pool[gid]);
    return graph_pool[gid];
}

GSet GSetTrain, GSetTest;