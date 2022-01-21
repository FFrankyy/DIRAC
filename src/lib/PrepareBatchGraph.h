#ifndef PREPAREBATCHGRAPH_H_
#define PREPAREBATCHGRAPH_H_

#include "graph.h"
#include "graph_struct.h"
#include <random>
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <set>
#include <math.h>


class sparseMatrix
{
 public:
    sparseMatrix();
    std::vector<int> rowIndex;
    std::vector<int> colIndex;
    std::vector<double> value;
    int rowNum;
    int colNum;
};

class PrepareBatchGraph{
public:
    PrepareBatchGraph(int _aggregatorID);
    void SetupGraphInput(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions,
                         int PE_dim);
    void SetupTrain(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         const int* actions,
                         int PE_dim);
    void SetupPredAll(std::vector<int> idxes,
                         std::vector< std::shared_ptr<Graph> > g_list,
                         std::vector< std::vector<int> > covered,
                         int PE_dim);
    std::shared_ptr<sparseMatrix> act_select;
    std::shared_ptr<sparseMatrix> rep_global;
    std::shared_ptr<sparseMatrix> n2nsum_param;
    std::shared_ptr<sparseMatrix> subgsum_param;
    std::shared_ptr<sparseMatrix> e2nsum_param;
    std::shared_ptr<sparseMatrix> n2esum_param;
    std::vector< std::vector<double> > edge_feat;
    std::vector< std::vector<double> > node_feat;

    GraphStruct graph;
    int aggregatorID;
};

std::shared_ptr<sparseMatrix> n2n_construct(GraphStruct* graph,int aggregatorID);

std::shared_ptr<sparseMatrix> e2n_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> n2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> e2e_construct(GraphStruct* graph);

std::shared_ptr<sparseMatrix> subg_construct(GraphStruct* graph);
#endif